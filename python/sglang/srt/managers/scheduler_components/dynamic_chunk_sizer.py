from __future__ import annotations

import logging
import math
import time
from array import array
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    is_dp_attention_enabled,
    set_is_extend_in_batch,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_components.pp_chunk_model_update import (
    PPChunkModelUpdate,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import broadcast_pyobj
from sglang.srt.utils.common import get_device_module

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


class DynamicChunkSizer:
    """Sizes PP prefill chunks from a profiled quadratic latency model."""

    def __init__(
        self,
        *,
        model_runner: ModelRunner,
        model_config: ModelConfig,
        tree_cache: BasePrefixCache,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        spec_algorithm: SpeculativeAlgorithm,
        chunked_prefill_size: int,
        max_prefill_tokens: int,
        page_size: int,
        device: str,
    ):
        self.model_runner = model_runner
        self.model_config = model_config
        self.tree_cache = tree_cache
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.spec_algorithm = spec_algorithm
        self.chunked_prefill_size = chunked_prefill_size
        self.max_prefill_tokens = max_prefill_tokens
        self.page_size = page_size
        self.device = device
        parallel = get_parallel()
        self.pp_rank = parallel.pp_rank
        # Only one rank per PP stage owns the online model; the rest receive the
        # update through the existing in-stage request broadcast.
        self.is_pp_chunk_leader = (
            self.pp_rank == 0
            and parallel.attn_tp_rank == 0
            and parallel.attn_cp_rank == 0
        )
        self.predictor = ChunkSizePredictor()

        # Online refinement state. Only the first PP rank times forwards. The
        # fitted state travels stage to stage with the existing PP request
        # sends, so chunk decisions stay identical without a new collective.
        self.min_fit_chunk = envs.SGLANG_DYNAMIC_CHUNKING_ONLINE_MIN_FIT_CHUNK.get()
        self.max_fit_chunk = envs.SGLANG_DYNAMIC_CHUNKING_ONLINE_MAX_FIT_CHUNK.get()
        self.target_latency_samples = (
            envs.SGLANG_DYNAMIC_CHUNKING_ONLINE_TARGET_LATENCY_SAMPLES.get()
        )
        self.online_deadline_s = envs.SGLANG_DYNAMIC_CHUNKING_ONLINE_DEADLINE_S.get()
        self.online_enabled = envs.SGLANG_DYNAMIC_CHUNKING_ONLINE_ENABLE.get()
        self.online_active = self.online_enabled
        self._online_started_at = time.monotonic()
        self._target_latency_sample_count = 0
        self._target_latency_done = not self.online_active
        self._model_version = 0
        self.chunked_fit_data: List[List[float]] = []

    def profile_and_fit(self) -> bool:
        """PP0 profiles synthetic prefills and every rank fits the same samples;
        returns whether the predictor is ready."""
        samples: Optional[Tuple[List[int], List[float]]] = None

        parallel = get_parallel()
        if parallel.pp_group.is_first_rank:
            try:
                samples = self._profile_prefill_latency()
            except Exception as e:
                logger.warning(
                    f"[PP Dynamic Chunk] Failed to profile prefill latency: {e!r}. "
                    "Dynamic chunking will be disabled."
                )

        # The samples are global, so one broadcast from global rank 0 (a PP0 rank)
        # reaches every stage and attention rank; a failure travels as None.
        world_group = parallel.world_group
        samples = broadcast_pyobj(
            [samples], world_group.rank, world_group.cpu_group, src=0
        )[0]

        if samples is None:
            return False

        seq_lens, latencies = samples
        # Quadratic model: f(l) = al^2 + bl + c
        try:
            self.predictor.fit(seq_lens, latencies)
        except Exception as e:
            # Every rank fits the same samples, so this fails on all of them alike.
            logger.warning(
                f"[PP Dynamic Chunk] Failed to fit the chunk-size predictor: {e!r}. "
                "Dynamic chunking will be disabled."
            )
            return False
        self.predictor.set_target_latency(self.chunked_prefill_size)
        self.predictor.is_ready = True
        self._online_started_at = time.monotonic()
        logger.info(
            f"[PP Dynamic Chunk] [PP{self.pp_rank}] Predictor ready (quadratic). "
            f"Target latency: {self.predictor.target_latency:.2f}ms, "
            f"online calibration: {self.online_active}"
        )
        return True

    def predict(self, history_len: int) -> Optional[int]:
        """Chunk size for the next prefill step, or None to keep the static size."""
        if not self.predictor.is_ready:
            return None

        max_chunk_size = self.max_prefill_tokens
        use_history = self.predictor.with_history_ready
        predicted_size = self.predictor.predict_next_chunk_size(
            history_len=history_len,
            base_chunk_size=self.chunked_prefill_size,
            page_size=self.page_size,
            context_len=self.model_config.context_len,
            max_chunk_size=max_chunk_size,
            use_history=use_history,
        )

        if predicted_size is not None:
            logger.debug(
                f"[PP Dynamic Chunk] [PP{self.pp_rank}] Predicted chunk size: "
                f"{predicted_size} (history_len={history_len}, "
                f"history_model={use_history})"
            )

        return predicted_size

    # ------------------------------------------------------------------
    # Online calibration
    # ------------------------------------------------------------------

    def disable_online_calibration(self) -> None:
        """Keep the startup-fitted model (e.g. for event loops without update transport)."""
        self.online_enabled = False
        self.online_active = False
        self._target_latency_done = True

    def _mark_model_updated(self) -> None:
        self._model_version += 1

    def _maybe_expire_online(self) -> None:
        if (
            self.is_pp_chunk_leader
            and self.online_active
            and self.online_deadline_s > 0
            and time.monotonic() - self._online_started_at >= self.online_deadline_s
        ):
            self.online_active = False
            self._target_latency_done = True
            self._mark_model_updated()
            logger.info(
                "[PP Dynamic Chunk] Online calibration deadline reached; "
                "keeping the fitted model."
            )

    def export_online_update(self) -> Optional[PPChunkModelUpdate]:
        """Current model state to attach to this stage's outgoing request send."""
        if not self.online_enabled or not self.predictor.is_ready:
            return None
        self._maybe_expire_online()
        return PPChunkModelUpdate(
            version=self._model_version,
            quadratic_chunk_a=self.predictor.quadratic_chunk_a,
            linear_chunk_b=self.predictor.linear_chunk_b,
            constant_chunk_c=self.predictor.constant_chunk_c,
            with_history_ready=self.predictor.with_history_ready,
            target_latency=self.predictor.target_latency,
            online_active=self.online_active,
            origin_dp_rank=get_parallel().attn_dp_rank,
        )

    def apply_online_update(self, update: PPChunkModelUpdate) -> None:
        """Apply a newer model state received from the previous PP stage."""
        if update.origin_dp_rank != get_parallel().attn_dp_rank:
            return
        if update.version <= self._model_version:
            return

        self.predictor.quadratic_chunk_a = update.quadratic_chunk_a
        self.predictor.linear_chunk_b = update.linear_chunk_b
        self.predictor.constant_chunk_c = update.constant_chunk_c
        self.predictor.with_history_ready = update.with_history_ready
        self.predictor.target_latency = update.target_latency
        self.online_active = update.online_active
        self._model_version = update.version
        if not self.online_active:
            self._target_latency_done = True
        logger.debug(
            f"[PP Dynamic Chunk] [PP{self.pp_rank}] Applied online update "
            f"v{update.version}"
        )

    def begin_forward_timing(self, batch: ScheduleBatch) -> Optional[float]:
        """Start timing a real prefill forward on PP0, if calibration is active."""
        if not self._should_time_forward(batch):
            return None
        get_device_module().synchronize()
        return time.perf_counter()

    def finish_forward_timing(
        self, start: Optional[float], batch: ScheduleBatch
    ) -> None:
        """Record the prefill latency measured since ``begin_forward_timing``."""
        if start is None or not batch.forward_mode.is_extend():
            return
        get_device_module().synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1e3
        self._record_batch_execution_time(batch, elapsed_ms)

    def _should_time_forward(self, batch: ScheduleBatch) -> bool:
        return (
            self.online_active
            and self.predictor.is_ready
            and self.is_pp_chunk_leader
            and batch.forward_mode.is_extend()
        )

    def _record_batch_execution_time(
        self, batch: ScheduleBatch, elapsed_ms: float
    ) -> None:
        if not batch.extend_lens or not batch.prefix_lens:
            return

        decoding_rids = {req.rid for req in (batch.decoding_reqs or [])}
        chunks = [
            (int(chunk), int(history))
            for req, chunk, history in zip(
                batch.reqs, batch.extend_lens, batch.prefix_lens
            )
            if req.rid not in decoding_rids and chunk > 0
        ]
        if not chunks:
            return

        # Re-calibrate the target from single first-chunk prefills, whose
        # latency is directly comparable to the per-request startup samples.
        if not self._target_latency_done and len(chunks) == 1 and chunks[0][1] == 0:
            self.predictor.set_target_latency(
                self.chunked_prefill_size, elapsed_time=elapsed_ms
            )
            self._target_latency_sample_count += 1
            self._mark_model_updated()
            if self._target_latency_sample_count >= self.target_latency_samples:
                self._target_latency_done = True

        # f(C, H) = a*C*(C+H) + b*(C+H) + c for the whole batch, so a batch
        # sample is the sum over its requests.
        x1 = sum((chunk + history) * chunk for chunk, history in chunks)
        x2 = sum(chunk + history for chunk, history in chunks)
        x3 = len(chunks)
        self.chunked_fit_data.append([x1, x2, x3, elapsed_ms])

        if self.predictor.fit_chunk(
            self.chunked_fit_data,
            min_points=self.min_fit_chunk,
            max_points=self.max_fit_chunk,
        ):
            self.predictor.with_history_ready = True
            self._mark_model_updated()

        if self.predictor.history_fitted:
            self.online_active = False
            self._target_latency_done = True
            self._mark_model_updated()
            logger.info(
                f"[PP Dynamic Chunk] [PP{self.pp_rank}] Online calibration "
                f"finished after {len(self.chunked_fit_data)} samples"
            )

    def _profile_prefill_latency(self) -> Tuple[List[int], List[float]]:
        from sglang.srt.model_executor.runner_utils.buffers import (
            allocate_pp_proxy_tensors,
        )

        seq_lens: List[int] = []
        latencies: List[float] = []
        model_runner = self.model_runner
        model_config = model_runner.model_config
        input_ids_list: List[array[int]] = []
        # Each sample is a complete request. Allocating past its mapping row
        # would leave KV slots that release_kv_cache cannot address.
        max_profile_tokens = min(
            int(self.chunked_prefill_size * 1.25),
            model_config.context_len,
            self.req_to_token_pool.max_context_len,
        )
        step = max(1, max_profile_tokens // 128)
        for i in range(128):
            chunk_size = max_profile_tokens - i * step
            if chunk_size <= 0:
                break
            input_ids = array(
                "q",
                np.random.randint(0, 10000, size=chunk_size, dtype=np.int64).tobytes(),
            )
            input_ids_list.append(input_ids)

        sampling_params = SamplingParams(
            temperature=0,
            max_new_tokens=1,
        )
        # Create and profile requests
        for i, input_ids in enumerate(
            tqdm(
                input_ids_list,
                desc="Profiling prefill latency for dynamic chunking",
            )
        ):
            req = Req(
                rid=str(i),
                origin_input_text="",
                origin_input_ids=input_ids,
                sampling_params=sampling_params,
            )
            # Walk the same match -> lock -> alloc lifecycle as a scheduled
            # request so release_kv_cache can release it symmetrically.
            req.init_next_round_input(self.tree_cache)
            req.lock = self.tree_cache.lock(req.last_node)
            req.extend_end = len(req.full_untruncated_fill_ids)

            # Prepare batch
            batch = ScheduleBatch.init_new(
                [req],
                self.req_to_token_pool,
                self.token_to_kv_pool_allocator,
                self.tree_cache,
                self.model_config,
                False,
                self.spec_algorithm,
            )

            current_seq_len = req.extend_end

            if is_dp_attention_enabled():
                # Profiling runs one request on this rank; other DP ranks report 0.
                global_num_tokens = [0] * get_parallel().attn_dp_size
                dp_rank = get_parallel().attn_dp_rank
                global_num_tokens[dp_rank] = current_seq_len
                batch.global_num_tokens = global_num_tokens
                batch.global_num_tokens_for_logprob = global_num_tokens

            # Keyed and shaped like the runner's PP buffers.
            with torch.device(self.device):
                proxy_tensors = allocate_pp_proxy_tensors(
                    max_num_tokens=current_seq_len,
                    max_hidden_tokens=current_seq_len,
                    hidden_size=model_config.hidden_size,
                    dtype=model_config.dtype,
                    hc_hidden_size=getattr(model_config, "hc_hidden_size", None),
                    pp_proxy_topk_size=model_runner.get_pp_proxy_topk_size(),
                    pp_proxy_residual_num_blocks=(
                        model_runner.get_pp_proxy_residual_num_blocks()
                    ),
                )

            pp_proxy = PPProxyTensors(proxy_tensors)

            # Measure latency with device synchronization for accurate timing
            device_module = get_device_module()
            # Synchronize before starting timing to ensure clean measurement
            device_module.synchronize()

            start = time.perf_counter()
            batch.prepare_for_extend()

            try:
                # Resolve deferred H2D: prepare_for_extend now leaves input_ids=None
                if batch.input_ids is None and batch.prefill_input_ids_cpu is not None:
                    batch.input_ids = batch.prefill_input_ids_cpu.to(
                        self.device, non_blocking=True
                    )
                    batch.prefill_input_ids_cpu = None

                forward_batch = ForwardBatch.init_new(
                    batch,
                    model_runner,
                    return_hidden_states_before_norm=False,
                )
                set_is_extend_in_batch(batch.forward_mode.is_extend())

                _ = model_runner.forward(
                    forward_batch=forward_batch, pp_proxy_tensors=pp_proxy
                )

                # Synchronize after forward to ensure GPU operations complete
                device_module.synchronize()

                latency_seconds = time.perf_counter() - start
                latency_ms = latency_seconds * 1e3  # Convert to milliseconds
                seq_lens.append(len(input_ids))
                latencies.append(latency_ms)
            finally:
                # Profiling failures disable the predictor; they must still
                # return the synthetic request's KV and recurrent state.
                release_kv_cache(req, self.tree_cache, checkpoint=False)

        logger.info(
            f"[PP Dynamic Chunk] [PP0] Profiled {len(seq_lens)} samples: "
            f"seq_lens={seq_lens}, latencies_ms={latencies}"
        )
        return seq_lens, latencies


class ChunkSizePredictor:
    """Quadratic latency model f(l) = a*l^2 + b*l + c; predicts the chunk x with
    f(L + x) - f(L) = target_latency.

    The online model f(C, H) = a*C*(C+H) + b*(C+H) + c additionally captures the
    prefix length H, so it can be re-fitted from real serving batches.
    """

    def __init__(self):
        self.quadratic_coeff_a = 0.0
        self.linear_coeff_b = 0.0
        self.constant_coeff_c = 0.0
        # History-aware online model: f(C, H) = a*C*(C+H) + b*(C+H) + c.
        self.quadratic_chunk_a = 0.0
        self.linear_chunk_b = 0.0
        self.constant_chunk_c = 0.0
        self.with_history_ready = False
        self.history_fitted = False
        self.target_latency: Optional[float] = None
        self.is_ready = False

    def fit(self, seq_lens: List[int], latencies: List[float]):
        """Fit quadratic coefficients f(l) = al^2 + bl + c from data points."""
        # Skip the first data point to reduce fitting bias, as the first run is slower without warmup
        L = np.array(seq_lens[1:], dtype=np.float64)
        T = np.array(latencies[1:], dtype=np.float64)

        if len(L) < 8:
            raise ValueError(
                f"Not enough data points for quadratic fitting ({len(L)} < 8). "
                "Need at least 8 samples with different sequence lengths."
            )

        # Build design matrix for f(l) = al^2 + bl + c
        X = np.column_stack([L * L, L, np.ones_like(L)])  # [l^2, l, 1]

        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X, T, rcond=None)
            if len(coeffs) >= 3:
                fitted_a = float(coeffs[0])  # quadratic coefficient
                fitted_b = float(coeffs[1])  # linear coefficient
                fitted_c = float(coeffs[2])  # constant coefficient
            else:
                raise ValueError("Failed to fit coefficients: insufficient rank")
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Failed to fit f(l) = al^2 + bl + c: {e}")

        # Validate coefficients
        if fitted_a <= 0:
            raise ValueError(
                f"Fitted quadratic coefficient a={fitted_a:.2e} is not positive. "
                "Attention has O(n^2) complexity, so a must be positive. "
                "Check warmup data quality."
            )

        if fitted_b < 0:
            logger.warning(
                f"Fitted linear coefficient b={fitted_b:.2e} is negative. Setting b=0."
            )
            fitted_b = 0.0

        self.quadratic_coeff_a = fitted_a
        self.linear_coeff_b = fitted_b
        self.constant_coeff_c = fitted_c

        logger.info(
            f"[ChunkSizePredictor] Fitted coefficients: a={fitted_a:.2e}, "
            f"b={fitted_b:.2e}, c={fitted_c:.2e}"
        )

    def set_target_latency(self, base_chunk_size: int, elapsed_time: float = 0.0):
        """Set target latency based on base chunk size: target = f(base_chunk_size) - f(0).

        When ``elapsed_time`` is positive it comes from a real prefill batch and
        replaces the profiled estimate.
        """

        def f(length: float) -> float:
            """Total latency function: f(length) = a*length^2 + b*length + c."""
            return (
                self.quadratic_coeff_a * length * length
                + self.linear_coeff_b * length
                + self.constant_coeff_c
            )

        if elapsed_time > 0:
            self.target_latency = elapsed_time
        else:
            self.target_latency = f(float(base_chunk_size)) - f(0.0)

        if self.target_latency <= 0:
            raise ValueError(
                f"Calculated target_latency={self.target_latency:.2f}ms is not positive. "
                "Check warmup data quality."
            )

        logger.info(
            f"[ChunkSizePredictor] Target latency: {self.target_latency:.2f}ms "
            f"(base_chunk_size={base_chunk_size})"
        )

    def fit_chunk(
        self,
        chunked_data: List[List[float]],
        min_points: int,
        max_points: int,
    ) -> bool:
        """Fit the history-aware model from aggregated batch samples.

        Each row is ``[sum(C*(C+H)), sum(C+H), num_reqs, elapsed_ms]``, so the
        model is ``T = a*x1 + b*x2 + c*x3``.
        """
        num_points = len(chunked_data)
        if num_points < min_points:
            return False
        if num_points > max_points:
            self.history_fitted = True
            return False

        data = np.asarray(chunked_data, dtype=np.float64)
        execute_time = data[:, -1]
        input_x = data[:, :-1]

        try:
            params, _, _, _ = np.linalg.lstsq(input_x, execute_time, rcond=None)
        except np.linalg.LinAlgError as e:
            logger.warning(f"[ChunkSizePredictor] Failed to fit chunked model: {e}")
            return False

        fitted_a, fitted_b, fitted_c = (float(v) for v in params)
        if fitted_a <= 0:
            logger.warning(
                f"[ChunkSizePredictor] Refusing history model with "
                f"a={fitted_a:.2e} <= 0"
            )
            return False

        self.quadratic_chunk_a = fitted_a
        self.linear_chunk_b = fitted_b
        self.constant_chunk_c = fitted_c

        logger.info(
            f"[ChunkSizePredictor] Fitted history model from {num_points} samples: "
            f"a={fitted_a:.2e}, b={fitted_b:.2e}, c={fitted_c:.2e}"
        )
        return True

    def get_time_with_history(
        self,
        query_len: int,
        num_computed_tokens: int,
    ) -> float:
        """Latency of a chunk ``query_len`` with ``num_computed_tokens`` history."""
        return (
            self.quadratic_chunk_a * query_len * (query_len + num_computed_tokens)
            + self.linear_chunk_b * (query_len + num_computed_tokens)
            + self.constant_chunk_c
        )

    def predict_next_chunk_size(
        self,
        history_len: int,
        base_chunk_size: int,
        page_size: int,
        context_len: int,
        max_chunk_size: Optional[int] = None,
        use_history: bool = False,
    ) -> Optional[int]:
        """Chunk size x with f(L + x) - f(L) = target_latency for L = history_len,
        or None when the model cannot say.

        With ``use_history`` the online model f(x, L) is solved directly.
        """
        if not self.is_ready or self.target_latency is None:
            return None

        if use_history:
            if not self.with_history_ready or self.quadratic_chunk_a <= 0:
                return None
            # f(x, H) = a*x*(x+H) + b*(x+H) + c = T expands to
            # a*x^2 + (a*H + b)*x + (b*H + c - T) = 0.
            A = self.quadratic_chunk_a
            B = A * history_len + self.linear_chunk_b
            C = (
                self.linear_chunk_b * history_len
                + self.constant_chunk_c
                - self.target_latency
            )
        else:
            if self.quadratic_coeff_a <= 0:
                return None
            # f(L+x) - f(L) = T expands to a*x^2 + (2aL + b)*x - T = 0.
            A = self.quadratic_coeff_a
            B = 2 * self.quadratic_coeff_a * history_len + self.linear_coeff_b
            C = -self.target_latency

        discriminant = B * B - 4 * A * C

        if discriminant < 0:
            logger.warning(
                f"Discriminant is negative ({discriminant:.2e}). "
                f"No real solution for chunk size. L={history_len}, T={self.target_latency:.2f}ms."
            )
            return None

        sqrt_discriminant = math.sqrt(discriminant)
        calculated_chunk_size_float = (-B + sqrt_discriminant) / (2 * A)

        if calculated_chunk_size_float <= 0:
            logger.warning(
                f"Calculated chunk size is non-positive ({calculated_chunk_size_float:.2f}). "
                f"L={history_len}, T={self.target_latency:.2f}ms."
            )
            return None

        # Use a smooth coefficient to reduce the abrupt decrease in chunk size
        smooth_coeff = envs.SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR.get()
        smoothed_chunk_size = base_chunk_size + smooth_coeff * (
            calculated_chunk_size_float - base_chunk_size
        )
        # Make sure the dynamic chunk size is at least 1/4 of the base chunk size
        calculated_chunk_size = max(int(smoothed_chunk_size), base_chunk_size // 4)

        # Align to page_size (minimum alignment size is 64)
        alignment_size = max(page_size, 64)
        dynamic_chunk_size = (calculated_chunk_size // alignment_size) * alignment_size

        # Ensure aligned size is at least alignment_size
        if dynamic_chunk_size < alignment_size:
            dynamic_chunk_size = alignment_size

        # Apply constraints
        max_allowed = context_len - history_len - 100  # Leave 100 tokens margin
        if max_chunk_size is not None:
            max_allowed = min(max_allowed, max_chunk_size)
        dynamic_chunk_size = min(dynamic_chunk_size, max_allowed)

        # Align again after min operation
        dynamic_chunk_size = (dynamic_chunk_size // alignment_size) * alignment_size

        if dynamic_chunk_size < alignment_size:
            return None

        return dynamic_chunk_size

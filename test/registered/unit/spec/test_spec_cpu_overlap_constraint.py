import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import (
    _check_draft_prefetch,
    handle_speculative_decoding,
)
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _make_spec_args(device: str, algorithm: str = "EAGLE", **overrides) -> ServerArgs:
    # model_path="dummy" short-circuits ServerArgs.__post_init__; invoke the
    # speculative hook directly (same pattern as the unit/server_args tests).
    args = ServerArgs(model_path="dummy")
    args.speculative_algorithm = algorithm
    args.device = device
    # Fully specify the chain config so the hook doesn't auto-choose params.
    args.speculative_num_steps = 3
    args.speculative_eagle_topk = 1
    args.speculative_num_draft_tokens = 4
    args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["LlamaForCausalLM"],
            get_text_config=lambda: SimpleNamespace(),
        )
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestSpecCPUOverlapConstraint(CustomTestCase):
    def test_cpu_eagle_forces_disable_overlap_schedule(self):
        args = _make_spec_args(device="cpu")
        self.assertFalse(resolution_result(args, "disable_overlap_schedule"))

        handle_speculative_decoding(args)

        self.assertTrue(resolution_result(args, "disable_overlap_schedule"))

    def test_cpu_eagle3_forces_disable_overlap_schedule(self):
        args = _make_spec_args(device="cpu", algorithm="EAGLE3")

        handle_speculative_decoding(args)

        self.assertTrue(resolution_result(args, "disable_overlap_schedule"))

    def test_cpu_explicit_disable_overlap_is_preserved(self):
        args = _make_spec_args(device="cpu", disable_overlap_schedule=True)

        # Already disabled: the hook must not flip the flag, and (unlike the
        # forced-disable cases) must not warn about overriding it.
        with self.assertLogs(
            "sglang.srt.arg_groups.speculative_hook", "WARNING"
        ) as logs:
            handle_speculative_decoding(args)

        self.assertTrue(resolution_result(args, "disable_overlap_schedule"))
        self.assertFalse(
            any("Overlap schedule" in message for message in logs.output),
            f"hook warned about overriding an already-disabled overlap: {logs.output}",
        )

    def test_cuda_eagle_keeps_overlap_schedule(self):
        # Guard the constraint's scope: the hook must not touch non-CPU devices.
        args = _make_spec_args(device="cuda")

        handle_speculative_decoding(args)

        self.assertFalse(resolution_result(args, "disable_overlap_schedule"))


class TestDraftPrefetchConfig(CustomTestCase):
    def _make_args(self, **overrides):
        args = ServerArgs(model_path="dummy")
        args.enable_draft_prefetch = True
        args.speculative_algorithm = "EAGLE3"
        args.speculative_num_steps = 3
        args.speculative_eagle_topk = 1
        args.speculative_num_draft_tokens = 4
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_valid_fixed_width_chain(self):
        _check_draft_prefetch(self._make_args())

    def test_rejects_incompatible_shapes_and_modes(self):
        for overrides, message in (
            ({"speculative_eagle_topk": 2}, "topk"),
            ({"speculative_num_steps": 1}, "num-steps"),
            ({"speculative_adaptive": True}, "adaptive"),
            ({"speculative_use_rejection_sampling": True}, "rejection"),
            ({"enable_multi_layer_eagle": True}, "multi-layer"),
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                _check_draft_prefetch(self._make_args(**overrides))

    def test_dspark_static_accepts_block_prefetch_without_eagle_num_steps(self):
        args = self._make_args(
            speculative_algorithm="DSPARK",
            speculative_num_steps=None,
            speculative_num_draft_tokens=8,
        )
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            _check_draft_prefetch(args)

    def test_dspark_prefetch_rejects_dynamic_verify_and_cpu_sync_skip(self):
        args = self._make_args(
            speculative_algorithm="DSPARK",
            speculative_num_steps=None,
            speculative_num_draft_tokens=8,
        )
        with (
            envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"),
            self.assertRaisesRegex(ValueError, "static"),
        ):
            _check_draft_prefetch(args)

        args.skip_draft_prefetch_seq_lens_cpu_sync = True
        with (
            envs.SGLANG_RAGGED_VERIFY_MODE.override("static"),
            self.assertRaisesRegex(ValueError, "exact next sequence lengths"),
        ):
            _check_draft_prefetch(args)


if __name__ == "__main__":
    unittest.main()

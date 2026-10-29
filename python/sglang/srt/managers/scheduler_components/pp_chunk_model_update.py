from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PPChunkModelUpdate:
    """Versioned online dynamic-chunk model state carried by the PP req path.

    The first PP rank is the only producer. Every stage applies an update when
    it consumes the request batch that carried it, then forwards the state to
    the next stage with its own outgoing request batch. This avoids a separate
    blocking collective inside the PP scheduling loop.
    """

    version: int
    quadratic_chunk_a: float
    linear_chunk_b: float
    constant_chunk_c: float
    with_history_ready: bool
    target_latency: Optional[float]
    online_active: bool
    origin_dp_rank: int

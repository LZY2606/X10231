"""Domain rules for replayable ADCP coordinate transformation evidence."""

from .derive import derive_velocity_evidence
from .fixtures import base_casepack, replay_counterexample

__all__ = ["derive_velocity_evidence", "base_casepack", "replay_counterexample"]

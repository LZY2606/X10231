"""Replayable ADCP beam-velocity / attitude / coordinate-transform evidence tool."""

__all__ = ["schema_version"]

SCHEMA_VERSION = 2


def schema_version() -> int:
    """On-disk schema version produced by this build (see storage.migrate)."""
    return SCHEMA_VERSION

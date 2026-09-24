"""Migration from the legacy v0 flat layout to the current separated model.

The legacy database mixes beam velocity, attitude, declination, mount angle
and quality flags in one row per ping. Migration:

1. backs the file up to ``<path>.v0.bak``;
2. loads every v0 row;
3. writes a fresh v1 database at the same path, splitting each row into
   separately-typed sources with stable identities;
4. records a ``case-imported`` style migration event.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3

from ..fingerprints import stable_id
from .schema import CURRENT_SCHEMA, initialize


def needs_migration(path: str) -> bool:
    if not os.path.exists(path):
        return False
    conn = sqlite3.connect(path)
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "ping_rows" in tables:
            return True
        if "meta" not in tables:
            return False
        row = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        return row is not None and int(row[0]) < CURRENT_SCHEMA
    finally:
        conn.close()


def read_legacy_v0(path: str) -> dict:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM ping_rows ORDER BY ping_id")]
    meta = {k: v for k, v in conn.execute("SELECT key, value FROM meta")}
    conn.close()
    return {"rows": rows, "meta": meta}


def migrate_legacy_v0(path: str) -> str:
    data = read_legacy_v0(path)
    backup = f"{path}.v0.bak"
    shutil.copy2(path, backup)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    initialize(conn)
    version_id = "ver-import-v0"
    # create the import version after the event so we know the event seq
    cur = conn.execute(
        "INSERT INTO events(event_id, version_id, code, actor, label, "
        "payload, created_at) VALUES(?,?,?,?,?,?,?)",
        (stable_id("evt", version_id, "case-imported", "v0"),
         version_id, "case-imported", "migrator",
         "legacy v0 flat-table migration",
         json.dumps({"source": "legacy_v0", "pings": len(data["rows"])}),
         "2026-09-24T00:00:00+00:00"),
    )
    event_seq = cur.lastrowid
    conn.execute(
        "INSERT INTO versions(version_id, parent_id, kind, label, status, "
        "created_event, based_on_version) VALUES(?,?,?,?,?,?,?)",
        (version_id, None, "import", "legacy v0 migrated", "committed",
         event_seq, None),
    )
    conn.execute("UPDATE meta SET value=? WHERE key='current_version'",
                 (version_id,))

    decl = float(data["meta"].get("declination_deg", 0.0))
    mount = float(data["meta"].get("mount_deg", 0.0))
    conn.execute(
        "INSERT INTO config_versions(config_type, version_id, payload, "
        "introduced_event, retired_event) VALUES(?,?,?,?,NULL)",
        ("declination", "decl-v1",
         json.dumps({"version_id": "decl-v1", "angle_deg": decl}), event_seq))
    conn.execute(
        "INSERT INTO config_versions(config_type, version_id, payload, "
        "introduced_event, retired_event) VALUES(?,?,?,?,NULL)",
        ("mount", "mount-v1",
         json.dumps({"version_id": "mount-v1", "angle_deg": mount}),
         event_seq))

    def src(source_id, kind, raw, unit, ping_id=None, cell_no=None,
            beam=None, stream_key=None, numeric=None, datum=None):
        conn.execute(
            "INSERT INTO sources(source_id, kind, ping_id, cell_no, beam, "
            "stream_key, raw, unit, datum, numeric, diagnostic, created_event,"
            " superseded_by_source) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,NULL)",
            (source_id, kind, ping_id, cell_no, beam, stream_key,
             str(raw), unit, datum, numeric, None, event_seq))

    for row in data["rows"]:
        pid = row["ping_id"]
        src(stable_id("src", pid, "time"), "ping_time", row["t_s"], "s",
            ping_id=pid, numeric=row["t_s"], stream_key="ping")
        src(stable_id("src", pid, "heading"), "heading",
            row["heading_deg"], "deg", ping_id=pid,
            numeric=row["heading_deg"], datum="magnetic",
            stream_key="attitude")
        src(stable_id("src", pid, "pitch"), "pitch", row["pitch_deg"],
            "deg", ping_id=pid, numeric=row["pitch_deg"],
            stream_key="attitude")
        src(stable_id("src", pid, "roll"), "roll", row["roll_deg"], "deg",
            ping_id=pid, numeric=row["roll_deg"], stream_key="attitude")
        for beam in range(1, 5):
            src(stable_id("src", pid, f"beam{beam}"), "beam_velocity",
                row[f"beam{beam}_cmps"], "cm/s", ping_id=pid, beam=beam,
                numeric=row[f"beam{beam}_cmps"] / 100.0)
            src(stable_id("src", pid, f"corr{beam}"), "correlation",
                row[f"corr{beam}"], "counts", ping_id=pid, beam=beam,
                numeric=float(row[f"corr{beam}"]))
            src(stable_id("src", pid, f"amp{beam}"), "amplitude",
                row[f"amp{beam}"], "dB", ping_id=pid, beam=beam,
                numeric=float(row[f"amp{beam}"]))
            src(stable_id("src", pid, f"bad{beam}"), "bad_flag",
                int(row[f"badflag{beam}"]), "flag", ping_id=pid, beam=beam,
                numeric=float(row[f"badflag{beam}"]))
        src(stable_id("src", pid, "bottom"), "bottom_depth",
            row["bottom_m"], "m", ping_id=pid, numeric=row["bottom_m"])
    conn.commit()
    conn.close()
    return backup

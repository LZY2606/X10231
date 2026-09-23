from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .fingerprint import stable_hash
from .store import SOURCE_TABLES, OptimisticConflict, transaction, utc_now


def _j(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def create_dataset(conn, dataset_id: str, name: str) -> None:
    with transaction(conn):
        conn.execute("INSERT OR IGNORE INTO datasets(dataset_id,name,created_at) VALUES(?,?,?)",
                     (dataset_id, name, utc_now()))


def latest_event_seq(conn, dataset_id: str) -> int:
    row = conn.execute("SELECT COALESCE(MAX(seq),0) FROM events WHERE dataset_id=?", (dataset_id,)).fetchone()
    return int(row[0])


def _version_hash(kind: str, payload: Dict[str, Any]) -> str:
    return stable_hash({"kind": kind, "payload": payload}, 20)


def _close_previous(conn, table: str, id_col: str, record_id: str, seq: int,
                    extra_where: str = "", params: Tuple[Any, ...] = ()) -> None:
    conn.execute(
        f"UPDATE {table} SET last_seq=? WHERE {id_col}=? AND last_seq IS NULL {extra_where}",
        (seq, record_id, *params),
    )


def _record_id(prefix: str, *parts: Any) -> str:
    return prefix + "_" + stable_hash("|".join(map(str, parts)), 16)


def _insert_version(conn, kind: str, seq: int, payload: Dict[str, Any]) -> Tuple[str, str, str]:
    table, id_col = SOURCE_TABLES[kind]
    ds = payload["dataset_id"]
    if kind == "beam":
        rid = _record_id("beam", payload["ping_id"], payload["cell_id"], payload["beam"])
        _close_previous(conn, table, id_col, rid, seq)
        conn.execute(f"INSERT INTO {table}({id_col},dataset_id,ping_id,cell_id,beam,raw_repr,raw_value,raw_unit,first_seq,version_hash) VALUES(?,?,?,?,?,?,?,?,?,?)",
                     (rid, ds, payload["ping_id"], payload["cell_id"], payload["beam"], str(payload["raw_repr"]),
                      payload.get("raw_value"), payload["raw_unit"], seq, _version_hash(kind, payload)))
    elif kind == "bt":
        rid = _record_id("bt", payload["ping_id"], payload["beam"])
        _close_previous(conn, table, id_col, rid, seq)
        conn.execute(f"INSERT INTO {table}({id_col},dataset_id,ping_id,beam,raw_repr,raw_value,raw_unit,first_seq,version_hash) VALUES(?,?,?,?,?,?,?,?,?)",
                     (rid, ds, payload["ping_id"], payload["beam"], str(payload["raw_repr"]),
                      payload.get("raw_value"), payload["raw_unit"], seq, _version_hash(kind, payload)))
    elif kind in ("correlation", "amplitude"):
        rid = _record_id(kind, payload["ping_id"], payload["cell_id"], payload["beam"])
        _close_previous(conn, table, id_col, rid, seq)
        conn.execute(f"INSERT INTO {table}({id_col},dataset_id,ping_id,cell_id,beam,raw_repr,raw_value,raw_unit,first_seq,version_hash) VALUES(?,?,?,?,?,?,?,?,?,?)",
                     (rid, ds, payload["ping_id"], payload["cell_id"], payload["beam"], str(payload["raw_repr"]),
                      payload.get("raw_value"), payload["raw_unit"], seq, _version_hash(kind, payload)))
    elif kind == "flag":
        rid = _record_id("flag", payload["ping_id"], payload.get("cell_id", ""), payload["beam"])
        _close_previous(conn, table, id_col, rid, seq)
        conn.execute(f"INSERT INTO {table}({id_col},dataset_id,ping_id,cell_id,beam,flag,first_seq,version_hash) VALUES(?,?,?,?,?,?,?,?)",
                     (rid, ds, payload.get("cell_id"), payload["beam"], payload["flag"], seq, _version_hash(kind, payload)))
    elif kind == "attitude":
        rid = _record_id("att", payload["ping_time"], payload["field"])
        if payload.get("deleted"):
            _close_previous(conn, table, id_col, rid, seq)
            return rid, table, ""
        _close_previous(conn, table, id_col, rid, seq)
        conn.execute(f"INSERT INTO {table}({id_col},dataset_id,ping_time,field,raw_repr,raw_value,raw_unit,first_seq,version_hash) VALUES(?,?,?,?,?,?,?,?,?)",
                     (rid, ds, payload["ping_time"], payload["field"], str(payload["raw_repr"]),
                      payload.get("raw_value"), payload["raw_unit"], seq, _version_hash(kind, payload)))
    elif kind == "geom":
        rid = payload.get("geom_id", "geom-main")
        _close_previous(conn, table, id_col, rid, seq)
        conn.execute(f"INSERT INTO {table}({id_col},dataset_id,beam_angle,angle_unit,geometry_version,first_seq,version_hash) VALUES(?,?,?,?,?,?,?)",
                     (rid, ds, payload["beam_angle"], payload["angle_unit"], payload["geometry_version"], seq,
                      _version_hash(kind, payload)))
    elif kind == "install":
        rid = payload.get("install_id", "install-main")
        _close_previous(conn, table, id_col, rid, seq)
        conn.execute(f"INSERT INTO {table}({id_col},dataset_id,angle,angle_unit,first_seq,version_hash) VALUES(?,?,?,?,?,?)",
                     (rid, ds, payload["angle"], payload["angle_unit"], seq, _version_hash(kind, payload)))
    elif kind == "declination":
        rid = payload.get("declination_id", "decl-main")
        _close_previous(conn, table, id_col, rid, seq)
        conn.execute(f"INSERT INTO {table}({id_col},dataset_id,degrees,model_version,first_seq,version_hash) VALUES(?,?,?,?,?,?,?)",
                     (rid, ds, payload["degrees"], payload["model_version"], seq, _version_hash(kind, payload)))
    elif kind == "cell":
        rid = _record_id("cellvalid", payload["ping_id"], payload["cell_id"])
        _close_previous(conn, table, id_col, rid, seq)
        conn.execute(f"INSERT INTO {table}({id_col},dataset_id,ping_id,valid,reason,first_seq,version_hash) VALUES(?,?,?,?,?,?,?)",
                     (rid, ds, payload["ping_id"], 1 if payload["valid"] else 0, payload.get("reason", ""), seq,
                      _version_hash(kind, payload)))
    else:
        raise ValueError(f"unknown source kind {kind}")
    row = conn.execute(f"SELECT version_hash FROM {table} WHERE {id_col}=? AND first_seq=?", (rid, seq)).fetchone()
    return rid, table, row[0]


def append_event(conn, dataset_id: str, event_type: str, payload: Dict[str, Any], actor: str = "reviewer",
                 base_seq: Optional[int] = None, parent_event_seq: Optional[int] = None,
                 resolution: Optional[str] = None, expect_conflict_id: Optional[str] = None) -> Dict[str, Any]:
    with transaction(conn):
        actual = latest_event_seq(conn, dataset_id)
        if base_seq is not None and base_seq != actual:
            conflict_id = "conflict_" + stable_hash([dataset_id, actual, event_type, payload, utc_now()], 16)
            conn.execute(
                "INSERT INTO save_conflicts(conflict_id,dataset_id,attempted_seq,expected_seq,actual_seq,actor,event_type,payload_json,status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (conflict_id, dataset_id, actual + 1, base_seq, actual, actor, event_type, _j(payload),
                 "open" if not expect_conflict_id else "resolved", utc_now()),
            )
            if not expect_conflict_id:
                raise OptimisticConflict(conflict_id)
        cur = conn.execute(
            "INSERT INTO events(dataset_id,event_type,actor,payload_json,base_seq,resolution,created_at,parent_event_seq) VALUES(?,?,?,?,?,?,?,?)",
            (dataset_id, event_type, actor, _j(payload), base_seq, resolution, utc_now(), parent_event_seq),
        )
        seq = int(cur.lastrowid)
        source_kind = event_type.replace("set_", "").replace("delete_", "")
        if event_type.startswith("set_") and source_kind in SOURCE_TABLES:
            rid, table, h = _insert_version(conn, source_kind, seq, payload)
        elif event_type.startswith("delete_") and source_kind in SOURCE_TABLES:
            payload = {**payload, "deleted": True, "raw_repr": None, "raw_value": None, "raw_unit": "deleted"}
            rid, table, h = _insert_version(conn, source_kind, seq, payload)
        else:
            rid = table = h = ""
        return {"seq": seq, "record_id": rid, "table": table, "version_hash": h}


def import_baseline(conn, baseline: Dict[str, Any]) -> int:
    ds = baseline["dataset_id"]
    with transaction(conn):
        conn.execute("INSERT OR IGNORE INTO datasets(dataset_id,name,created_at) VALUES(?,?,?)",
                     (ds, baseline.get("name", ds), utc_now()))
        for ping in baseline["pings"]:
            conn.execute("INSERT OR IGNORE INTO pings(dataset_id,ping_id,ping_time,ordinal) VALUES(?,?,?,?)",
                         (ds, ping["ping_id"], ping["ping_time"], ping["ordinal"]))
        for cell in baseline["cells"]:
            conn.execute("INSERT OR IGNORE INTO cells(dataset_id,cell_id,ordinal,depth_m) VALUES(?,?,?,?)",
                         (ds, cell["cell_id"], cell["ordinal"], cell["depth_m"]))
        cur = conn.execute(
            "INSERT INTO events(dataset_id,event_type,actor,payload_json,created_at) VALUES(?,?,?,?,?)",
            (ds, "import_baseline", "system", _j({"baseline": baseline["case_version"]}), utc_now()),
        )
        seq = int(cur.lastrowid)
        # All first versions share the import event; their hashes identify the
        # independently stored facts.
        for section, kind in [("beam", "beam"), ("bottom_track", "bt"), ("correlation", "correlation"),
                              ("amplitude", "amplitude"), ("flags", "flag"), ("attitude", "attitude"),
                              ("cell_validity", "cell")]:
            for payload in baseline.get(section, []):
                payload = {"dataset_id": ds, **payload}
                _insert_version(conn, kind, seq, payload)
        for kind, payload in [("geom", baseline["geometry"]), ("install", baseline["installation"]),
                              ("declination", baseline["declination"])]:
            _insert_version(conn, kind, seq, {"dataset_id": ds, **payload})
        return seq


def _rows_at(conn, table: str, dataset: str, seq: int, where: str = "", params: Tuple[Any, ...] = ()):
    sql = f"SELECT * FROM {table} WHERE dataset_id=? AND first_seq<=? AND (last_seq IS NULL OR last_seq>?) {where} ORDER BY first_seq"
    return conn.execute(sql, (dataset, seq, seq, *params)).fetchall()


def load_state(conn, dataset: str, seq: Optional[int] = None) -> Dict[str, Any]:
    if seq is None:
        seq = latest_event_seq(conn, dataset)
    groups = {name: [] for name in SOURCE_TABLES}
    for name, (table, _) in SOURCE_TABLES.items():
        groups[name] = [dict(r) for r in _rows_at(conn, table, dataset, seq)]
    groups["pings"] = [dict(r) for r in conn.execute(
        "SELECT * FROM pings WHERE dataset_id=? ORDER BY ordinal", (dataset,))]
    groups["cells"] = [dict(r) for r in conn.execute(
        "SELECT * FROM cells WHERE dataset_id=? ORDER BY ordinal", (dataset,))]
    groups["seq"] = seq
    events = conn.execute("SELECT * FROM events WHERE dataset_id=? AND seq<=? ORDER BY seq", (dataset, seq)).fetchall()
    groups["events"] = [dict(e) for e in events]
    return groups


def list_events(conn, dataset: str) -> List[Dict[str, Any]]:
    return [dict(r) for r in conn.execute("SELECT * FROM events WHERE dataset_id=? ORDER BY seq", (dataset,))]


def current_run(conn, dataset: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT r.* FROM current_pointer p JOIN derivation_runs r ON r.run_id=p.run_id WHERE p.dataset_id=?",
        (dataset,)).fetchone()
    return dict(row) if row else None

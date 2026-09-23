"""Storage semantics: crash safety, migration, conflicts, export/import."""
import os
import sqlite3

from adcp.casepack import import_sources, replay
from adcp.pipeline import Pipeline
from adcp.storage import Store


def test_interrupted_write_leaves_old_version_visible(tmp_path):
    path = str(tmp_path / "crash.db")
    store = Store(path)
    import_sources(store, {"label": "base", "sources": [
        {"kind": "beam_velocity", "ping": 1, "beam": b, "cell": 1, "t": 1.0,
         "frame": "beam", "unit": "m/s", "raw": "0.1", "value": 0.1}
        for b in (1, 2, 3, 4)]})
    v1 = Pipeline(store).derive("v1")
    assert store.current_version() == v1
    # simulate a crash: begin a version, write rows, never commit/point
    vid = store.begin_version(v1, "interrupted")
    store.put_derived(vid, "instr_vel/p1/c1", "ok", {"x": 1}, "fp", [], "r")
    store.db.close()  # process dies without commit_version
    # restart: recovery marks the interrupted version aborted
    store2 = Store(path)
    assert store2.current_version() == v1
    versions = {v["id"]: v for v in store2.versions()}
    assert versions[vid]["status"] == "aborted"
    assert store2.derived(vid) == []  # uncommitted rows rolled back


def test_completed_write_survives_restart(tmp_path):
    path = str(tmp_path / "ok.db")
    store = Store(path)
    import_sources(store, {"label": "base", "sources": [
        {"kind": "beam_velocity", "ping": 1, "beam": b, "cell": 1, "t": 1.0,
         "frame": "beam", "unit": "m/s", "raw": "0.1", "value": 0.1}
        for b in (1, 2, 3, 4)]})
    v1 = Pipeline(store).derive("v1")
    store.close()
    store2 = Store(path)
    assert store2.current_version() == v1
    assert store2.derived(v1)  # complete new version visible


def test_old_schema_migrates(tmp_path):
    path = str(tmp_path / "old.db")
    db = sqlite3.connect(path)
    db.executescript("""
    CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
    INSERT INTO meta VALUES('schema_version','1');
    CREATE TABLE batch(id INTEGER PRIMARY KEY, label TEXT, created_at REAL);
    CREATE TABLE source(id INTEGER PRIMARY KEY, batch_id INTEGER, kind TEXT,
      ping INTEGER, beam INTEGER, cell INTEGER, t REAL, frame TEXT,
      unit TEXT, raw TEXT, value REAL, meta TEXT);
    CREATE TABLE param(id INTEGER PRIMARY KEY, name TEXT, version INTEGER,
      value TEXT, unit TEXT, meta TEXT);
    CREATE TABLE version(id INTEGER PRIMARY KEY, parent_id INTEGER,
      seq INTEGER, status TEXT, label TEXT, fingerprint TEXT,
      created_at REAL);
    CREATE TABLE derived(version_id INTEGER, node_key TEXT, seq INTEGER,
      status TEXT, payload TEXT, fingerprint TEXT, inputs TEXT,
      PRIMARY KEY(version_id, node_key));
    CREATE TABLE pointer(id INTEGER PRIMARY KEY CHECK(id=1),
      version_id INTEGER);
    CREATE TABLE event(id INTEGER PRIMARY KEY, version_id INTEGER,
      seq INTEGER, kind TEXT, payload TEXT, created_at REAL);
    INSERT INTO version VALUES(1,NULL,1,'complete','legacy','fp1',0.0);
    INSERT INTO derived VALUES(1,'instr_vel/p1/c1',1,'ok','{}','fp','[]');
    INSERT INTO pointer VALUES(1,1);
    """)
    db.commit()
    db.close()
    store = Store(path)  # triggers migrate()
    assert store.current_version() == 1
    row = store.derived(1, "instr_vel/p1/c1")[0]
    assert row["node_key"] == "instr_vel/p1/c1"
    assert "reason" in row  # new column added by migration
    store.db.execute("SELECT * FROM conflict").fetchall()  # new table
    v = store.db.execute(
        "SELECT value FROM meta WHERE key='schema_version'").fetchone()
    assert v["value"] == "2"


def test_concurrent_edit_from_same_base_conflicts(tmp_path):
    path = str(tmp_path / "conf.db")
    store = Store(path)
    replay(store)
    base = store.current_version()
    pipe = Pipeline(store)
    # user A edits from base -> mainline moves
    va, ca = pipe.decide(base, "instr_vel/p10/c3", "cand-beam3")
    assert ca is None and store.current_version() == va
    # user B edits from the same (now stale) base -> conflict
    vb, cb = pipe.decide(base, "instr_vel/p10/c3", "cand-beam4")
    assert cb is not None
    assert store.current_version() == va  # mainline untouched
    # resolve keeping B; A retained as unselected branch
    store.resolve_conflict(cb, "b")
    assert store.current_version() == vb
    assert store.derived(va, "instr_vel/p10/c3")  # A still readable
    assert store.conflicts()[0]["status"] == "resolved"


def test_export_delete_import_preserves_identity(tmp_path):
    path = str(tmp_path / "exp.db")
    store = Store(path)
    replay(store)
    dump = store.export()
    before_events = [(e["id"], e["seq"], e["kind"]) for e in store.events()]
    before_derived = [(r["version_id"], r["node_key"], r["seq"],
                       r["fingerprint"]) for r in
                      store.derived(store.current_version())]
    before_current = store.current_version()
    store.close()
    os.remove(path)
    for suffix in ("-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)
    # fresh database, import the dump
    store2 = Store(path)
    store2.import_dump(dump)
    assert store2.current_version() == before_current
    after_events = [(e["id"], e["seq"], e["kind"]) for e in store2.events()]
    assert after_events == before_events          # identity + order
    after_derived = [(r["version_id"], r["node_key"], r["seq"],
                      r["fingerprint"]) for r in
                     store2.derived(store2.current_version())]
    assert after_derived == before_derived        # lineage + identity
    # source relationships intact
    chain = store2.derived(store2.current_version(), "instr_vel/p10/c3")[0]
    assert chain["inputs"]

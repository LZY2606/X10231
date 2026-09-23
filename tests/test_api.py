"""End-to-end API/UI tests against a fresh replayed database."""
import importlib
import os
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db = str(tmp_path_factory.mktemp("api") / "api.db")
    os.environ["ADCP_DB"] = db
    for m in list(sys.modules):
        if m == "app" or m.startswith("adcp"):
            del sys.modules[m]
    import app as app_module
    importlib.reload(app_module)
    with TestClient(app_module.app) as c:
        yield c, app_module
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(db + suffix):
            os.remove(db + suffix)


def test_home_shows_sample_path(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200
    assert "样例路径" in r.text and "ping 10" in r.text


def test_ping_page_has_stable_identities(client):
    c, _ = client
    r = c.get("/ping/10?version=1")
    assert r.status_code == 200
    assert "instr_vel/p10/c3" in r.text
    assert "cand-beam3" in r.text and "cand-beam4" in r.text
    assert "待裁定" in r.text
    assert "<svg" in r.text  # vertical profile


def test_ping_page_gap_and_excluded(client):
    c, _ = client
    r = c.get("/ping/9")
    assert "姿态" in r.text and "gap" in r.text.lower() or "缺口" in r.text
    r = c.get("/ping/3")
    assert "排除" in r.text  # cell 6 beyond max depth


def test_events_are_machine_readable(client):
    c, _ = client
    events = c.get("/api/events").json()
    kinds = {e["kind"] for e in events}
    assert {"single_beam_failure", "heading_unwrap", "attitude_gap",
            "three_beam_ambiguity", "param_revision",
            "fingerprint_components", "conflict_detected",
            "conflict_resolved", "invalid_depth_cell"} <= kinds
    for e in events:
        assert isinstance(e["payload"], dict)


def test_api_ping_payload_and_source_chain(client):
    c, _ = client
    data = c.get("/api/ping/10").json()
    node = data["nodes"]["instr_vel/p10/c3"]
    assert node["stable_id"].endswith(f"@v{data['version']}")
    chain = c.get("/api/source_chain",
                  params={"node": "instr_vel/p10/c3"}).json()
    assert chain["sources"]  # raw source rows reachable from the node


def test_compare_branches(client):
    c, _ = client
    r = c.get("/compare", params={"a": 5, "b": 6})
    assert r.status_code == 200
    assert "instr_vel/p10/c3" in r.text


def test_export_reimport_roundtrip(client, tmp_path):
    c, app_module = client
    dump = c.get("/api/export").json()
    assert dump["tables"]["derived"]
    from adcp.storage import Store
    fresh = Store(str(tmp_path / "reimport.db"))
    fresh.import_dump(dump)
    assert fresh.current_version() == \
        app_module.store.current_version()
    assert [e["id"] for e in fresh.events()] == \
        [e["id"] for e in app_module.store.events()]


def test_decision_from_page_creates_version(client):
    c, app_module = client
    before = app_module.store.current_version()
    r = c.post("/decision", data={
        "version": before, "node": "instr_vel/p10/c4",
        "candidate": "cand-beam4", "note": "api test"}, follow_redirects=False)
    assert r.status_code == 303
    after = app_module.store.current_version()
    assert after != before
    row = app_module.store.derived(after, "instr_vel/p10/c4")[0]
    adopted = [x["id"] for x in row["payload"]["candidates"] if x["adopted"]]
    assert adopted == ["cand-beam4"]
    # historical version unchanged
    old = app_module.store.derived(before, "instr_vel/p10/c4")[0]
    assert all(x["adopted"] is None for x in old["payload"]["candidates"])

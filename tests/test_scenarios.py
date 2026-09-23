"""The seven boundary scenarios as repeatable cases (repo fixtures only)."""
import json
import os

CASE_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cases", "liushu_bench")


def scenario_doc():
    with open(os.path.join(CASE_DIR, "scenarios.json"),
              encoding="utf-8") as f:
        return json.load(f)


def node(store, version, key):
    rows = store.derived(version, key)
    assert rows, f"missing {key} in v{version}"
    return rows[0]


def events_of(store, version, kind):
    return [e for e in store.events()
            if e["version_id"] == version and e["kind"] == kind]


def test_single_beam_failure(replayed):
    store, v = replayed
    r = node(store, v["v1_baseline"], "instr_vel/p6/c2")
    kinds = [c["kind"] for c in r["payload"]["candidates"]]
    assert kinds == ["beam3"]            # no 4-beam value fabricated
    assert r["payload"]["candidates"][0]["adopted"] is True
    ev = events_of(store, v["v1_baseline"], "single_beam_failure")
    assert ev and ev[0]["payload"]["failed_beam"] == 3


def test_heading_crosses_360(replayed):
    store, v = replayed
    att = node(store, v["v1_baseline"], "attitude/p5")["payload"]
    assert att["attitude"] is not None
    assert att["heading_unwrapped"] is True
    # interpolated heading is near 0/360, not 180 (wrong long arc)
    h = att["attitude"]["heading"]
    assert h < 10.0 or h > 350.0
    assert events_of(store, v["v1_baseline"], "heading_unwrap")


def test_time_gap_blocks_interpolation(replayed):
    store, v = replayed
    att = node(store, v["v1_baseline"], "attitude/p9")["payload"]
    assert att["attitude"] is None
    assert "gap" in att["diagnostic"]
    earth = node(store, v["v1_baseline"], "earth_vel/p9/c1")["payload"]
    assert earth["status"] == "no_attitude"
    assert earth["candidates"][0]["earth"] is None  # stays in place
    ev = events_of(store, v["v1_baseline"], "attitude_gap")
    assert ev and "invariant" in ev[0]["payload"]


def test_three_beam_ambiguity_keeps_all_candidates(replayed):
    store, v = replayed
    r = node(store, v["v1_baseline"], "instr_vel/p10/c3")
    assert r["status"] == "ambiguous"
    cands = {c["id"]: c for c in r["payload"]["candidates"]}
    assert set(cands) == {"cand-beam3", "cand-beam4"}
    # never merged into one value
    assert cands["cand-beam3"]["value"] != cands["cand-beam4"]["value"]
    assert all(c["adopted"] is None for c in cands.values())
    ev = events_of(store, v["v1_baseline"], "three_beam_ambiguity")
    assert ev and "never merged" in ev[0]["payload"]["invariant"]


def test_mounting_angle_revision_local_propagation(replayed):
    store, v = replayed
    rows = store.derived(v["v3_mounting_revision"])
    recomputed = {r["node_key"] for r in rows
                  if r["reason"].startswith("recomputed")}
    copied = {r["node_key"] for r in rows
              if r["reason"].startswith("still-valid")}
    # mounting angle only affects ship/earth/profile nodes
    assert any(k.startswith("ship_vel/") for k in recomputed)
    assert any(k.startswith("earth_vel/") for k in recomputed)
    assert not any(k.startswith("beam_qc/") for k in recomputed)
    assert not any(k.startswith("instr_vel/") for k in recomputed)
    assert any(k.startswith("beam_qc/") for k in copied)
    ev = events_of(store, v["v3_mounting_revision"], "param_revision")
    assert ev and ev[0]["payload"]["param"] == "mounting_angle_deg"


def test_transform_order_enters_fingerprint(replayed):
    store, v = replayed
    versions = {x["id"]: x for x in store.versions()}
    main = versions[v["v3_mounting_revision"]]
    variant = versions[v["v4_order_variant"]]
    assert main["fingerprint"] != variant["fingerprint"]
    e3 = node(store, v["v3_mounting_revision"], "earth_vel/p1/c1")
    e4 = node(store, v["v4_order_variant"], "earth_vel/p1/c1")
    assert e3["payload"]["order"] != e4["payload"]["order"]
    assert e3["payload"]["candidates"][0]["earth"] != \
        e4["payload"]["candidates"][0]["earth"]
    ev = events_of(store, v["v4_order_variant"], "fingerprint_components")
    comp = ev[0]["payload"]
    assert "transform_order" in comp and "declination_version" in comp
    assert "beam_angle_version" in comp


def test_invalid_depth_cells_excluded_but_kept(replayed):
    store, v = replayed
    # cell 6 beyond max depth at every ping
    r = node(store, v["v1_baseline"], "cell_validity/p3/c6")
    assert r["status"] == "excluded"
    assert any("beyond max" in x for x in r["payload"]["reasons"])
    # low-correlation cell
    r = node(store, v["v1_baseline"], "cell_validity/p11/c4")
    assert r["status"] == "excluded"
    # excluded cells remain in the profile with diagnostics
    prof = node(store, v["v1_baseline"], "profile/p3")["payload"]
    c6 = [c for c in prof["cells"] if c["cell"] == 6][0]
    assert c6["included"] is False and c6["reasons"]
    assert events_of(store, v["v1_baseline"], "invalid_depth_cell")


def test_local_source_change_touches_only_local_nodes(replayed):
    store, v = replayed
    rows = store.derived(v["v2_local_change"])
    recomputed = sorted(r["node_key"] for r in rows
                        if r["reason"].startswith("recomputed"))
    assert recomputed == ["beam_qc/p4/c3", "cell_validity/p4/c3",
                          "earth_vel/p4/c3", "instr_vel/p4/c3",
                          "profile/p4", "ship_vel/p4/c3"]


def test_conflicting_decisions_resolvable_and_retained(replayed):
    store, v = replayed
    assert v["conflict_id"] is not None
    conflicts = store.conflicts()
    c = conflicts[0]
    assert c["status"] == "resolved"
    res = json.loads(c["resolution"])
    kept, dropped = res["kept"], res["retained_unselected"]
    # unselected branch retained with its own adopted candidate
    ra = node(store, kept, "instr_vel/p10/c3")["payload"]
    rb = node(store, dropped, "instr_vel/p10/c3")["payload"]
    adopted_a = [x["id"] for x in ra["candidates"] if x["adopted"]]
    adopted_b = [x["id"] for x in rb["candidates"] if x["adopted"]]
    assert adopted_a != adopted_b
    assert store.current_version() == kept


def test_scenario_fixture_covers_all_seven():
    doc = scenario_doc()
    ids = {s["id"] for s in doc["boundary_scenarios"]}
    assert ids == {"single_beam_failure", "heading_360_crossing",
                   "time_gap", "three_beam_ambiguity",
                   "mounting_angle_revision", "transform_order",
                   "invalid_depth_cell"}

"""Deterministic generator for the offline domain case pack
"流束换算台: ADCP 波束速度、姿态旋转与质量层级" (beam-flow bench).

Run:  .venv/bin/python cases/generate_casepack.py
Writes JSON fixtures into cases/liushu_bench/.  All randomness is seeded;
the fixtures are committed to the repo and tests only use these files.
"""
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "liushu_bench")

PINGS = list(range(1, 13))          # t = 1..12 s
CELLS = list(range(1, 7))           # 6 depth cells
BEAMS = (1, 2, 3, 4)
THETA = math.radians(20.0)


def beam_matrix():
    s, c = math.sin(THETA), math.cos(THETA)
    return [(s, 0.0, c), (-s, 0.0, c), (0.0, s, c), (0.0, -s, c)]


def main():
    rng = random.Random(20260919)
    os.makedirs(OUT, exist_ok=True)
    sources = []

    # --- attitude: 0.5 s sampling, heading crosses 360 deg near t=5,
    #     samples missing between t=8.0 and t=10.0 (time gap)
    t = 0.5
    while t <= 12.0 + 1e-9:
        if 8.0 < t < 10.0:
            t += 0.5
            continue
        heading = (350.0 + 2.0 * t) % 360.0
        pitch = 1.5 + 0.4 * math.sin(t)
        roll = -2.0 + 0.6 * math.cos(t)
        raw = {"heading": round(heading, 3), "pitch": round(pitch, 3),
               "roll": round(roll, 3)}
        sources.append({"kind": "attitude", "t": round(t, 3),
                        "frame": "ship", "unit": "deg",
                        "raw": json.dumps(raw)})
        t += 0.5

    # --- transducer config (kept as its own record)
    sources.append({
        "kind": "transducer_config", "frame": "instrument", "unit": "-",
        "raw": json.dumps({"beam_angle_deg": 20.0, "geometry": "janus-convex",
                           "geometry_version": "geom/1",
                           "frequency_khz": 600})})

    m = beam_matrix()
    for ping in PINGS:
        t = float(ping)
        sources.append({"kind": "bottom_track", "ping": ping, "t": t,
                        "frame": "earth", "unit": "m/s",
                        "raw": json.dumps({"u": 1.2, "v": 0.1})})
        for beam in BEAMS:
            flag = 0
            if ping == 6 and beam == 3:
                flag = 1            # single beam failure
            if ping == 10 and beam == 2:
                flag = 1            # contradicted flag -> ambiguity
            sources.append({"kind": "bad_beam_flag", "ping": ping,
                            "beam": beam, "t": t, "frame": "beam",
                            "unit": "flag", "raw": str(flag),
                            "value": flag})
        for cell in CELLS:
            u = 0.30 + 0.02 * cell + 0.03 * math.sin(ping)
            v = -0.10 + 0.005 * cell
            w = 0.02
            for bi, beam in enumerate(BEAMS):
                b = m[bi][0] * u + m[bi][1] * v + m[bi][2] * w
                b += rng.gauss(0.0, 0.004)
                corr = 120 + rng.gauss(0, 5)
                amp = 90 + rng.gauss(0, 8)
                if ping == 6 and beam == 3:
                    corr = 35.0     # flag consistent with low quality
                if ping == 10 and beam == 2:
                    corr = 131.0    # flag contradicted -> two legal sols
                if ping == 11 and cell == 4:
                    corr = 42.0     # low-correlation cell (all beams)
                if ping == 12 and cell == 3 and beam == 2:
                    amp = 245.0     # amplitude out of range
                sources.append({"kind": "beam_velocity", "ping": ping,
                                "beam": beam, "cell": cell, "t": t,
                                "frame": "beam", "unit": "m/s",
                                "raw": f"{b:.6f}", "value": round(b, 6)})
                sources.append({"kind": "correlation", "ping": ping,
                                "beam": beam, "cell": cell, "t": t,
                                "frame": "beam", "unit": "count",
                                "raw": f"{corr:.0f}", "value": round(corr)})
                sources.append({"kind": "amplitude", "ping": ping,
                                "beam": beam, "cell": cell, "t": t,
                                "frame": "beam", "unit": "count",
                                "raw": f"{amp:.0f}", "value": round(amp)})

    baseline = {
        "label": "流束换算台 baseline: beam velocity / correlation / "
                 "amplitude / bad-beam flags / attitude / declination / "
                 "mounting angle recorded separately",
        "sources": sources,
    }
    with open(os.path.join(OUT, "baseline.json"), "w",
              encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=1)

    # --- one local source change: corrected beam velocity at p4/b1/c3
    change_local = {
        "label": "local source correction: single beam velocity re-read",
        "sources": [{"kind": "beam_velocity", "ping": 4, "beam": 1,
                     "cell": 3, "t": 4.0, "frame": "beam", "unit": "m/s",
                     "raw": "0.221105", "value": 0.221105,
                     "meta": {"correction": "re-read from raw recorder",
                              "supersedes": "baseline"}}],
    }
    with open(os.path.join(OUT, "change_local.json"), "w",
              encoding="utf-8") as f:
        json.dump(change_local, f, ensure_ascii=False, indent=1)

    # --- one parameter revision: mounting angle re-survey
    revision = {
        "label": "mounting angle revision after dockside re-survey",
        "param": "mounting_angle_deg", "value": 1.2, "unit": "deg",
        "meta": {"reason": "dockside re-survey 2026-09-12",
                 "previous": 0.5},
    }
    with open(os.path.join(OUT, "revision_mounting_angle.json"), "w",
              encoding="utf-8") as f:
        json.dump(revision, f, ensure_ascii=False, indent=1)

    # --- two conflicting human decisions from the same base version
    conflict = {
        "label": "two reviewers decide on the ambiguous ping-10 cell-3 "
                 "three-beam case from the same base version",
        "node": "instr_vel/p10/c3",
        "decision_a": {"candidate": "cand-beam4",
                       "note": "reviewer A: flag contradicted by good "
                               "correlation; adopt 4-beam solution"},
        "decision_b": {"candidate": "cand-beam3",
                       "note": "reviewer B: trust the flag; adopt 3-beam "
                               "solution"},
        "resolution": "a",
    }
    with open(os.path.join(OUT, "conflict_decisions.json"), "w",
              encoding="utf-8") as f:
        json.dump(conflict, f, ensure_ascii=False, indent=1)

    print(f"wrote {len(sources)} sources to {OUT}")


if __name__ == "__main__":
    main()

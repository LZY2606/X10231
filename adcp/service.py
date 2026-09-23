"""应用编排：把存储层与规则层的依赖图传播串成可重放剧本。"""
from __future__ import annotations

from typing import Any, Callable

from . import pipeline
from .repo import ConflictError, Repository


class Service:
    def __init__(self, repo: Repository):
        self.repo = repo

    # ---------- 初始导入：基线 + 首次完整推导 ----------
    def initialize(self, baseline: dict[str, Any], *, actor: str = "importer",
                   message: str = "baseline import") -> dict[str, Any]:
        res = self.repo.ingest(baseline["records"], actor=actor, message=message)
        summary = self._recompute_at(res["seq"])
        self._emit_domain_events(res["seq"])
        return {"ingest": res, "derivations": summary}

    def _recompute_at(self, seq: int) -> dict:
        return self.repo._apply_recompute(self.repo.conn.cursor(), seq, pipeline.recompute)

    def recompute(self) -> dict[str, Any]:
        with self.repo.conn:
            cur = self.repo.conn.cursor()
            seq = self.repo._next_seq(cur)
            return self.repo._apply_recompute(cur, seq, pipeline.recompute)

    def _emit_domain_events(self, seq: int) -> None:
        events = []
        for d in self.repo.current_derivations():
            for ev in d["payload"].get("events", []):
                events.append({
                    "event_type": ev["type"], "scope": d["kind"],
                    "subject": ev.get("cell_id") or d.get("ping_id") or d["derivation_id"],
                    "detail": {**{k: v for k, v in ev.items() if k != "type"},
                               "derivation_id": d["derivation_id"],
                               "fingerprint": d["fingerprint"]}})
        self.repo.add_events(events, seq=seq)

    # ---------- 单源编辑 + 受影响范围报告 ----------
    def edit(self, stream_id: str, payload: dict[str, Any], *, base_version: int,
             actor: str = "reviewer", note: str | None = None, force: bool = False) -> dict[str, Any]:
        res = self.repo.edit_stream(stream_id, payload, base_version=base_version,
                                    actor=actor, note=note, force=force,
                                    recompute_fn=pipeline.recompute)
        self._emit_domain_events(res["seq"])
        return res

    def decide(self, scope: str, subject: str, action: str, **kw) -> dict[str, Any]:
        return self.repo.decision(scope, subject, action, **kw)

    # ---------- 影响范围（依赖图传播） ----------
    def affected(self, stream_id: str) -> dict[str, Any]:
        st = self.repo.current_state()
        rec = next((r for r in st["records"] if r["stream_id"] == stream_id), None)
        if rec is None:
            return {"stream_id": stream_id, "affected": [], "reason": "not found"}
        kind = rec["kind"]
        ping_id = rec["payload"].get("ping_id")
        derivs = self.repo.current_derivations()
        affected = []
        for d in derivs:
            hit = False
            if kind in ("geom", "cellgrid", "declination", "mount", "attitude"):
                hit = True  # 全局配置/姿态影响全部派生
            elif ping_id and d.get("ping_id") == ping_id:
                if d["kind"] == "cell_flow":
                    cell_versions = d["basis"].get("cell_versions", [])
                    hit = any(cv.startswith(f"{stream_id}@") for cv in cell_versions) or kind == "bottom_depth"
                else:
                    hit = d["kind"] == "bottom_track_flow" and kind == "bottom_track"
            affected.append({"derivation_id": d["derivation_id"], "would_recompute": hit})
        return {"stream_id": stream_id, "kind": kind,
                "affected": [a["derivation_id"] for a in affected if a["would_recompute"]],
                "unaffected_count": sum(1 for a in affected if not a["would_recompute"])}

    def explain_cell(self, cell_id: str) -> dict[str, Any]:
        return self.repo.explain(cell_id)

    def branch_compare(self, seq_a: int, seq_b: int, derivation_id: str) -> dict[str, Any]:
        a = self.repo.state_at(seq_a)
        b = self.repo.state_at(seq_b)
        da = next((x for x in a["derivations"] if x["derivation_id"] == derivation_id), None)
        db = next((x for x in b["derivations"] if x["derivation_id"] == derivation_id), None)
        return {"derivation_id": derivation_id, "seq_a": seq_a, "seq_b": seq_b,
                "a": None if da is None else {"version": da["version"], "fingerprint": da["fingerprint"],
                                              "payload": da["payload"]},
                "b": None if db is None else {"version": db["version"], "fingerprint": db["fingerprint"],
                                              "payload": db["payload"]},
                "fingerprint_changed": (da and db and da["fingerprint"] != db["fingerprint"]) or bool(da) != bool(db)}

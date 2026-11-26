"""Presentation records shared by archive replay and live single-device runs."""

import json
import hashlib
from collections import Counter, defaultdict
from functools import lru_cache

from retrieval_contract import verified_example
from .inputs import ROOT

STAGES = {
    "input": "指纹输入", "local": "局部向量检索", "community": "综合社区检索",
    "reasoning": "推理路径", "unseen": "未见类型与厂商", "drift": "类内概念漂移", "branches": "双路决策",
}


def mark(run, key, state, note, data=None):
    run["stages"][key] = {"state": state, "note": note, "data": data}


def record(ip, source, mode):
    return {"ip": ip, "source": source, "mode": mode, "state": "running", "final": None,
            "stages": {k: {"state": "pending", "note": "等待前置阶段。"} for k in STAGES}}


def observe(run, fp):
    preview = {k: str(v)[:320] for k, v in fp.items() if k != "ip" and v is not None and str(v).strip()}
    mark(run, "input", "done", f"已读取 {len(preview)} 个非空指纹字段。长字段仅截短展示。", preview)


def select(item, *keys):
    return {k: item[k] for k in keys if item.get(k) is not None}


def retrieval(run, key, value):
    if value is None:
        mark(run, key, "missing", "该归档未保存这个阶段的明细。")
        return
    if value.get("error"):
        mark(run, key, "error", "该阶段执行失败，不能作为识别证据。")
        return
    if key == "local":
        hits = value.get("similar_devices", [])
        data = {"candidates": [select(x, "device_type", "similarity_score") for x in hits[:5]],
                **select(value, "total_compared", "missing_perspectives")}
        mark(run, key, "done" if hits else "empty", f"返回 {len(hits)} 条近邻；相似度不是分类置信度。", data)
    elif key == "community":
        rows = value.get("matched_clusters", [])
        reasons = Counter(x.get("reason", "unknown") for x in value.get("unavailable_clusters", []))
        below = len(value.get("below_threshold", []))
        failures = sum(n for why, n in reasons.items() if "failed" in why or "unreadable" in why)
        state = "done" if rows else "error" if failures and not below else "empty"
        data = {"matches": [select(x, "device_type", "community_key", "cluster_id", "similarity_score",
                                   "score_source", "cluster_size", "report_source", "is_outlier") for x in rows[:5]],
                "unavailable_reasons": dict(reasons), **select(value, "min_similarity")}
        mark(run, key, state, f"匹配 {len(rows)} 个社区，{below} 个低于门槛，{sum(reasons.values())} 个不可用。"
             "单成员报告不代表群体共性；其余弦分来自局部检索，不是第二份独立证据。", data)
    elif key == "reasoning":
        paths = value.get("path_matching_results", [])
        supported = [x for x in paths if (x.get("evidence_coverage") or 0) > 0]
        reason = (value.get("summary") or {}).get("reason")
        explanation = {"no_reliable_community_clusters": "无可靠上游社区。",
                       "singleton_has_no_population_path": "只有单成员社区，无法估计群体熵。",
                       "no_supported_path_evidence": "缺少可比较的群体证据。"}
        mark(run, key, "done" if supported else "empty",
             f"{len(paths)} 条候选路径，其中 {len(supported)} 条有可比较证据。" + explanation.get(reason, ""),
             [select(x, "cluster_key", "path_matching_score", "evidence_coverage", "important_features")
              for x in paths[:3]])


def show_unseen(run, unseen):
    usable = bool(unseen) and unseen.get("available") is not False
    mark(run, "unseen", "done" if usable else "missing",
         "本地 Llama + LoRA 的新颖性判断。" if usable else "没有可用的未见设备检测结果。",
         select(unseen, "classified_type", "classified_vendor", "type_confidence", "vendor_confidence",
                "predicted_type", "predicted_vendor", "new_type_probability", "new_vendor_probability",
                "is_unseen") if usable else None)


def show_drift(run, first):
    drift = first.get("drift") or {}
    if drift.get("drift_score") is not None and drift.get("available") is not False:
        mark(run, "drift", "done", "超过漂移阈值。" if drift.get("is_drift") else "未超过漂移阈值。",
             select(drift, "drift_score", "threshold", "threshold_scope", "drifted_perspectives"))
    elif first.get("drift_checked") is False:
        mark(run, "drift", "skipped", "按第一阶段门控跳过类内漂移检查。")
    else:
        mark(run, "drift", "missing", "没有可用的漂移检测结果。")


def finish(run, result):
    branches = {k: select(result.get(k) or {}, "device_type", "vendor", "confidence", "device_type_reason")
                for k in ("gemini", "claude")}
    votes = [b.get("device_type") for b in branches.values()]
    meaningful = [v for v in votes if v and str(v).upper() not in {"UNKNOWN", "NONE"}]
    branch_state = "missing" if not all(votes) else "done" if meaningful else "empty"
    mark(run, "branches", branch_state, "按项目规则分别合并类型与厂商；UNKNOWN 表示分支弃权。", branches)
    label = result.get("predicted_device_type")
    run["final"] = {"device_type": label, "vendor": result.get("predicted_vendor"),
                    "confidence": result.get("final_confidence"), "vendor_confidence": result.get("final_vendor_confidence"),
                    "method": result.get("decision_source") or "joint_vote",
                    "reason": result.get("device_type_reason"), "winning_llm": result.get("winning_llm")}
    run["state"] = "complete" if label and (meaningful or result.get("decision_source") == "adapter_fallback") else "partial"


def fail(run, message):
    for k, v in run["stages"].items():
        if v["state"] == "running":
            mark(run, k, "error", message)
        elif v["state"] == "pending":
            mark(run, k, "skipped", "任务提前结束。")
    run.update(state="failed", error=message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def linked_evidence(base, name, inputs):
    """Supplement legacy rows only from shipped, co-versioned artifacts for the exact fingerprint."""
    wanted = {digest(fp): fp for fp in inputs.values()}
    levels = defaultdict(dict)
    for level in ("local", "community", "reasoning"):
        path = base / "evidence" / f"{name}_{level}.json"
        if not path.is_file():
            continue
        for entry in json.loads(path.read_text()):
            meta = entry.get("cache_metadata") or {}
            key = meta.get("fingerprint_sha256")
            if (key in wanted and meta.get("schema_version") and meta.get("index_signature") and
                    meta.get("top_k") == 5 and digest(entry.get("query_fingerprint")) == key):
                identity = (key, meta["schema_version"], meta["index_signature"])
                levels[identity][level] = entry
    matches = defaultdict(list)
    for (key, version, _), evidence in levels.items():
        if len(evidence) == 3:
            matches[wanted[key]["ip"]].append((version, evidence))
    return {ip: runs[0] for ip, runs in matches.items() if len(runs) == 1}


@lru_cache(maxsize=1)
def archives():
    cases = {}
    types = json.loads((ROOT / "config/rag_devices.json").read_text())["IoT"]
    for name in sorted(types, key=lambda x: (x != "CAMERA", x)):
        base = ROOT / "demo"
        path = base / "fullflow" / name / "results.json"
        if not path.is_file():
            continue
        inputs = {}
        for fn in ("cases.json", "cases_extended.json"):
            p = base / name / fn
            if p.is_file():
                inputs.update({r["ip"]: r["fingerprint"] for r in json.loads(p.read_text())})
        rows = json.loads(path.read_text())
        linked = linked_evidence(base, name, inputs) if any(not r.get("retrieval_evidence") for r in rows) else {}
        for i, row in enumerate(rows, 1):
            key = f"{name}-{i:02}"
            if not row.get("retrieval_evidence") and row["ip"] in linked:
                version, evidence = linked[row["ip"]]
                if row.get("pipeline_version") in (None, version):
                    row = {**row, "retrieval_evidence": evidence, "linked_version": version}
            cases[key] = (row, inputs.get(row["ip"]))
    return cases


@lru_cache(maxsize=1)
def archive_groups():
    cases = archives()
    return {"verified": [k for k, (r, _) in cases.items() if verified_example(r)],
            "errors": [k for k, (r, _) in cases.items() if r.get("predicted_type") and r["predicted_type"] != r["true_type"]],
            "all": list(cases)}


@lru_cache(maxsize=1)
def error_notes():
    return json.loads((ROOT / "demo/workbench/archive_notes.json").read_text(encoding="utf-8"))


def replay(key):
    row, fp = archives()[key]
    run = record(key, "archive", "archive")
    evidence = row.get("retrieval_evidence") or {}
    hits = (evidence.get("local") or {}).get("similar_devices") or []
    run["comparison"] = {"reference": row["true_type"], "local_top1": hits[0]["device_type"] if hits else None,
                         "verified": verified_example(row), "linked_version": row.get("linked_version"),
                         "version": row.get("pipeline_version")}
    run["review"] = error_notes().get(digest(row)[:16])
    if fp:
        # For replay, publish field names, not a list of third-party addresses.
        mark(run, "input", "done", "归档观测的非空字段（不重新查询该设备）。",
             [k for k, v in fp.items() if k != "ip" and v is not None and str(v).strip()])
    else:
        mark(run, "input", "missing", "该归档未附输入指纹。")
    for stage in ("local", "community", "reasoning"):
        retrieval(run, stage, evidence.get(stage))
    first = row.get("first_stage") or {}
    show_unseen(run, first.get("unseen") or {})
    show_drift(run, first)
    finish(run, {"predicted_device_type": row.get("predicted_type"), "predicted_vendor": row.get("predicted_vendor"),
                 "final_confidence": row.get("decision_confidence"), "final_vendor_confidence": row.get("vendor_confidence"),
                 "winning_llm": row.get("winning_llm"),
                 **{k: {"device_type": row.get(f"{k}_type")} for k in ("gemini", "claude")}})
    return run

#!/usr/bin/env python
"""
Full-pipeline agent showcase for all 11 RAG device types.

For each type: take the fixed-seed random validation candidates and run the
complete IdentificationAgent flow (decompose → local + community + reasoning
retrieval → unseen + drift first stage → Gemini+Claude decision vote). Results
retain every case, including incomplete and incorrect outcomes.

New runs land in demo/runs/{schema}/{bits}bit/{TYPE}/:
  results.json   per-IP: prediction, confidences, first-stage outputs, timing
  showcase.json  the selected best case(s) for the UI

Run (sequential over types, one GPU):
  CUDA_VISIBLE_DEVICES=0 python demo/run_fullflow_demo.py [--types T1 T2 ...]
"""

import argparse
import json
import os
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "agent"))
sys.path.insert(0, REPO)

from retrieval_contract import RETRIEVAL_SCHEMA_VERSION as PIPELINE_VERSION, verified_example

TYPES = json.load(open(os.path.join(REPO, "config", "rag_devices.json")))["IoT"]
DEMO = os.path.join(REPO, "demo")
ADAPTER = os.path.join(REPO, "evaluation/unseen/llama3/results_v2/final_model")


def atomic_json_dump(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_json_records(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, list) else []


def load_retrieval_map(device_type, level):
    path = os.path.join(os.environ["IOTPROBER_QUERY_DB_DIR"], level, f"{device_type}_{level}.json")
    return {
        str(row.get("query_fingerprint", {}).get("ip")): row
        for row in load_json_records(path)
        if row.get("cache_metadata", {}).get("schema_version") == PIPELINE_VERSION
    }


def retrieval_evidence(ip, local_map, community_map, reasoning_map):
    local = local_map.get(ip, {})
    community = community_map.get(ip, {})
    reasoning = reasoning_map.get(ip, {})
    return {
        "local": {
            "similar_devices": local.get("similar_devices", []),
            "confidence_score": local.get("confidence_score"),
            "missing_perspectives": local.get("missing_perspectives", []),
            "total_compared": local.get("total_compared"),
        },
        "community": {
            "matched_clusters": community.get("matched_clusters", []),
            "unavailable_clusters": community.get("unavailable_clusters", []),
            "below_threshold": community.get("below_threshold", []),
            "min_similarity": community.get("min_similarity"),
        },
        "reasoning": {
            "path_matching_results": reasoning.get("path_matching_results", []),
            "summary": reasoning.get("summary", {}),
        },
    }


def prepare_test_csv(t, cases):
    import csv
    validation = os.environ["IOTPROBER_VALIDATION_DIR"]
    os.makedirs(validation, exist_ok=True)
    path = os.path.join(validation, f"test_{t}_1.csv")
    backup = path + ".fullflow_bak"
    if os.path.exists(path) and not os.path.exists(backup):
        shutil.copy2(path, backup)          # keep whatever was there before
    cols = list(cases[0]["fingerprint"].keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in cases:
            w.writerow([c["fingerprint"][k] if c["fingerprint"][k] is not None else ""
                        for k in cols])


def clear_caches(t):
    for sub in ("local", "community", "reasoning"):
        p = os.path.join(os.environ["IOTPROBER_QUERY_DB_DIR"], sub, f"{t}_{sub}.json")
        if os.path.exists(p):
            os.remove(p)
    pred = os.environ["IOTPROBER_PREDICTION_DIR"]
    for p in (os.path.join(pred, f"{t}_type_prediction.json"),
              os.path.join(pred, f"{t}_vendor_prediction.json")):
        if os.path.exists(p):
            os.remove(p)


def restore_test_csv(t):
    path = os.path.join(os.environ["IOTPROBER_VALIDATION_DIR"], f"test_{t}_1.csv")
    backup = path + ".fullflow_bak"
    if os.path.exists(backup):
        shutil.move(backup, path)


def run_type(t, output_dir, showcase_only=False, fresh=False, retrieval_llm="DEEPSEEK",
             quantization_bits=4, adapter=ADAPTER, update_graph=True):
    from agent import IdentificationAgent
    from path_config import DRIFT_OUTPUT_DIR
    case_name = "cases.json" if showcase_only else "validation_cases.json"
    case_path = os.path.join(DEMO, t, case_name)
    if not os.path.exists(case_path):
        raise FileNotFoundError(
            f"{case_path} is missing; run select_demo_data.py --merge first"
        )
    cases = load_json_records(case_path)
    if not cases:
        raise ValueError(f"No cases in {case_path}")
    prepare_test_csv(t, cases)
    if fresh:
        clear_caches(t)
    t0 = time.time()
    try:
        agent = IdentificationAgent(llm=retrieval_llm, gpu=0)
        agent.run_retrieval(whether_decompose=True, whether_local=True,
                            whether_community=True, whether_reasoning=True,
                            devices=[t], top_k=5, quick_resume=True)
        agent.run_decision(devices=[t], whether_local=True, whether_community=True,
                           whether_reasoning=True, top_k=5, enable_first_stage=True,
                           unseen_adapter_path=adapter, unseen_load_in_4bit=quantization_bits == 4,
                           unseen_load_in_8bit=quantization_bits == 8,
                           drift_model_dir=os.environ.get("IOTPROBER_DRIFT_MODEL_DIR", DRIFT_OUTPUT_DIR),
                           gpu=0, quick_resume=True, update_graph=update_graph)
    finally:
        restore_test_csv(t)
    elapsed = time.time() - t0

    unseen_config = {"bits": quantization_bits, "adapter": os.path.realpath(adapter)}
    pred = os.environ["IOTPROBER_PREDICTION_DIR"]
    type_pred = {
        str(r["ip"]): r for r in load_json_records(
            os.path.join(pred, f"{t}_type_prediction.json")
        ) if r.get("pipeline_version") == PIPELINE_VERSION and r.get("unseen_config") == unseen_config
    }
    ven_pred = {
        str(r["ip"]): r for r in load_json_records(
            os.path.join(pred, f"{t}_vendor_prediction.json")
        ) if r.get("pipeline_version") == PIPELINE_VERSION and r.get("unseen_config") == unseen_config
    }
    local_map = load_retrieval_map(t, "local")
    community_map = load_retrieval_map(t, "community")
    reasoning_map = load_retrieval_map(t, "reasoning")

    rows = []
    for c in cases:
        ip = str(c["ip"])
        tp, vp = type_pred.get(ip, {}), ven_pred.get(ip, {})
        retrieval_complete = all(
            ip in level_map for level_map in (local_map, community_map, reasoning_map)
        )
        decision_complete = bool(tp.get("predicted_device_type"))
        unseen_available = ((tp.get("first_stage") or {}).get("unseen") or {}).get("available", False)
        rows.append({
            "ip": ip,
            "true_type": t,
            "adapter_only": c["result"],
            "predicted_type": tp.get("predicted_device_type"),
            "decision_confidence": tp.get("confidence"),
            "winning_llm": tp.get("winning_llm"),
            "gemini_type": tp.get("gemini_device_type"),
            "claude_type": tp.get("claude_device_type"),
            "predicted_vendor": vp.get("predicted_vendor"),
            "vendor_confidence": vp.get("vendor_confidence", vp.get("confidence")),
            "first_stage": tp.get("first_stage"),
            "unseen_config": unseen_config,
            "pipeline_version": tp.get("pipeline_version"),
            "retrieval_evidence": retrieval_evidence(
                ip, local_map, community_map, reasoning_map
            ),
            "status": "completed" if decision_complete and retrieval_complete and unseen_available else "incomplete",
            "retrieval_complete": retrieval_complete,
            "decision_complete": decision_complete,
            "type_correct": tp.get("predicted_device_type") == t,
        })
    completed = [r for r in rows if r["status"] == "completed"]
    correct = [r for r in completed if r["type_correct"]]
    incorrect = [r for r in completed if not r["type_correct"]]
    showcase = [r for r in completed if verified_example(r)][:3]

    outdir = os.path.join(output_dir, t)
    os.makedirs(outdir, exist_ok=True)
    atomic_json_dump(os.path.join(outdir, "results.json"), rows)
    atomic_json_dump(os.path.join(outdir, "showcase.json"), {
        "type": t,
        "elapsed_sec": round(elapsed, 1),
        "source": case_name,
        "n_cases": len(rows),
        "n_completed": len(completed),
        "n_correct": len(correct),
        "n_incorrect": len(incorrect),
        "n_incomplete": len(rows) - len(completed),
        "selection": "verified_local_top1_and_final_type",
        "showcase": showcase,
    })
    ok = [r["ip"] for r in correct]
    print(f"[{t}] {len(correct)}/{len(completed)} completed correct ({','.join(ok) or '-'}) "
          f"best_conf={showcase[0]['decision_confidence'] if showcase else None} "
          f"in {elapsed:.0f}s", flush=True)
    return {
        "cases": len(rows),
        "completed": len(completed),
        "correct": len(correct),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quantization-bits", type=int, choices=(4, 8), default=4,
                   help="Quantization of the local unseen Llama base (default: 4)")
    p.add_argument("--unseen-adapter", default=ADAPTER, help="Released LoRA adapter directory")
    p.add_argument("--no-graph-update", action="store_true",
                   help="Evaluate without writing predictions back to the knowledge graph")
    p.add_argument("--output-dir", help="Output root (default: demo/runs/<schema>/<bits>bit)")
    p.add_argument("--types", nargs="*", default=None)
    p.add_argument("--showcase-only", action="store_true",
                   help="Run curated UI cases instead of the random validation set")
    p.add_argument("--fresh", action="store_true",
                   help="Delete this type's generated retrieval/prediction cache before running")
    p.add_argument(
        "--retrieval-llm",
        choices=["CLAUDE", "DEEPSEEK", "GEMINI", "OPENAI"],
        default=os.environ.get("IOTPROBER_FULLFLOW_RETRIEVAL_LLM", "DEEPSEEK"),
        help="LLM used for cluster matching (default: DEEPSEEK via llm_config.local.json)",
    )
    a = p.parse_args()
    output_dir = os.path.abspath(a.output_dir or os.path.join(
        DEMO, "runs", PIPELINE_VERSION, f"{a.quantization_bits}bit"))
    archive = os.path.realpath(os.path.join(DEMO, "fullflow"))
    if os.path.commonpath((os.path.realpath(output_dir), archive)) == archive:
        p.error("demo/fullflow is the historical replay archive; choose a separate --output-dir")
    # Configure these before the first agent import: path_config resolves them on import.
    for name, directory in (("VALIDATION", "validation"), ("QUERY_DB", "query_db"), ("PREDICTION", "predict")):
        os.environ.setdefault(f"IOTPROBER_{name}_DIR", os.path.join(output_dir, "_runtime", directory))
    todo = a.types or TYPES
    run_status = {}
    run_counts = {}
    for t in todo:
        try:
            run_counts[t] = run_type(
                t,
                output_dir,
                showcase_only=a.showcase_only,
                fresh=a.fresh,
                retrieval_llm=a.retrieval_llm,
                quantization_bits=a.quantization_bits,
                adapter=a.unseen_adapter,
                update_graph=not a.no_graph_update,
            )
            run_status[t] = "completed"
        except Exception as exc:  # noqa: BLE001 — keep the sweep going
            print(f"[{t}] FAILED: {type(exc).__name__}: {str(exc)[:180]}", flush=True)
            restore_test_csv(t)
            run_status[t] = f"failed: {type(exc).__name__}: {str(exc)[:180]}"
        atomic_json_dump(os.path.join(output_dir, "manifest.json"), {
            "schema_version": PIPELINE_VERSION,
            "status": "complete" if set(todo) == set(TYPES) and all(
                value == "completed" for value in run_status.values()
            ) else "partial",
            "source": "cases.json" if a.showcase_only else "validation_cases.json",
            "retrieval_llm": a.retrieval_llm,
            "unseen_quantization_bits": a.quantization_bits,
            "unseen_adapter": a.unseen_adapter,
            "graph_update_enabled": not a.no_graph_update,
            "requested_types": todo,
            "type_status": run_status,
            "type_counts": run_counts,
        })
    print("FULLFLOW SWEEP DONE", flush=True)

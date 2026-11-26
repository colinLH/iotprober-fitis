"""One CPU-only subprocess per live run; all per-run writes stay in its private directory."""

import json
import logging
import os
import sys
from pathlib import Path

from .inputs import ROOT
from .results import fail, finish, mark, retrieval, show_drift, show_unseen

CPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false",
}


def worker_env(folder):
    return {**os.environ, **CPU_ENV, "IOTPROBER_QUERY_DB_DIR": str(folder / "evidence"),
            "IOTPROBER_SINGLETON_REPORT_DIR": os.getenv("IOTPROBER_SINGLETON_REPORT_DIR")
            or str(Path.home() / ".cache/iotprober-workbench/knowledge-reports"),
            "IOTPROBER_UNSEEN_SUMMARY_CACHE": str(folder / "summaries.jsonl"),
            "IOTPROBER_PREDICTION_DIR": str(folder / "predictions"), "IOTPROBER_VALIDATION_DIR": str(folder / "validation")}


def memory_available():
    """Consider both host memory and (where present) the container's memory ceiling."""
    available = int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines()
                         if x.startswith("MemAvailable:"))) * 1024
    cg = Path("/sys/fs/cgroup")
    limit = cg / "memory.max"
    used = cg / "memory.current"
    if not limit.is_file():
        limit = cg / "memory/memory.limit_in_bytes"
        used = cg / "memory/memory.usage_in_bytes"
    if limit.is_file() and used.is_file() and limit.read_text().strip().isdigit():
        available = min(available, int(limit.read_text()) - int(used.read_text()))
    return available


def process(run, fp, publish):
    from path_config import DEFAULT_LLM_MODEL_DIR, LOCAL_VECTOR_DB_DIR
    from retrieval import MultiLevelRetrieval

    def begin(key, note):
        mark(run, key, "running", note)
        publish(run)

    full = run["mode"] == "full"
    begin("local", "正在 CPU 上加载 Qwen 并读取知识库向量。")
    if not any((Path(LOCAL_VECTOR_DB_DIR) / "local_npz").glob("*_embeddings.npy")):
        fail(run, "未找到本地 NumPy 索引，请检查 IOTPROBER_VECTOR_DB_DIR。")
        return
    if memory_available() < (48 if full else 5) * 1024**3:
        fail(run, "当前可用 CPU 内存不足，已停止而不加载模型。可切换为归档回放。")
        return
    if full and not Path(DEFAULT_LLM_MODEL_DIR, "config.json").is_file():
        fail(run, "未找到 Llama 基座，请检查 IOTPROBER_LLM_MODEL_DIR。")
        return
    engine = MultiLevelRetrieval(llm="DEEPSEEK", gpu=-1, offline_local=not full)
    local = engine.local_retrieval(fp, top_k=5, graph_neighbor=False)
    retrieval(run, "local", local)
    if local.get("error"):
        fail(run, "局部检索失败，请检查查询特征和索引版本。")
        return
    if local.get("reason") == "no_observed_perspectives":
        fail(run, "指纹只含缺失值或空模板：未生成候选，也未调用决策 API。")
        return
    if not full:
        for key in ("community", "reasoning", "unseen", "drift", "branches"):
            mark(run, key, "skipped", "快速本地模式只运行向量检索；未执行此阶段。")
        hits = local.get("similar_devices", [])
        candidate = hits[0] if hits and (hits[0].get("confidence_adjusted_score") or 0) > 0 else {}
        run["final"] = {"device_type": candidate.get("device_type"), "vendor": None, "method": "local_candidate",
                        "reason": "仅作近邻候选参考，未判定厂商或执行联合投票。"}
        run["state"] = "complete"
        return

    begin("community", "正在比对候选社区与指纹。")
    community = engine.community_retrieval(fp, local.get("community_candidates", []),
                                           cache_metadata=local.get("cache_metadata"))
    retrieval(run, "community", community)
    begin("reasoning", "正在比较关键视角和路径证据。")
    reasoning = engine.reasoning_path_retrieval(local, community)
    retrieval(run, "reasoning", reasoning)
    # The production decision tool reads these files; the subprocess owns QUERY_DB_DIR.
    label = "WORKBENCH"
    engine.save_history_by_type(label, local, community, reasoning)
    engine.clear_history()
    del engine
    from agent import IoTDecisionGraph

    def node_done(node, values):
        if node == "unseen":
            show_unseen(run, values.get("unseen_result") or {})
        elif node == "gate" and values["first_stage"].get("drift_checked"):
            mark(run, "drift", "running", "正在 CPU 上运行 PACA 漂移检测。")
        elif node in {"gate", "drift"}:
            show_drift(run, values["first_stage"])
        elif node == "prepare":
            mark(run, "branches", "running", "双路 LLM 正在读取证据并分别判定。")
        else:
            return
        publish(run)

    begin("unseen", "正在 CPU 上运行第一阶段及决策图，耗时可能较长。")
    graph = IoTDecisionGraph(retrieval_agent=None, gpu=-1, allow_web_search=False)
    result = graph.classify(fp["ip"], label, fp, retrieval_results=(local, community, reasoning), on_update=node_done)
    if result.get("error"):
        fail(run, "决策未完成，请检查模型、API 配置或服务额度。已完成的检索结果已保留。")
        return
    finish(run, result)


def main(folder):
    folder = Path(folder).resolve()
    os.environ.update(worker_env(folder))
    os.nice(10)
    os.chdir(folder)
    sys.path[:0] = [str(ROOT / "agent"), str(ROOT)]
    # Do not let legacy modules configure project-wide fingerprint logs.
    logging.basicConfig(handlers=[logging.NullHandler()], force=True)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    task = json.loads((folder / "input.json").read_text())
    run = task["run"]

    def publish(state):
        temp = folder / "progress.tmp"
        temp.write_text(json.dumps(state, ensure_ascii=False, allow_nan=False))
        temp.replace(folder / "progress.json")

    try:
        process(run, task["fingerprint"], publish)
    except Exception as exc:
        fail(run, f"运行失败（{type(exc).__name__}）。请检查模型、索引及相应服务配置。")
    publish(run)


if __name__ == "__main__":
    main(sys.argv[1])

"""Small paired CPU diagnostic, not a final-decision accuracy benchmark."""

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "agent")]
os.environ.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                  MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1",
                  TOKENIZERS_PARALLELISM="false")
logging.basicConfig(handlers=[logging.NullHandler()])

import numpy as np
import torch

from path_config import LOCAL_VECTOR_DB_DIR, QUERY_DB_DIR
from retrieval import MultiLevelRetrieval, RETRIEVAL_SCHEMA_VERSION, _ambiguous_kb_ips, local_index_signature


def inputs(per_type, signature, view_texts, max_view_chars):
    samples = []
    archive_dir = ROOT / "demo/fullflow"
    archived_ips = set()
    for p in sorted(archive_dir.glob("*/results.json")):
        rows = json.loads(p.read_text())
        archived_ips.update(row["ip"] for row in rows)
        if p.parent.name != "CAMERA":
            continue
        fps = {c["ip"]: c["fingerprint"] for c in json.loads((ROOT / "demo/CAMERA/cases.json").read_text())}
        for i, row in enumerate(rows, 1):
            hits = row["retrieval_evidence"]["local"]["similar_devices"]
            samples.append((f"CAMERA-{i:02}", "CAMERA", fps[row["ip"]], hits, "diagnostic"))
    for p in sorted(Path(QUERY_DB_DIR, "local").glob("*_local.json")):
        typ = p.name.removesuffix("_local.json")
        by_ip = {}
        for row in json.loads(p.read_text()):
            fp = row["query_fingerprint"]
            meta = row.get("cache_metadata") or {}
            if (meta.get("schema_version") == "2026-09-25-v3" and meta.get("index_signature") == signature
                    and meta.get("top_k") == 5 and not row.get("graph_neighbor") and fp["ip"] not in archived_ips):
                if not max_view_chars or max(map(len, view_texts(fp))) <= max_view_chars:
                    by_ip[fp["ip"]] = row
        # Choose without inspecting either method's hits, including failed/empty rankings.
        ordered = sorted(by_ip, key=lambda ip: hashlib.sha256(f"upgrade-v1:{typ}:{ip}".encode()).digest())
        for i, ip in enumerate(ordered[:per_type], 1):
            row = by_ip[ip]
            samples.append((f"{typ}-check-{i}", typ, row["query_fingerprint"], row.get("similar_devices", []), "paired"))
    return samples


def compare(per_type, max_view_chars):
    torch.set_num_threads(1)
    signature = local_index_signature()
    engine = MultiLevelRetrieval(gpu=-1, offline_local=True)
    cases = inputs(per_type, signature, engine._view_texts, max_view_chars)
    encode = lru_cache(maxsize=512)(engine.embedding_model.embed_query)
    weights = np.array(engine.perspective_weights, dtype=np.float32)
    queries, top = [], [[] for _ in cases]
    for i, (_, _, fp, _, _) in enumerate(cases):
        parts = engine._view_texts(fp)
        blocks = np.asarray([encode(part) if part else np.zeros(engine.embedding_dim) for part in parts],
                            dtype=np.float32)
        mask_w = weights * np.asarray([bool(p) for p in parts])
        norms = np.linalg.norm(blocks, axis=1)
        blocks /= np.where(norms > 0, norms, 1)[:, None]
        queries.append((blocks * (mask_w / (mask_w.sum() or 1))[:, None]).ravel())
        print(f"encoded {i + 1}/{len(cases)}", flush=True)
    query_vecs = np.stack(queries).T.copy()
    del encode, engine
    gc.collect()
    # Scan each mmap only once for the entire batch (important on shared storage).
    index = Path(LOCAL_VECTOR_DB_DIR, "local_npz")
    for p in sorted(index.glob("*_embeddings.npy")):
        typ = p.name.removesuffix("_embeddings.npy")
        ips = np.load(index / f"{typ}_ips.npy", allow_pickle=False)
        matrix = np.load(p, mmap_mode="r")
        scores = np.clip(matrix @ query_vecs, -1, 1)
        scores[np.isin(ips, list(_ambiguous_kb_ips()))] = -np.inf
        for j, (_, _, fp, _, _) in enumerate(cases):
            if not query_vecs[:, j].any():
                continue
            col = scores[:, j].copy()
            col[ips == fp["ip"]] = -np.inf
            idx = np.argpartition(col, -min(5, len(col)))[-5:]
            top[j] = sorted(top[j] + [(float(col[k]), typ) for k in idx if np.isfinite(col[k])],
                            reverse=True)[:5]
        del matrix
    return {"schema": RETRIEVAL_SCHEMA_VERSION, "index_signature": signature, "per_type": per_type,
            "max_paired_view_chars": max_view_chars,
            "cases": [{"id": key, "split": split, "label": label,
                       "old_types": [hit["device_type"] for hit in old[:5]],
                       "new_types": [typ for _, typ in top[i]], "new_scores": [round(s, 5) for s, _ in top[i]]}
                      for i, (key, label, _, old, split) in enumerate(cases)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-type", type=int, default=2)
    parser.add_argument("--max-view-chars", type=int, default=4096,
                        help="Restrict paired diagnostic to short fingerprints (0 = unlimited); never truncate text")
    parser.add_argument("--output", required=True, type=Path, help="New JSON file (never overwritten)")
    args = parser.parse_args()
    if args.per_type < 1 or args.max_view_chars < 0 or args.output.exists():
        parser.error("choose a new output path and a positive sample size")
    result = compare(args.per_type, args.max_view_chars)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2)

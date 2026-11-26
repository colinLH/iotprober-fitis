#!/usr/bin/env python
"""生成"真实新颖厂商"验证样本 (vendor novelty 真交叉修复的数据面)。

背景: release gate 的 vendor novelty 折此前用 pseudo_unseen_vendor 样本做
正例 —— 其目标厂商只是从 prompt 候选列表里删掉, 仍在完整注册表中, 打分
端按注册表匹配必然命中 → score 强制 0 → recall 按构造为 0 (v2 report
vendor_novelty_f1=0.010 的根因)。

本脚本从 OOD 源数据 (dataset_MEDIA_SERVER.csv / dataset_VPN.csv) 里筛出
目标厂商**真实不在完整注册表**的设备 (Unraid/Emby/Inseego/AnyDesk 等,
约 160 行), 用与 prepare_data 完全相同的指纹管线 (info-cols + summarizer
+ 同款 prompt 契约) 生成补充验证样本, split_origin=real_novel_vendor_validation。

产物: dataset/novel_vendor_validation.jsonl
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPT_DIR)

import pandas as pd  # noqa: E402
from util import (  # noqa: E402
    CLASSIFIER_SYSTEM,
    DeepSeekFingerprintSummarizer,
    UNSEEN_INFO_COLS,
    build_fingerprint_info_text,
    build_unseen_type_vendor_classification_prompt,
    normalize_unseen_vendor,
)

MODEL_ID = "/home/nfs/models/Llama-3.1-8B-Instruct"
LLM_CFG_PATH = os.path.join(ROOT, "config", "llm_config.local.json")
CACHE_PATH = os.path.join(SCRIPT_DIR, "dataset", "summary_cache.jsonl")
MAX_LEN = 32768
ANSWER_BUDGET = 64
SOURCES = [
    os.path.join(SCRIPT_DIR, "..", "dataset", "dataset_MEDIA_SERVER.csv"),
    os.path.join(SCRIPT_DIR, "..", "dataset", "dataset_VPN.csv"),
]


def main():
    from transformers import AutoTokenizer

    kv = json.load(open(os.path.join(
        SCRIPT_DIR, "results_v2", "final_model", "known_vendors.json")))
    known_types = kv["rag_device_types"]
    known_vendors_by_type = kv["known_vendors_by_type"]
    full_registry = {
        normalize_unseen_vendor(v)
        for vs in known_vendors_by_type.values()
        for v in (vs or [])
    }

    frames = []
    for src in SOURCES:
        df = pd.read_csv(src, low_memory=False)
        df["device_type"] = os.path.basename(src)[len("dataset_"):-len(".csv")]
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["vendor"] = data["vendor"].fillna("").astype(str).str.strip()
    mask = (data["vendor"] != "") & ~data["vendor"].map(
        lambda v: normalize_unseen_vendor(v) in full_registry)
    novel = data[mask].drop_duplicates(subset=["ip"]).reset_index(drop=True)
    print(f"真实新颖厂商设备: {len(novel)} 行 "
          f"(涉及 {novel['vendor'].nunique()} 个厂商)", flush=True)
    if novel.empty:
        raise SystemExit("没有可用样本, 终止")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    empty_prompt = build_unseen_type_vendor_classification_prompt(
        "", known_types, known_vendors_by_type
    )
    overhead = len(tokenizer.apply_chat_template(
        [{"role": "system", "content": CLASSIFIER_SYSTEM},
         {"role": "user", "content": empty_prompt}],
        tokenize=True, add_generation_prompt=True,
    ))
    budget = MAX_LEN - overhead - ANSWER_BUDGET
    summarizer = DeepSeekFingerprintSummarizer(
        LLM_CFG_PATH, CACHE_PATH, budget, tokenizer, max_workers=4)

    info_cols = [c for c in UNSEEN_INFO_COLS if c in novel.columns]
    out_path = os.path.join(SCRIPT_DIR, "dataset", "novel_vendor_validation.jsonl")
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        records = novel[info_cols].fillna("").astype(str).to_dict("records")
        fingerprints = [build_fingerprint_info_text(r) for r in records]
        for (fp_text, _sum), (_, row) in zip(
                summarizer.fit_many(fingerprints), novel.iterrows()):
            if not fp_text:
                continue
            prompt = build_unseen_type_vendor_classification_prompt(
                fp_text, known_types, known_vendors_by_type)
            out.write(json.dumps({
                "instruction": CLASSIFIER_SYSTEM,
                "input": prompt,
                "output": json.dumps({
                    "device_type": row["device_type"],
                    "device_vendor": row["vendor"],
                }),
                "metadata": {
                    "ip": str(row["ip"]),
                    "split_origin": "real_novel_vendor_validation",
                    "is_new_type": True,
                    "is_new_vendor": True,
                },
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"写出 {n} 条 -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

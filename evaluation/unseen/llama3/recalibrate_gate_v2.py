#!/usr/bin/env python
"""对既有 final_model 重跑校准+release gate (vendor novelty 真交叉修复版)。

不重训: 加载 results_v2/final_model 的 LoRA adapter, 用
validation_sft.jsonl + novel_vendor_validation.jsonl (真实新颖厂商补充集)
跑修复后的 cross_validate_open_set_thresholds, 按原四项门槛出诚实判决。

产物: results_v2/release_gate_report_v2_vendorfix.json (旧报告不动)。
"""

import importlib.util
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location(
    "fine_tune_v2", os.path.join(SCRIPT_DIR, "fine-tune-v2.py"))
ftv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ftv)

MODEL_ID = "/home/nfs/models/Llama-3.1-8B-Instruct"
ADAPTER = os.path.join(SCRIPT_DIR, "results_v2", "final_model")
VALID = os.path.join(SCRIPT_DIR, "dataset", "validation_sft.jsonl")
NOVEL = os.path.join(SCRIPT_DIR, "dataset", "novel_vendor_validation.jsonl")
OUT = os.path.join(SCRIPT_DIR, "results_v2", "release_gate_report_v2_vendorfix.json")
SAMPLES_PER_GROUP = 200


def load_rows():
    rows = []
    for path in (VALID, NOVEL):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kv = json.load(open(os.path.join(ADAPTER, "known_vendors.json")))
    known_types = kv["rag_device_types"]
    known_vendors_by_type = kv["known_vendors_by_type"]

    rows = load_rows()
    print(f"验证样本: {len(rows)} 条 (含真实新颖厂商补充集)", flush=True)

    tokenizer = ftv.AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    model.config.use_cache = True

    calibration = ftv.cross_validate_open_set_thresholds(
        model, tokenizer, rows, known_types, known_vendors_by_type,
        ftv.MAX_LEN, SAMPLES_PER_GROUP,
    )
    gates = ftv.evaluate_release_gates(
        calibration,
        min_type_f1=0.80,
        min_vendor_f1=0.80,
        min_type_exact_accuracy=0.50,
        min_vendor_exact_accuracy=0.50,
    )
    report = {
        "note": (
            "vendor novelty re-scored with real out-of-registry vendor "
            "positives (novel_vendor_validation.jsonl); ground truth "
            "recomputed as absolute vendor novelty to match the scoring "
            "scope. Supersedes the vendor columns of "
            "release_gate_report_v2.json, whose pseudo positives were "
            "unfailingly matched against the full registry (score forced "
            "to 0, recall 0 by construction)."
        ),
        "calibration": calibration,
        "release_gate": gates,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(json.dumps({
        "type_novelty_f1": gates["checks"]["type_novelty_f1"]["actual"],
        "vendor_novelty_f1": gates["checks"]["vendor_novelty_f1"]["actual"],
        "type_exact": gates["checks"]["type_exact_accuracy"]["actual"],
        "vendor_exact": gates["checks"]["vendor_exact_accuracy"]["actual"],
        "passed": gates["passed"],
    }, indent=2), flush=True)
    print(f"报告 -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

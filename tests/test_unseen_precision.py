import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

import agent as pipeline
import unseen
from llama_quantization import resolve_load_bits


class UnseenPrecisionTests(unittest.TestCase):
    def test_gpu_default_and_explicit_8bit_reach_model_and_adapter_loader(self):
        with tempfile.TemporaryDirectory() as adapter, \
                mock.patch("transformers.AutoTokenizer.from_pretrained"), \
                mock.patch("transformers.AutoModelForCausalLM.from_pretrained") as load, \
                mock.patch("peft.PeftModel.from_pretrained") as attach, \
                mock.patch.object(unseen.UnseenDeviceDetector, "_load_classification_metadata", return_value={}), \
                mock.patch.object(unseen.UnseenDeviceDetector, "_init_fingerprint_summarizer"), \
                mock.patch.object(torch.cuda, "is_available", return_value=True):
            for int8, method in ((False, "nf4"), (True, "llm_int8")):
                detector = unseen.UnseenDeviceDetector(
                    model_path="local-base", adapter_path=adapter, gpu=0, load_in_8bit=int8,
                )
                kwargs = load.call_args.kwargs
                self.assertEqual(kwargs["quantization_config"].quantization_method(), method)
                self.assertEqual(kwargs["device_map"], {"": 0})
                attach.assert_called_with(load.return_value, adapter)
                self.assertIs(detector.model, attach.return_value)

    def test_cpu_and_unquantized_opt_out_and_conflicting_requests(self):
        self.assertIsNone(resolve_load_bits(-1))
        self.assertIsNone(resolve_load_bits(0, load_in_4bit=False))
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            pipeline.IoTDecisionGraph(unseen_load_in_4bit=True, unseen_load_in_8bit=True)

    def test_read_only_decision_does_not_touch_the_graph(self):
        agent = pipeline.IdentificationAgent.__new__(pipeline.IdentificationAgent)
        agent.retrieval_agent = None
        agent.llm_type = "DEEPSEEK"
        with mock.patch("agent.IoTDecisionGraph") as graph, \
                mock.patch.object(agent, "load_test_fingerprints", return_value=[{"ip": "fixture"}]), \
                mock.patch.object(agent, "_save_decision_merge") as save, \
                mock.patch.object(agent, "_run_graph_update") as update:
            graph.return_value.classify.return_value = {"predicted_device_type": "CAMERA"}
            agent.run_decision(devices=["CAMERA"], unseen_load_in_8bit=True, gpu=0, update_graph=False)
            self.assertTrue(graph.call_args.kwargs["unseen_load_in_8bit"])
            save.assert_called_once()
            update.assert_not_called()

    def test_resume_refuses_stale_wrong_precision_and_unavailable_predictions(self):
        wanted = {"bits": 8, "adapter": "adapter-1"}
        with tempfile.TemporaryDirectory() as out, mock.patch.object(pipeline, "PREDICTION_DIR", out):
            records = []
            for name, config in [("match", wanted), ("wrong_bits", dict(wanted, bits=4)),
                                 ("wrong_adapter", dict(wanted, adapter="other")),
                                 ("legacy", None), ("unavailable", wanted)]:
                records.append({
                    "ip": name, "pipeline_version": pipeline.RETRIEVAL_SCHEMA_VERSION,
                    "predicted_device_type": "CAMERA", "unseen_config": config,
                    "first_stage": {"unseen": {"available": name != "unavailable"}},
                })
            (Path(out) / "CAMERA_type_prediction.json").write_text(json.dumps(records))
            agent = pipeline.IdentificationAgent.__new__(pipeline.IdentificationAgent)
            self.assertEqual(agent._quick_load_decision_done_ips("CAMERA", wanted), {"match"})

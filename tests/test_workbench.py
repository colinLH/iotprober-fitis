import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from demo.workbench import app as web
from demo.workbench import inputs, results, worker
from retrieval_contract import verified_example


class InputTests(unittest.TestCase):
    def test_unique_match_and_evidence_required(self):
        ip = "2001:db8::1"
        data = "ip,hw-products\n192.0.2.1,Other\n2001:db8:0:0:0:0:0:1,Camera\n"
        self.assertEqual(inputs.uploaded(ip, "test.csv", data.encode("utf-8-sig")),
                         {"ip": ip, "hw-products": "Camera"})
        with self.assertRaisesRegex(inputs.InputError, "多条"):
            inputs.uploaded(ip, "test.csv", (data + f"{ip},Router\n").encode())
        with self.assertRaisesRegex(inputs.InputError, "一致"):
            inputs.fingerprint(ip, {"ip": "192.0.2.1", "hw-products": "Camera"})
        with self.assertRaisesRegex(inputs.InputError, "指纹字段"):
            inputs.fingerprint(ip, {"ip": ip, "true_type": "CAMERA"})
        with self.assertRaisesRegex(inputs.InputError, "指纹字段"):
            inputs.fingerprint(ip, {"hw-info": "RTSP-554: [vendor: Unknown|product: Unknown]", "http-tags": "HTTP-80: []"})
        self.assertEqual(inputs.fingerprint(ip, {"cert-info": "TLS: tlsv1_3"})["cert-info"], "TLS: tlsv1_3")
        for text in ("127.0.0.1/24", "https://example.test/", "fe80::1%eth0"):
            with self.assertRaises(inputs.InputError):
                inputs.address(text)

    def test_censys_is_one_passive_request_with_shared_extraction(self):
        # No HTTP request leaves this test. An address validated as global is
        # needed to exercise the connector; keep the fixture unrelated to any IoT host.
        ip = "1.1.1.1"
        body = {"result": {"resource": {"ip": ip, "location": {"city": "Test"},
                "services": [{"port": 443, "protocol": "HTTPS", "hardware": [
                    {"vendor": "Fixture", "product": "Sensor"}]}]}}}
        with patch.object(inputs, "credentials", return_value=("server-PAT", "server-org")), \
                patch.object(inputs.requests, "get") as get:
            reply = get.return_value.__enter__.return_value
            reply.status_code = 200
            reply.iter_content.return_value = [json.dumps(body).encode()]
            fp = inputs.censys(ip, "one-use")
            self.assertEqual(fp, inputs.fingerprint(ip, {"result": {"hits": [{"host_v1": body["result"]}]}}))
            self.assertEqual(fp["hw-products"], "Sensor")
            self.assertEqual(get.call_args.args, ("https://api.platform.censys.io/v3/global/asset/host/1.1.1.1",))
            self.assertEqual(get.call_args.kwargs["params"], {})
            self.assertFalse(get.call_args.kwargs["allow_redirects"])
            self.assertNotIn("one-use", str(fp))
            reply.status_code = 429
            with self.assertRaisesRegex(inputs.InputError, "429"):
                inputs.censys(ip)
            with self.assertRaisesRegex(inputs.InputError, "公网"):
                inputs.censys("127.0.0.1")
            self.assertEqual(get.call_count, 2)


class WebTests(unittest.TestCase):
    def setUp(self):
        self.c = web.app.test_client()
        self.setup = self.c.get("/api/setup").get_json()
        self.h = {"X-Workbench-Token": self.setup["token"]}
        self.assertEqual(self.c.post("/api/runs", headers=self.h).status_code, 403)
        self.assertEqual(self.c.post("/api/unlock", data={"key":"invalid"}, headers=self.h).status_code, 403)
        self.assertEqual(self.c.post("/api/unlock", data={"key":web.app.config["LIVE_KEY"]}, headers=self.h).status_code, 200)

    def test_archive_replays_predictions_not_reference_labels(self):
        key = self.setup["samples"][0]
        actual = self.c.get(f"/api/archive/{key}").get_json()
        row, _ = results.archives()[key]
        self.assertEqual(actual["mode"], "archive")
        self.assertEqual(actual["final"]["device_type"], row["predicted_type"])
        self.assertEqual(actual["final"]["vendor"], row["predicted_vendor"])
        self.assertEqual(set(actual["stages"]), set(results.STAGES))
        self.assertEqual(self.c.get("/api/archive/not-a-sample").status_code, 404)

    def test_showcase_requires_positive_nonself_top1_and_preserves_failure_review(self):
        groups = self.setup["archive_groups"]
        self.assertEqual(self.setup["samples"], groups["verified"])
        for key in groups["verified"]:
            row, _ = results.archives()[key]
            self.assertTrue(verified_example(row))
            self.assertFalse(verified_example({**row, "predicted_type": "WRONG"}))
            local = row["retrieval_evidence"]["local"]
            self.assertFalse(verified_example({**row, "retrieval_evidence": {"local": {
                **local, "similar_devices": [{**local["similar_devices"][0], "ip": row["ip"]}]}}}))
        self.assertIn("CAMERA-03", groups["verified"])
        self.assertEqual(set(groups["errors"]), {"CAMERA-01", "CAMERA-02", "NAS-02"})
        self.assertTrue(set(groups["errors"]).isdisjoint(groups["verified"]))
        for key in groups["errors"]:
            run = self.c.get(f"/api/archive/{key}").get_json()
            self.assertTrue(run["review"]["observed"])
            self.assertIn(key, groups["all"])
        self.assertIsNone(self.c.get("/api/archive/NAS-02").get_json()["comparison"]["local_top1"])

    def test_no_external_request_or_worker_without_explicit_consent(self):
        data = dict(ip="192.0.2.3", mode="full", source="censys")
        with patch.object(web, "censys") as collect, patch.object(web, "supervise") as launch:
            self.assertEqual(self.c.post("/api/runs", data=data).status_code, 403)
            self.assertEqual(self.c.post("/api/runs", data=data, headers=self.h).status_code, 400)
            data.update(llm_consent="yes")
            self.assertEqual(self.c.post("/api/runs", data=data, headers=self.h).status_code, 400)
            collect.assert_not_called()
            launch.assert_not_called()
            self.assertEqual(self.c.get("/api/setup", headers={"Host": "other.test"}).status_code, 403)

    def test_upload_slot_ownership_and_release(self):
        def submitted():
            return self.c.post("/api/runs", headers=self.h, data={
                "ip": "192.0.2.3", "source": "upload", "mode": "local",
                "file": (io.BytesIO(b'{"hw-products":"Fixture"}'), "fp.json")})
        with patch.object(web.threading, "Thread") as thread:
            r = submitted()
            self.assertEqual(r.status_code, 202)
            jid = r.get_json()["id"]
            try:
                other = web.app.test_client()
                self.assertEqual(other.get(f"/api/runs/{jid}").status_code, 404)
                self.assertEqual(self.c.get(f"/api/runs/{jid}").get_json()["stages"]["input"]["state"], "done")
                self.assertEqual(submitted().status_code, 429)
                # Have the real supervisor take the failure/cleanup path.
                with patch.object(web.subprocess, "run", side_effect=OSError):
                    web.supervise(*thread.call_args.kwargs["args"])
                self.assertEqual(self.c.get(f"/api/runs/{jid}").get_json()["state"], "failed")
            finally:
                with web.lock:
                    web.jobs.pop(jid, None)


class FullFlowTests(unittest.TestCase):
    def test_real_decision_graph_uses_private_evidence_and_streams_final_vote(self):
        # Actual LangGraph nodes, shared retrieval tool and joint_vote. Substitute
        # only model I/O and the large knowledge base; network and GPU are not needed.
        agentdir = str(inputs.ROOT / "agent")
        if agentdir not in sys.path:
            sys.path.insert(0, agentdir)
        import agent
        import decision
        import path_config
        import retrieval
        from langchain_core.messages import AIMessage
        fp = {"ip": "192.0.2.42", "hw-products": "fixture camera"}
        cache = retrieval.build_cache_metadata(fp, 5)
        common = {"query_fingerprint": fp, "cache_metadata": cache}
        client = MagicMock()
        def answer(messages):
            if not any(getattr(m, "type", "") == "tool" for m in messages):
                return AIMessage(content="", tool_calls=[{"name": "configurable_multi_level_retrieval",
                    "args": {"ip": fp["ip"]}, "id": "read-fixture"}])
            self.assertIn("CAMERA", messages[-1].content)
            return AIMessage(content=json.dumps({"device_type": "CAMERA", "vendor": "Fixture", "confidence": 0.8}))
        client.bind_tools.return_value.invoke.side_effect = answer
        synthetic = MagicMock()
        synthetic.detect_unseen.return_value = {"classified_type": "CAMERA", "classified_vendor": "Fixture",
                                                "new_type_probability": 0.1, "needs_web_search": False}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "local_npz").mkdir()
            (folder / "local_npz/FIXTURE_embeddings.npy").touch()
            (folder / "config.json").write_text("{}")
            cfg = folder / "services.json"
            cfg.write_text(json.dumps({k: {"MODEL":"fixture", "BASE_URL":"http://invalid.invalid",
                                           "PROTOCOL":"openai", "API_KEY":"unused"} for k in ("GEMINI","CLAUDE")}))
            run = results.record(fp["ip"], "upload", "full")
            with patch.object(path_config, "LOCAL_VECTOR_DB_DIR", tmp), patch.object(path_config, "DEFAULT_LLM_MODEL_DIR", tmp), \
                    patch.object(retrieval, "QUERY_DB_DIR", tmp), patch.object(decision, "_QDB_PATH", tmp), \
                    patch.object(agent, "LLM_CONFIG_FILE", str(cfg)), patch("langchain_openai.ChatOpenAI", return_value=client), \
                    patch.object(agent.IoTDecisionGraph, "_get_unseen_detector", return_value=synthetic), \
                    patch.object(agent.IoTDecisionGraph, "_get_drift_detector") as drift, \
                    patch.object(worker, "memory_available", return_value=64*1024**3), \
                    patch.object(retrieval.MultiLevelRetrieval, "__init__", return_value=None), \
                    patch.object(retrieval.MultiLevelRetrieval, "clear_history"), \
                    patch.object(retrieval.MultiLevelRetrieval, "local_retrieval", return_value={
                        **common, "similar_devices":[{"device_type":"CAMERA","similarity_score":0.8}]}), \
                    patch.object(retrieval.MultiLevelRetrieval, "community_retrieval", return_value={**common, "matched_clusters":[]}), \
                    patch.object(retrieval.MultiLevelRetrieval, "reasoning_path_retrieval", return_value={**common, "path_matching_results":[]}):
                drift.return_value.detect_query_device.return_value = {"drift_score":0.2, "threshold":0.8, "is_drift":False}
                published = []
                worker.process(run, fp, lambda r: published.append({k:v["state"] for k,v in r["stages"].items()}))
            self.assertEqual(run["state"], "complete", run.get("error"))
            self.assertEqual(run["final"]["device_type"], "CAMERA")
            self.assertEqual(run["final"]["vendor"], "Fixture")
            self.assertEqual(run["stages"]["unseen"]["data"]["classified_type"], "CAMERA")
            self.assertEqual(run["stages"]["drift"]["state"], "done")
            self.assertTrue(any(p["branches"] == "running" for p in published))
            self.assertFalse(synthetic.detect_unseen.call_args.kwargs["allow_web_search"])
            self.assertTrue((folder / "local/WORKBENCH_local.json").exists())


if __name__ == "__main__":
    unittest.main()

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT = os.path.join(ROOT, "agent")
for path in (ROOT, AGENT):
    if path not in sys.path:
        sys.path.insert(0, path)

import retrieval  # noqa: E402
import singleton_reports
import decision  # noqa: E402
import agent as agent_module  # noqa: E402
import util  # noqa: E402
from graph import cluster, vector


def bare_retriever(directory, perspectives=("p1", "p2"), embedding_dim=2):
    obj = retrieval.MultiLevelRetrieval.__new__(retrieval.MultiLevelRetrieval)
    obj.local_npz_dir = directory
    obj.retrieval_perspective_names = list(perspectives)
    obj.embedding_dim = embedding_dim
    obj.embedding_overall_dim = len(perspectives) * embedding_dim
    obj._validated_vector_files = set()
    obj.index_columns = {p: [p] for p in perspectives}
    obj.perspective_info_config = {p: {"cols": [p]} for p in perspectives}
    obj.perspective_weights = [1 / len(perspectives)] * len(perspectives)
    obj.llm = None
    obj.used_llm_model = "DEEPSEEK"
    obj._cluster_assignment_cache = {}
    obj._cluster_assignment_errors = {}
    obj._cluster_report_cache = {}
    obj._statistical_report_cache = {}
    obj._single_assignment_cache = {}
    obj._single_summary_cache = {}
    obj.retrieval_history = []
    return obj


class RetrievalIntegrityTests(unittest.TestCase):
    def test_llm_fingerprint_payload_is_bounded_without_mutating_source(self):
        fingerprint = {
            "ip": "1.2.3.4",
            "http-bodys": "A" * 50_000,
            "cert-info": "B" * 12_000,
            "sw-info": "C" * 12_000,
        }

        trimmed = util.truncate_fingerprint_for_llm(
            fingerprint, total_value_limit=4_000
        )

        self.assertEqual(len(fingerprint["http-bodys"]), 50_000)
        self.assertLessEqual(
            sum(len(str(value)) for value in trimmed.values()), 4_000
        )
        self.assertIn("truncated", trimmed["http-bodys"])
        self.assertEqual(trimmed["ip"], "1.2.3.4")

    def test_empty_local_index_preserves_search_return_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = bare_retriever(directory)
            rows, compared, device_best = obj._search_local_vectors(
                np.zeros(4, dtype=np.float32), top_k=5
            )

            self.assertEqual(rows, [])
            self.assertEqual(compared, 0)
            self.assertEqual(device_best, [])

    def test_simple_similarity_does_not_count_ip_as_evidence(self):
        obj = bare_retriever("unused")

        score = obj._simple_similarity(
            {"ip": "query", "os": "Linux"},
            {"ip": "other", "os": "Linux"},
        )

        self.assertEqual(score, 1.0)

    def test_corrupt_cluster_assignments_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = bare_retriever(directory)
            obj.com_view_path = directory
            with open(
                os.path.join(directory, "ipraw_POWER_METER_embedding_overall_pca.csv"),
                "wb",
            ) as handle:
                handle.write(b"\x00\x00not-a-csv")

            result = obj.community_retrieval(
                {"ip": "query"},
                [{"ip": "candidate", "device_type": "POWER_METER"}],
                cache_metadata={"schema_version": "test"},
            )

            self.assertEqual(result["matched_clusters"], [])
            self.assertEqual(
                result["unavailable_clusters"][0]["reason"],
                "cluster_assignments_invalid",
            )

    def test_local_search_is_weighted_cosine_and_excludes_query_ip(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = bare_retriever(directory)
            embeddings = np.array([
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
            ], dtype=np.float32)
            np.save(os.path.join(directory, "CAMERA_embeddings.npy"), embeddings)
            np.save(os.path.join(directory, "CAMERA_ips.npy"), np.array(["query", "other"]))

            query = np.array([0.75, 0.0, 0.25, 0.0], dtype=np.float32)
            rows, compared, device_best = obj._search_local_vectors(
                query, top_k=2, exclude_ips={"query"}
            )

            self.assertEqual(compared, 2)
            self.assertEqual([row["ip"] for row in rows], ["other"])
            self.assertEqual(rows[0]["similarity_score"], 0.75)
            self.assertGreaterEqual(rows[0]["similarity_score"], -1.0)
            self.assertLessEqual(rows[0]["similarity_score"], 1.0)
            self.assertEqual([row["ip"] for row in device_best], ["other"])

    def test_cache_is_bound_to_full_fingerprint_and_top_k(self):
        fingerprint = {"ip": "1.2.3.4", "os-vendor": "vendor-a"}
        record = {"cache_metadata": retrieval.build_cache_metadata(fingerprint, 5)}

        self.assertTrue(retrieval.cache_record_matches(record, fingerprint, 5))
        self.assertFalse(retrieval.cache_record_matches(
            record, {"ip": "1.2.3.4", "os-vendor": "vendor-b"}, 5
        ))
        self.assertFalse(retrieval.cache_record_matches(record, fingerprint, 10))
        self.assertFalse(retrieval.cache_record_matches({}, fingerprint, 5))

    def test_community_marks_unprepared_singleton_and_missing_normal_report_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = bare_retriever(directory)
            obj.com_view_path = os.path.join(directory, "overall")
            os.makedirs(obj.com_view_path)
            pd.DataFrame([
                {"ip": "noise", "cluster": -1},
                {"ip": "member", "cluster": 0},
            ]).to_csv(
                os.path.join(obj.com_view_path, "ipraw_CAMERA_embedding_overall_pca.csv"),
                index=False,
            )

            raw_dir = Path(directory) / "rag"
            raw_dir.mkdir()
            (raw_dir / "ipraw_CAMERA.csv").write_text("ip,p1\nnoise,profile\n")
            with mock.patch.object(retrieval, "CSV_DATA_DIR", directory):
                result = obj.community_retrieval(
                    {"ip": "query"},
                    [{"ip": "noise", "device_type": "CAMERA", "similarity_score": 0.85},
                     {"ip": "member", "device_type": "CAMERA"}],
                    cache_metadata={"schema_version": "test"},
                )

            self.assertEqual(result["matched_clusters"], [])
            self.assertEqual(
                {row["reason"] for row in result["unavailable_clusters"]},
                {"singleton_report_not_prepared", "cluster_report_missing"},
            )

    def test_index_text_observed_mask_and_abstention_use_same_fields(self):
        obj = bare_retriever("unused", perspectives=("hw", "fav"))
        obj.index_columns = {"hw": ["hw-info"], "fav": ["http-favicon-urls"]}
        obj.perspective_info_config = {"hw": {"cols": ["hw-products"]}, "fav": {"cols": ["http-favicon-urls"]}}
        obj.embedding_model = mock.Mock()
        obj.embedding_model.embed_query.return_value = [1.0, 0.0]
        fp = {"hw-products": "CAM", "hw-info": "RTSP-554: [product: CAM]", "http-favicon-urls": "HTTP-80: []"}
        with mock.patch.object(obj, "_search_local_vectors", return_value=([], 0, [])) as search:
            r = obj.local_retrieval(fp)
            self.assertEqual(r["missing_perspectives"], ["fav"])
            obj.embedding_model.embed_query.assert_called_once_with(fp["hw-info"])
            np.testing.assert_array_equal(search.call_args.args[0], [1, 0, 0, 0])
            self.assertEqual(obj._view_texts({"hw-products": "CAM"}), ["product: CAM", ""])
            fp.update({"hw-products": "unknown", "hw-info": "RTSP-554: [vendor: Unknown|product: Unknown]"})
            self.assertEqual(obj.local_retrieval(fp)["reason"], "no_observed_perspectives")
            self.assertEqual(search.call_count, 1)
            self.assertEqual(obj.embedding_model.embed_query.call_count, 1)

    def test_isolated_index_build_preserves_noise_labels_and_query_representation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ipraw_CAMERA.csv").write_text("ip,hw-info,hw-products\n"
                "q,product: camera,other\na,product: camera,alternate\nb,Unknown|Unknown,unknown\n")
            builder = cluster.GraphClustering.__new__(cluster.GraphClustering)
            builder.gpu, builder.save_path, builder.device_labels = -1, directory, ["CAMERA"]
            builder.perspective_info_config = {"hw": {"cols": ["hw-products"]}}
            emb = mock.Mock()
            emb.embed_query.return_value = [1.0] + [0.0] * 1023
            builder.embedding_model = emb
            with mock.patch.object(cluster.hdbscan, "HDBSCAN") as hdb, \
                    mock.patch.object(cluster.joblib, "dump"), mock.patch.object(builder, "draw_cluster_2D"):
                hdb.return_value.fit_predict.return_value = np.full(3, -1)
                builder.cluster_embedding(["hw-info"], "hw", whether_pca=False)
            self.assertEqual(emb.embed_query.call_args_list, [mock.call("product: camera")] * 2)
            stored = pd.read_csv(root / "embedding_hw/ipraw_CAMERA_embedding_hw_cluster.csv")
            self.assertEqual(stored.pop("cluster").tolist(), [-1, -1, -1])
            self.assertEqual(json.loads((root / "embedding_hw/cluster_info.json").read_text()), {"CAMERA": {}})
            source = root / "vectors.csv"
            stored.rename(columns=lambda k: k.replace("embedding", "hw")).to_csv(source, index=False)
            vector.insert_csv_to_local_npz("CAMERA", str(source), ["hw"], npz_dir=directory)
            obj = bare_retriever(directory, perspectives=("hw",), embedding_dim=1024)
            obj.index_columns, obj.perspective_info_config = {"hw": ["hw-info"]}, builder.perspective_info_config
            obj.embedding_model = emb
            out = obj.local_retrieval({"ip": "q", "hw-info": "product: camera", "hw-products": "different"})
            hits = out["similar_devices"]
            self.assertEqual([(r["ip"], r["similarity_score"]) for r in hits], [("a", 1.0), ("b", 0.0)])

    def test_outliers_get_separate_lazy_persistent_member_reports_not_query_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            obj = bare_retriever(directory, perspectives=("hw",))
            obj.com_view_path = directory
            obj.index_columns = {"hw": ["hw-info"]}
            (root / "ipraw_CAMERA_embedding_overall_pca.csv").write_text(
                "ip,cluster\nN1,-1\nN2,-1\nN3,-1\nM,1\n")
            (root / "CAMERA_cluster_summaries.json").write_text(json.dumps(
                [{"cluster_id": 1, "analysis": {"common_patterns": {"hw": "known"}}}]))
            (root / "rag").mkdir()
            (root / "rag/ipraw_CAMERA.csv").write_text("ip,hw-info\nN1,first\nN2,second\n")
            def reply(model, messages, **kwargs):
                self.assertEqual(kwargs["thinking"], "disabled")
                if "exactly one member" in messages[0]["content"]:
                    member = json.loads(messages[1]["content"])
                    self.assertIn(member["hw"], ("first", "second"))
                    return {"common_patterns": member}
                return {"similarity_score": 0.7, "matched_features": ["hardware"], "unmatched_features": []}
            obj.llm = mock.Mock(llm_config={})
            obj.llm.chat_with_llm.side_effect = reply
            candidates = [{"device_type": "CAMERA", "ip": ip, "similarity_score": 0.83}
                          for ip in ("N1", "N2", "N1", "N3", "M")]
            with mock.patch.object(retrieval, "CSV_DATA_DIR", directory), \
                    mock.patch.dict(os.environ, {"IOTPROBER_SINGLETON_REPORT_DIR": str(root / "cache")}):
                for q, score in (("query1", 0.83), ("query2", 0.91)):
                    for c in candidates:
                        c["similarity_score"] = score
                    out = obj.community_retrieval({"ip": q, "hw-info": "not in KB"}, candidates)
                    self.assertEqual([u["reason"] for u in out["unavailable_clusters"]], ["singleton_fingerprint_missing"])
                    single = [r for r in out["matched_clusters"] if r["is_outlier"]]
                    self.assertEqual(len({r["community_key"] for r in single}), 2)
                    self.assertEqual({r["report"]["common_patterns"]["hw"] for r in single}, {"first", "second"})
                    self.assertTrue(all(r["cluster_size"] == 1 and len(r["related_ips"]) == 1 for r in single))
                    self.assertTrue(all(r["score_source"] == "singleton_member_cosine"
                                        and r["similarity_score"] == score for r in single))
                    self.assertEqual(obj.reasoning_path_retrieval({}, {"matched_clusters": single})["summary"]["reason"],
                                     "singleton_has_no_population_path")
                with mock.patch.object(singleton_reports, "LLM_CONFIG_FILE", root / "model.json"):
                    (root / "model.json").write_text("{}")
                    calls = obj.llm.chat_with_llm
                    obj.llm = None
                    cached = obj.community_retrieval({"ip": "query3"}, candidates[:2])
                    self.assertEqual(len(cached["matched_clusters"]), 2)
            # Two one-time singleton summaries + one normal cluster match per online query.
            self.assertEqual(calls.call_count, 4)
            self.assertEqual(len(list((root / "cache").glob("*.json"))), 2)

    def test_invalid_match_retries_once_and_below_threshold_is_not_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = bare_retriever(directory)
            obj.com_view_path = directory
            (Path(directory) / "ipraw_CAMERA_embedding_overall_pca.csv").write_text("ip,cluster\nm,2\n")
            obj._cluster_report_cache["CAMERA"] = {2: "archived summary"}
            obj.llm = mock.Mock()
            obj.llm.chat_with_llm.side_effect = ["invalid", json.dumps({
                "matched_features": [], "unmatched_features": ["model"], "similarity_score": 0.4})]
            r = obj.community_retrieval({"ip": "q", "hw-info": "camera"}, [{"ip": "m", "device_type": "CAMERA"}])
            self.assertEqual(r["matched_clusters"], [])
            self.assertEqual(r["unavailable_clusters"], [])
            self.assertEqual(r["below_threshold"][0]["similarity_score"], 0.4)
            obj.llm.chat_with_llm.side_effect = ValueError("bad output")
            r = obj.community_retrieval({"ip": "q"}, [{"ip": "m", "device_type": "CAMERA"}])
            self.assertFalse(r["below_threshold"])
            self.assertTrue(r["unavailable_clusters"][0]["reason"].startswith("llm_match_failed"))
            self.assertEqual(obj.llm.chat_with_llm.call_count, 4)

    def test_path_with_no_comparable_features_never_returns_a_winner(self):
        obj = bare_retriever("unused", perspectives=("hw",))
        obj.index_columns = {"hw": ["hw-info"]}
        candidate = {"cluster_id": 3, "device_type": "CAMERA"}
        importance = {"CAMERA_3": {"cluster_info": candidate, "important_features": ["hw"],
                      "feature_importance": {"hw": {"importance_score": 1, "num_clusters": 1,
                      "feature_values": [{"analysis": {"common_patterns": {"hw": "camera"}}}]}}}}
        with mock.patch.object(obj, "calculate_importance", return_value=importance):
            result = obj.reasoning_path_retrieval({"query_fingerprint": {"hw": "Unknown"}},
                                                  {"matched_clusters": [candidate]})
        self.assertEqual(result["path_matching_results"], [])
        self.assertIsNone(result["summary"]["top_cluster"])

    def test_importance_uses_complete_cluster_not_local_hit_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = bare_retriever(directory, perspectives=("p",), embedding_dim=2)
            obj.com_view_path = os.path.join(directory, "overall")
            obj.single_view_path = os.path.join(directory, "single")
            os.makedirs(obj.com_view_path)
            single_dir = os.path.join(obj.single_view_path, "embedding_p")
            os.makedirs(single_dir)

            pd.DataFrame([
                {"ip": "a", "cluster": 7},
                {"ip": "b", "cluster": 7},
                {"ip": "c", "cluster": 7},
            ]).to_csv(
                os.path.join(obj.com_view_path, "ipraw_CAMERA_embedding_overall_pca.csv"),
                index=False,
            )
            pd.DataFrame([
                {"ip": "a", "cluster": 2},
                {"ip": "b", "cluster": 2},
                {"ip": "c", "cluster": 2},
                {"ip": "outside", "cluster": -1},
            ]).to_csv(
                os.path.join(single_dir, "ipraw_CAMERA_embedding_p_pca.csv"),
                index=False,
            )
            with open(os.path.join(single_dir, "CAMERA_cluster_summaries.json"), "w") as handle:
                json.dump([{
                    "cluster_id": 2,
                    "analysis": {"common_patterns": {"field": "value"}},
                }], handle)

            result = obj.calculate_importance([{
                "device_type": "CAMERA",
                "cluster_id": 7,
                "related_ips": ["a"],
            }])

            cluster = result["CAMERA_7"]
            self.assertEqual(cluster["cluster_size"], 3)
            self.assertEqual(cluster["important_features"], ["p"])
            self.assertEqual(cluster["feature_importance"]["p"]["support"], 3)

    def test_missing_summary_uses_statistical_report_when_raw_members_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = bare_retriever(directory, perspectives=("p",), embedding_dim=2)
            obj.com_view_path = os.path.join(directory, "overall")
            obj.perspective_info_config = {"p": {"cols": ["field"]}}
            os.makedirs(obj.com_view_path)
            pd.DataFrame([
                {"ip": "a", "cluster": 3},
                {"ip": "b", "cluster": 3},
                {"ip": "c", "cluster": 3},
            ]).to_csv(
                os.path.join(obj.com_view_path, "ipraw_CAMERA_embedding_overall_pca.csv"),
                index=False,
            )
            all_dir = os.path.join(directory, "all")
            os.makedirs(all_dir)
            pd.DataFrame([
                {"ip": "a", "field": "same"},
                {"ip": "b", "field": "same"},
                {"ip": "c", "field": "other"},
            ]).to_csv(os.path.join(all_dir, "ipraw_CAMERA.csv"), index=False)
            obj._match_fingerprint_with_cluster = lambda *args, **kwargs: {
                "available": True,
                "similarity_score": 0.8,
                "matched_features": ["field"],
                "unmatched_features": [],
            }

            with mock.patch.object(retrieval, "CSV_DATA_DIR", directory):
                result = obj.community_retrieval(
                    {"ip": "query", "field": "same"},
                    [{"ip": "a", "device_type": "CAMERA"}],
                    cache_metadata={"schema_version": "test"},
                )

            self.assertEqual(len(result["matched_clusters"]), 1)
            match = result["matched_clusters"][0]
            self.assertEqual(match["report_source"], "deterministic_frequency_summary")
            self.assertEqual(match["cluster_size"], 3)
            self.assertEqual(match["report"]["sample_size"], 3)

    def test_decision_tool_does_not_expose_filename_label(self):
        with tempfile.TemporaryDirectory() as directory:
            local_dir = os.path.join(directory, "local")
            os.makedirs(local_dir)
            with open(os.path.join(local_dir, "CAMERA_local.json"), "w") as handle:
                json.dump([{
                    "query_fingerprint": {"ip": "1.2.3.4"},
                    "top_k": 5,
                    "similar_devices": [],
                    "confidence_score": 0.5,
                    "missing_perspectives": ["dns"],
                }], handle)

            with mock.patch.object(decision, "_QDB_PATH", directory), \
                    mock.patch.object(decision, "_dev_labels", ["CAMERA"]), \
                    mock.patch.object(decision, "_retrieval_runtime", None):
                section = decision._local_section("1.2.3.4")

            self.assertEqual(section["status"], "found")
            self.assertNotIn("candidate_dev", section)
            self.assertEqual(section["missing_perspectives"], ["dns"])

    def test_joint_vote_never_boosts_uncalibrated_confidence(self):
        gemini = {
            "device_type": "CAMERA", "vendor": "A", "confidence": 1.0, "llm": "G"
        }
        claude = {
            "device_type": "CAMERA", "vendor": "A", "confidence": 1.0, "llm": "C"
        }
        agreed = decision.DecisionAgent._joint_vote(gemini, claude)
        self.assertEqual(agreed["final_confidence"], 0.89)

        claude["device_type"] = "SCADA"
        disagreed = decision.DecisionAgent._joint_vote(gemini, claude)
        self.assertEqual(disagreed["final_confidence"], 0.69)

    def test_vendor_vote_uses_vendor_confidence_independently(self):
        gemini = {
            "device_type": "CAMERA", "vendor": "Vendor-G",
            "type_confidence": 0.9, "vendor_confidence": 0.2, "llm": "gemini",
        }
        claude = {
            "device_type": "ROUTER", "vendor": "Vendor-C",
            "type_confidence": 0.4, "vendor_confidence": 0.8, "llm": "claude",
        }

        result = decision.DecisionAgent._joint_vote(gemini, claude)

        self.assertEqual(result["final_device_type"], "CAMERA")
        self.assertEqual(result["winning_llm"], "gemini")
        self.assertEqual(result["final_vendor"], "Vendor-C")
        self.assertEqual(result["winning_vendor_llm"], "claude")
        self.assertLessEqual(result["final_vendor_confidence"], 0.69)

    def test_vendor_novelty_does_not_disable_type_drift_check(self):
        graph = agent_module.IoTDecisionGraph.__new__(agent_module.IoTDecisionGraph)
        graph.enable_first_stage = True
        update = graph._gate_node({
            "unseen_result": {
                "new_type_probability": 0.1,
                "new_vendor_probability": 0.99,
            }
        })
        self.assertTrue(update["first_stage"]["drift_checked"])


if __name__ == "__main__":
    unittest.main()

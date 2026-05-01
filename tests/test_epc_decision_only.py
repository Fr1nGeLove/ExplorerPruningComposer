import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import epc_decision_only as epc  # noqa: E402
import epc_hybrid_gated_eval as hybrid  # noqa: E402
from gendr import UnitTestOracle  # noqa: E402


class FakeOracle:
    def __init__(self, passing_codes):
        self.passing_codes = set(passing_codes)
        self.calls = []

    def check_one(self, task_id, solution):
        self.calls.append((task_id, solution))
        return solution in self.passing_codes


class FakeHandler:
    def __init__(self):
        self.verify_calls = 0

    def build_verify_unit_test(self, log_file_prefix, results, sol_field="solution"):
        return log_file_prefix + ".jsonl"

    def save_formatted_gt(self, log_file_prefix, data):
        return log_file_prefix + ".jsonl"

    def verify_unit_test(self, verify_file, gt_file=None, timeout_per_task=20, timeout=1800):
        self.verify_calls += 1
        return ["Task/1__gendr_case_1"], ["Task/1__gendr_case_0"], ["AssertionError: expected 4"]


class TestEPCDecisionOnly(unittest.TestCase):
    def test_selects_smallest_passing_pruned_candidate(self):
        candidates = [
            {
                "explorer_id": "E3",
                "pruned_edit_size": 4,
                "pruned_repair_code": "large",
                "gendr_meta": {"final_passed": True},
            },
            {
                "explorer_id": "E2",
                "pruned_edit_size": 2,
                "pruned_repair_code": "small",
                "gendr_meta": {"final_passed": True},
            },
            {
                "explorer_id": "E1",
                "pruned_edit_size": 1,
                "pruned_repair_code": "broken",
                "gendr_meta": {"final_passed": False},
            },
        ]

        selected = epc._choose_best_passing_pruned_candidate(candidates)

        self.assertEqual(selected["explorer_id"], "E2")

    def test_falls_back_when_composer_final_code_fails(self):
        oracle = FakeOracle(passing_codes={"fallback"})
        pruned_candidates = [
            {
                "explorer_id": "E1",
                "explorer_role": "Minimal Explorer",
                "pruned_edit_size": 3,
                "pruned_repair_code": "fallback",
                "pruned_diff": {"2": {"type": "Modify", "original": "bad", "modified": "good"}},
                "gendr_meta": {"final_passed": True},
            }
        ]

        final_code, final_status, final_pass, meta = epc._apply_final_test_fallback(
            task_id="Task/1",
            final_code="composer-bad",
            final_status="ok",
            pruned_candidates=pruned_candidates,
            oracle=oracle,
        )

        self.assertEqual(final_code, "fallback")
        self.assertEqual(final_status, "fallback_best_pruned_due_to_final_test_failure")
        self.assertTrue(final_pass)
        self.assertTrue(meta["used"])
        self.assertEqual(meta["fallback_explorer_id"], "E1")
        self.assertEqual(oracle.calls, [("Task/1", "composer-bad"), ("Task/1", "fallback")])

    def test_resolves_multi_patch_granularity_to_hunk_by_default(self):
        self.assertEqual(epc._resolve_patch_granularity(None, "multi"), "hunk")
        self.assertEqual(epc._resolve_patch_granularity(None, "single"), "line")
        self.assertEqual(epc._resolve_patch_granularity("line", "multi"), "line")

    def test_feedback_retry_mode_preserves_minimal_or_free_family(self):
        self.assertEqual(epc._feedback_retry_mode("minimal"), "minimal_with_feedback")
        self.assertEqual(epc._feedback_retry_mode("free"), "free_with_feedback")

    def test_formats_failed_attempts_for_retry_prompt(self):
        attempts = [
            {
                "explorer_id": "E1",
                "explorer_role": "Minimal Explorer",
                "raw_diff": {"3": {"type": "Modify", "original": "bad", "modified": "wrong"}},
                "unit_feedback": "AssertionError: expected 4",
            }
        ]

        feedback = epc._format_failed_attempts_for_feedback(attempts)

        self.assertIn("E1", feedback)
        self.assertIn("Minimal Explorer", feedback)
        self.assertIn("AssertionError", feedback)

    def test_builds_partial_patch_bank_from_failed_candidate_consensus(self):
        buggy_code = "def f():\n    value = 1\n    return value\n"
        shared_diff = {
            "2": {"type": "Modify", "original": "    value = 1", "modified": "    value = 2"}
        }
        noise_diff = {
            "3": {"type": "Modify", "original": "    return value", "modified": "    return value + 1"}
        }
        candidates = [
            {
                "explorer_id": "E1",
                "explorer_role": "Minimal Explorer",
                "unit_pass": False,
                "raw_diff": shared_diff,
            },
            {
                "explorer_id": "E2",
                "explorer_role": "Boundary Explorer",
                "unit_pass": False,
                "raw_diff": shared_diff,
            },
            {
                "explorer_id": "E3",
                "explorer_role": "Data-flow Explorer",
                "unit_pass": False,
                "raw_diff": noise_diff,
            },
        ]

        patches = epc._build_partial_patch_bank_from_failed_candidates(
            buggy_code=buggy_code,
            candidates=candidates,
            patch_granularity="line",
            min_support=2,
        )

        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0]["diff"], shared_diff)
        self.assertEqual(patches[0]["partial_support_count"], 2)
        self.assertEqual(patches[0]["partial_support_explorer_ids"], ["E1", "E2"])
        self.assertIn("failed candidates", " ".join(patches[0]["candidate_local_evidence"]))

    def test_choose_best_partial_candidate_prefers_test_score_then_small_edit(self):
        candidates = [
            {
                "explorer_id": "E1",
                "raw_repair_code": "large-but-better-tests",
                "raw_diff": {
                    "2": {"type": "Modify", "original": "bad", "modified": "good"},
                    "3": {"type": "Modify", "original": "x", "modified": "y"},
                },
                "raw_edit_size": 2,
                "unit_pass": False,
                "num_tests_passed": 2,
                "num_tests_total": 3,
            },
            {
                "explorer_id": "E2",
                "raw_repair_code": "small-but-worse-tests",
                "raw_diff": {"2": {"type": "Modify", "original": "bad", "modified": "good"}},
                "raw_edit_size": 1,
                "unit_pass": False,
                "num_tests_passed": 0,
                "num_tests_total": 3,
            },
        ]

        selected = epc._choose_best_partial_candidate(candidates)

        self.assertEqual(selected["explorer_id"], "E1")

    def test_hybrid_falls_back_to_best_partial_candidate_before_buggy(self):
        buggy_code = "def f():\n    return 1\n"
        partial_code = "def f():\n    return 2\n"
        item = {
            "task_id": "Task/partial",
            "buggy_code": buggy_code,
            "debug_results": {"solution": buggy_code},
            "epc": {
                "final_pass": False,
                "pruned_candidates": [],
                "explorer_candidates": [
                    {
                        "explorer_id": "E1",
                        "raw_repair_code": partial_code,
                        "raw_diff": {
                            "2": {
                                "type": "Modify",
                                "original": "    return 1",
                                "modified": "    return 2",
                            }
                        },
                        "raw_edit_size": 1,
                        "unit_pass": False,
                    }
                ],
            },
        }

        results, summary = hybrid.build_hybrid_results([item])

        self.assertEqual(results[0]["debug_results"]["solution"], partial_code)
        self.assertEqual(results[0]["epc_hybrid"]["source"], "best_partial_failed_explorer")
        self.assertEqual(summary["partial_fallback"], 1)
        self.assertEqual(summary["buggy_fallback"], 0)

    def test_oracle_returns_feedback_alongside_pass_flags(self):
        oracle = object.__new__(UnitTestOracle)
        oracle.dataset_name = "fake"
        oracle.handler = FakeHandler()
        oracle.log_dir = "/tmp"
        oracle.timeout_per_task = 20
        oracle.timeout = 1800
        oracle._cache = {}
        oracle._feedback_cache = {}
        oracle._run_id = 0
        oracle.oracle_calls = 0

        results = oracle.check_many_with_feedback("Task/1", ["passing", "failing"])

        self.assertEqual(results[0], {"passed": True, "feedback": ""})
        self.assertEqual(results[1], {"passed": False, "feedback": "AssertionError: expected 4"})
        self.assertEqual(oracle.handler.verify_calls, 1)

    def test_oracle_short_circuits_empty_solutions(self):
        oracle = object.__new__(UnitTestOracle)
        oracle.dataset_name = "fake"
        oracle.handler = FakeHandler()
        oracle.log_dir = "/tmp"
        oracle.timeout_per_task = 20
        oracle.timeout = 1800
        oracle._cache = {}
        oracle._feedback_cache = {}
        oracle._run_id = 0
        oracle.oracle_calls = 0

        results = oracle.check_many_with_feedback("Task/1", ["", "   "])

        self.assertEqual(
            results,
            [
                {"passed": False, "feedback": "empty solution"},
                {"passed": False, "feedback": "empty solution"},
            ],
        )
        self.assertEqual(oracle.handler.verify_calls, 0)
        self.assertEqual(oracle.oracle_calls, 0)


if __name__ == "__main__":
    unittest.main()

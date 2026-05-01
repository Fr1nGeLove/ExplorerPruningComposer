import json
import os
import tempfile
import unittest
from unittest.mock import patch

from lcb_runner.benchmarks.code_generation import load_code_generation_dataset_subset


def _make_row(question_id: str) -> dict:
    return {
        "question_title": f"title-{question_id}",
        "question_content": "content",
        "platform": "leetcode",
        "question_id": question_id,
        "contest_id": "contest",
        "contest_date": "2023-06-01T00:00:00",
        "starter_code": "class Solution:\n    pass",
        "difficulty": "easy",
        "public_test_cases": json.dumps(
            [{"input": "1", "output": "1", "testtype": "functional"}]
        ),
        "private_test_cases": json.dumps(
            [{"input": "2", "output": "2", "testtype": "functional"}]
        ),
        "metadata": json.dumps({"func_name": "solve"}),
    }


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


class TestCodeGenerationSubsetLoader(unittest.TestCase):
    def test_local_first_without_hf_when_all_ids_present(self):
        with tempfile.TemporaryDirectory() as td:
            local_file = os.path.join(td, "release_v1_full.jsonl")
            _write_jsonl(local_file, [_make_row("1001"), _make_row("1002")])

            with patch.dict(os.environ, {"LCB_CODEGEN_RELEASE_FILE": local_file}, clear=False):
                with patch("lcb_runner.benchmarks.code_generation.load_dataset") as mock_load_dataset:
                    mock_load_dataset.side_effect = AssertionError(
                        "HF should not be called when local file already has all requested ids"
                    )
                    data = load_code_generation_dataset_subset(["1001", "1002"])

            self.assertEqual({d.question_id for d in data}, {"1001", "1002"})

    def test_fallback_to_hf_only_for_local_miss(self):
        with tempfile.TemporaryDirectory() as td:
            local_file = os.path.join(td, "release_v1_full.jsonl")
            _write_jsonl(local_file, [_make_row("2001")])

            hf_rows = iter([_make_row("2002")])

            with patch.dict(os.environ, {"LCB_CODEGEN_RELEASE_FILE": local_file}, clear=False):
                with patch("lcb_runner.benchmarks.code_generation.load_dataset") as mock_load_dataset:
                    mock_load_dataset.return_value = hf_rows
                    data = load_code_generation_dataset_subset(["2001", "2002"])

            self.assertEqual({d.question_id for d in data}, {"2001", "2002"})
            mock_load_dataset.assert_called_once()

    def test_local_directory_can_span_multiple_jsonl_files(self):
        with tempfile.TemporaryDirectory() as td:
            _write_jsonl(os.path.join(td, "test.jsonl"), [_make_row("3001")])
            _write_jsonl(os.path.join(td, "test2.jsonl"), [_make_row("3002")])

            with patch.dict(os.environ, {"LCB_CODEGEN_RELEASE_FILE": td}, clear=False):
                with patch("lcb_runner.benchmarks.code_generation.load_dataset") as mock_load_dataset:
                    mock_load_dataset.side_effect = AssertionError(
                        "HF should not be called when local directory has all requested ids"
                    )
                    data = load_code_generation_dataset_subset(["3001", "3002"])

            self.assertEqual({d.question_id for d in data}, {"3001", "3002"})


if __name__ == "__main__":
    unittest.main()

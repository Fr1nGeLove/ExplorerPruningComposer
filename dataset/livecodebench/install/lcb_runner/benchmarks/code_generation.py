import json
import zlib
import pickle
import base64
import os
from enum import Enum
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = json.loads(self.input)
        #     self.output = json.loads(self.output)


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)

        self.public_test_cases = json.loads(self.public_test_cases)  # type: ignore
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = json.loads(self.private_test_cases)  # type: ignore
        except:
            self.private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)  # type: ignore

    def insert_output(self, output_list: list[str], code_list: list[str]) -> dict:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "output_list": output_list,
            "code_list": code_list,
        }

    def insert_output_evaluation(
        self,
        output_list: list[str],
        code_list: list[str],
        graded_list: list[bool],
        **kwargs,
    ) -> dict:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self):
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }


def load_code_generation_dataset(release_version="release_v1", start_date=None, end_date=None) -> list[CodeGenerationProblem]:
    local_dataset = _load_local_dataset_rows(release_version)
    if local_dataset is not None:
        dataset = [CodeGenerationProblem(**p) for p in local_dataset]  # type: ignore
    else:
        dataset = load_dataset("livecodebench/code_generation_lite", split="test", version_tag=release_version, trust_remote_code=True)
        dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [e for e in dataset if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [e for e in dataset if e.contest_date <= p_end_date]

    print(f"Loaded {len(dataset)} problems")
    return dataset


def _default_local_release_file(release_version: str) -> Path:
    install_dir = Path(__file__).resolve().parents[2]
    return install_dir / "cache" / f"code_generation_lite_{release_version}_full.jsonl"


def _expand_local_release_source(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.glob("*.jsonl") if p.is_file())
    return [path]


def _resolve_local_release_sources(release_version: str) -> list[Path]:
    env_file = os.environ.get("LCB_CODEGEN_RELEASE_FILE")
    if env_file:
        return _expand_local_release_source(Path(env_file))
    env_dir = os.environ.get("LCB_CODEGEN_CACHE_DIR")
    if env_dir:
        cache_dir = Path(env_dir)
        release_file = cache_dir / f"code_generation_lite_{release_version}_full.jsonl"
        if release_file.exists():
            return [release_file]
        return _expand_local_release_source(cache_dir)
    return [_default_local_release_file(release_version)]


def _resolve_local_release_file(release_version: str) -> Path:
    """Backward-compatible single-file resolver for older callers/tests."""
    return _resolve_local_release_sources(release_version)[0]


def _existing_local_release_sources(release_version: str) -> list[Path]:
    return [p for p in _resolve_local_release_sources(release_version) if p.is_file()]


def _iter_local_rows(local_file: Path):
    with local_file.open("r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: invalid JSON at {local_file}:{line_no}, skipping")


def _iter_local_source_rows(local_files: list[Path]):
    for local_file in local_files:
        yield from _iter_local_rows(local_file)


def _format_local_sources(local_files: list[Path]) -> str:
    if len(local_files) <= 3:
        return ", ".join(str(p) for p in local_files)
    preview = ", ".join(str(p) for p in local_files[:3])
    return f"{preview}, ... ({len(local_files)} files)"


def _load_local_dataset_rows(release_version: str) -> list[dict] | None:
    local_files = _existing_local_release_sources(release_version)
    if not local_files:
        return None
    print(f"Loading LiveCodeBench from local cache: {_format_local_sources(local_files)}")
    return list(_iter_local_source_rows(local_files))


def _filter_row_by_date(row: dict, p_start_date: datetime | None, p_end_date: datetime | None) -> bool:
    contest_date = datetime.fromisoformat(row["contest_date"])
    if p_start_date is not None and contest_date < p_start_date:
        return False
    if p_end_date is not None and contest_date > p_end_date:
        return False
    return True


def load_code_generation_dataset_subset(
    question_ids,
    release_version="release_v1",
    start_date=None,
    end_date=None,
) -> list[CodeGenerationProblem]:
    """
    Load only a subset of problems by question_id using streaming.

    This avoids downloading the full benchmark when custom_evaluator receives a
    small custom_output_file containing only a handful of question_ids.
    """
    target_ids = {str(qid) for qid in question_ids}
    if not target_ids:
        print("Loaded 0 problems (empty subset)")
        return []

    p_start_date = datetime.strptime(start_date, "%Y-%m-%d") if start_date is not None else None
    p_end_date = datetime.strptime(end_date, "%Y-%m-%d") if end_date is not None else None

    selected = {}
    local_files = _existing_local_release_sources(release_version)
    if local_files:
        print(f"Subset load: trying local cache first: {_format_local_sources(local_files)}")
        for row in _iter_local_source_rows(local_files):
            qid = str(row.get("question_id"))
            if qid not in target_ids or qid in selected:
                continue
            if not _filter_row_by_date(row, p_start_date, p_end_date):
                continue
            selected[qid] = CodeGenerationProblem(**row)  # type: ignore
            if len(selected) == len(target_ids):
                break

    missing_ids = target_ids - set(selected.keys())
    if missing_ids:
        if os.environ.get("LCB_SKIP_HF_FALLBACK", "0") == "1":
            print(
                f"Subset load: {len(missing_ids)} ids missing locally; "
                "skip HF fallback due to LCB_SKIP_HF_FALLBACK=1"
            )
        else:
            print(f"Subset load: {len(missing_ids)} ids missing locally, falling back to HF stream")
            dataset = load_dataset(
                "livecodebench/code_generation_lite",
                split="test",
                version_tag=release_version,
                trust_remote_code=True,
                streaming=True,
            )
            for row in dataset:
                qid = str(row["question_id"])
                if qid not in missing_ids or qid in selected:
                    continue
                if not _filter_row_by_date(row, p_start_date, p_end_date):
                    continue

                selected[qid] = CodeGenerationProblem(**row)  # type: ignore
                if len(selected) == len(target_ids):
                    break

    def _qid_sort_key(qid: str):
        return (0, int(qid)) if qid.isdigit() else (1, qid)

    missing = sorted(target_ids - set(selected.keys()), key=_qid_sort_key)
    if missing:
        print(f"Warning: {len(missing)} requested question_ids not found in {release_version}: {missing[:10]}")

    loaded = [selected[qid] for qid in sorted(selected.keys(), key=_qid_sort_key)]
    print(f"Loaded {len(loaded)} problems (subset mode)")
    return loaded


def load_code_generation_dataset_not_fast(release_version="release_v1") -> list[CodeGenerationProblem]:
    dataset = load_dataset("livecodebench/code_generation", split="test")
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    print(f"Loaded {len(dataset)} problems")
    return dataset


if __name__ == "__main__":
    dataset = load_code_generation_dataset()

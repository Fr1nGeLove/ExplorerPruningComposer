"""
GenDR (Generate-Diff-Revert) post-processing for precise debugging.

This module adds a test-guided pruning stage after LLM generation:
    1) take model fix C_fix
    2) split predicted edits into blocks
    3) try reverting blocks and keep only test-essential edits

It is model-agnostic and reuses the dataset handlers' existing unit-test
execution paths.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from dataset import get_handler
from utils import apply_diff, expand_blocks_to_diff, file_diff, parse_diff_to_blocks


DiffDict = Dict[str, Dict[str, str]]
Block = Dict[str, object]


@dataclass
class GenDRConfig:
    strategy: str = "sequential"  # sequential | independent | hierarchical
    block_granularity: str = "hunk"  # hunk | line
    max_blocks: int = 80  # <=0 means no cap
    timeout_per_task: int = 20
    timeout: int = 1800
    only_when_fix_passes: bool = True


class UnitTestOracle:
    """
    Thin wrapper over dataset handlers with caching + batch checking.
    """

    def __init__(self, dataset_name: str, log_dir: str, timeout_per_task: int = 20, timeout: int = 1800):
        self.dataset_name = dataset_name
        self.handler = get_handler(dataset_name)
        self.log_dir = log_dir
        self.timeout_per_task = timeout_per_task
        self.timeout = timeout
        self._cache: Dict[Tuple[str, str], bool] = {}
        self._feedback_cache: Dict[Tuple[str, str], str] = {}
        self._run_id = 0
        self.oracle_calls = 0
        os.makedirs(self.log_dir, exist_ok=True)

    def _next_case_id(self, task_id: str) -> str:
        case_id = f"{task_id}__gendr_case_{self._run_id}"
        self._run_id += 1
        return case_id

    def check_many_with_feedback(self, task_id: str, solutions: Sequence[str]) -> List[dict]:
        if not solutions:
            return []

        results: List[dict] = [{"passed": False, "feedback": ""} for _ in solutions]
        pending_by_solution: Dict[str, List[int]] = {}
        for i, sol in enumerate(solutions):
            if not str(sol or "").strip():
                results[i] = {"passed": False, "feedback": "empty solution"}
                self._cache[(task_id, sol)] = False
                self._feedback_cache[(task_id, sol)] = "empty solution"
                continue
            key = (task_id, sol)
            if key in self._cache:
                results[i] = {
                    "passed": self._cache[key],
                    "feedback": self._feedback_cache.get(key, ""),
                }
            else:
                pending_by_solution.setdefault(sol, []).append(i)

        if not pending_by_solution:
            return results

        case_ids: List[str] = []
        payload = []
        pending_solutions = list(pending_by_solution.keys())
        for sol in pending_solutions:
            case_id = self._next_case_id(task_id)
            case_ids.append(case_id)
            payload.append({"task_id": case_id, "solution": sol})

        pass_map = {case_id: False for case_id in case_ids}
        feedback_map = {case_id: "" for case_id in case_ids}
        try:
            prefix = os.path.join(self.log_dir, f"oracle_{self._run_id}")
            verify_file = self.handler.build_verify_unit_test(prefix, payload, sol_field="solution")
            gt_file = self.handler.save_formatted_gt(prefix + "_gt", [{"task_id": cid} for cid in case_ids])
            fail_ids, correct_ids, fail_feedback = self.handler.verify_unit_test(
                verify_file,
                gt_file=gt_file,
                timeout_per_task=self.timeout_per_task,
                timeout=self.timeout,
            )
            correct_set = set(correct_ids)
            feedback_by_fail_id = dict(zip(fail_ids, fail_feedback))
            for cid in case_ids:
                pass_map[cid] = cid in correct_set
                feedback_map[cid] = feedback_by_fail_id.get(cid, "")
        except Exception as e:
            # Fail-safe: if oracle fails, mark candidates as non-pass so pruning
            # becomes conservative (keep edits) instead of destructive.
            print(f"[GenDR] Oracle failure on task_id={task_id}: {e}")

        self.oracle_calls += len(case_ids)
        for case_id, sol in zip(case_ids, pending_solutions):
            passed = pass_map[case_id]
            feedback = feedback_map[case_id]
            self._cache[(task_id, sol)] = passed
            self._feedback_cache[(task_id, sol)] = feedback
            for idx in pending_by_solution[sol]:
                results[idx] = {"passed": passed, "feedback": feedback}

        return results

    def check_many(self, task_id: str, solutions: Sequence[str]) -> List[bool]:
        return [bool(item["passed"]) for item in self.check_many_with_feedback(task_id, solutions)]

    def check_one(self, task_id: str, solution: str) -> bool:
        return self.check_many(task_id, [solution])[0]


def _diff_to_line_blocks(pred_diff: DiffDict) -> List[Block]:
    ordered_items = sorted(pred_diff.items(), key=lambda x: (int(x[0]), x[1]["type"]))
    blocks: List[Block] = []
    for i, (line_no, edit) in enumerate(ordered_items):
        blocks.append(
            {
                "block_start": int(line_no),
                "block_end": int(line_no),
                "diff": {line_no: copy.deepcopy(edit)},
                "block_id": i,
            }
        )
    return blocks


def build_blocks(pred_diff: DiffDict, granularity: str = "hunk") -> List[Block]:
    if not pred_diff:
        return []
    if granularity == "hunk":
        return parse_diff_to_blocks(pred_diff)
    if granularity == "line":
        return _diff_to_line_blocks(pred_diff)
    raise ValueError(f"Unsupported block granularity: {granularity}")


def _rebuild_from_mask(buggy_code: str, blocks: Sequence[Block], keep_mask: Sequence[bool]) -> Tuple[str, DiffDict]:
    kept_blocks = [copy.deepcopy(block) for block, keep in zip(blocks, keep_mask) if keep]
    merged_diff = expand_blocks_to_diff(kept_blocks, ordered=False) if kept_blocks else {}
    rebuilt = apply_diff(buggy_code, merged_diff, with_delta=True)
    return rebuilt, merged_diff


def _sequential_keep_mask(task_id: str, buggy_code: str, blocks: Sequence[Block], oracle: UnitTestOracle) -> List[bool]:
    keep_mask = [True] * len(blocks)
    for i in range(len(blocks)):
        if not keep_mask[i]:
            continue
        candidate_mask = keep_mask[:]
        candidate_mask[i] = False
        candidate_solution, _ = _rebuild_from_mask(buggy_code, blocks, candidate_mask)
        if oracle.check_one(task_id, candidate_solution):
            keep_mask[i] = False
    return keep_mask


def _independent_keep_mask(task_id: str, buggy_code: str, blocks: Sequence[Block], oracle: UnitTestOracle) -> List[bool]:
    candidates = []
    for i in range(len(blocks)):
        candidate_mask = [True] * len(blocks)
        candidate_mask[i] = False
        candidate_solution, _ = _rebuild_from_mask(buggy_code, blocks, candidate_mask)
        candidates.append(candidate_solution)
    removable = oracle.check_many(task_id, candidates)
    return [not can_remove for can_remove in removable]


def _hierarchical_keep_mask(task_id: str, buggy_code: str, blocks: Sequence[Block], oracle: UnitTestOracle) -> List[bool]:
    # Pass 1: independent check to identify definitely-essential blocks.
    candidates = []
    for i in range(len(blocks)):
        candidate_mask = [True] * len(blocks)
        candidate_mask[i] = False
        candidate_solution, _ = _rebuild_from_mask(buggy_code, blocks, candidate_mask)
        candidates.append(candidate_solution)
    removable = oracle.check_many(task_id, candidates)

    # Pass 2: sequential pruning only over potentially removable blocks.
    keep_mask = [True] * len(blocks)
    maybe_removable = [i for i, can_remove in enumerate(removable) if can_remove]
    for i in maybe_removable:
        candidate_mask = keep_mask[:]
        candidate_mask[i] = False
        candidate_solution, _ = _rebuild_from_mask(buggy_code, blocks, candidate_mask)
        if oracle.check_one(task_id, candidate_solution):
            keep_mask[i] = False
    return keep_mask


def _compute_keep_mask(
    strategy: str,
    task_id: str,
    buggy_code: str,
    blocks: Sequence[Block],
    oracle: UnitTestOracle,
) -> List[bool]:
    if strategy == "sequential":
        return _sequential_keep_mask(task_id, buggy_code, blocks, oracle)
    if strategy == "independent":
        return _independent_keep_mask(task_id, buggy_code, blocks, oracle)
    if strategy == "hierarchical":
        return _hierarchical_keep_mask(task_id, buggy_code, blocks, oracle)
    raise ValueError(f"Unsupported GenDR strategy: {strategy}")


def apply_gendr_to_item(item: dict, oracle: UnitTestOracle, config: GenDRConfig, target_round: int | None = None) -> dict:
    """
    Apply GenDR to one result row. Returns a deep-copied row.
    """
    out = copy.deepcopy(item)
    debug_results = out.get("debug_results")
    if not isinstance(debug_results, dict):
        return out
    if target_round is not None and out.get("round") != target_round:
        return out

    task_id = str(out.get("task_id", ""))
    buggy_code = out.get("buggy_code", "")
    solution = debug_results.get("solution", "") or ""
    pred_diff = debug_results.get("pred_diff")
    if pred_diff is None:
        _, _, pred_diff = file_diff(buggy_code, solution, cleaned=True)

    blocks = build_blocks(pred_diff, granularity=config.block_granularity)
    oracle_before = oracle.oracle_calls
    meta = {
        "enabled": True,
        "strategy": config.strategy,
        "block_granularity": config.block_granularity,
        "n_blocks": len(blocks),
        "n_removed_blocks": 0,
        "n_removed_lines": 0,
        "skip_reason": None,
        "base_passed": None,
        "final_passed": None,
        "fallback_used": False,
        "oracle_calls": 0,
    }

    if not blocks:
        meta["skip_reason"] = "no_predicted_edits"
        meta["oracle_calls"] = oracle.oracle_calls - oracle_before
        debug_results["gendr"] = meta
        return out

    if config.max_blocks > 0 and len(blocks) > config.max_blocks:
        meta["skip_reason"] = "too_many_blocks"
        meta["oracle_calls"] = oracle.oracle_calls - oracle_before
        debug_results["gendr"] = meta
        return out

    if config.only_when_fix_passes:
        base_passed = oracle.check_one(task_id, solution)
        meta["base_passed"] = base_passed
        if not base_passed:
            meta["skip_reason"] = "base_fix_failed_tests"
            meta["oracle_calls"] = oracle.oracle_calls - oracle_before
            debug_results["gendr"] = meta
            return out
    else:
        meta["base_passed"] = None

    keep_mask = _compute_keep_mask(config.strategy, task_id, buggy_code, blocks, oracle)
    pruned_solution, pruned_diff = _rebuild_from_mask(buggy_code, blocks, keep_mask)
    final_passed = oracle.check_one(task_id, pruned_solution)
    meta["final_passed"] = final_passed

    if not final_passed:
        meta["fallback_used"] = True
        pruned_solution = solution
        pruned_diff = pred_diff
        keep_mask = [True] * len(blocks)

    removed_indices = [i for i, keep in enumerate(keep_mask) if not keep]
    meta["n_removed_blocks"] = len(removed_indices)
    meta["n_removed_lines"] = sum(len(blocks[i]["diff"]) for i in removed_indices)
    meta["oracle_calls"] = oracle.oracle_calls - oracle_before

    debug_results["solution"] = pruned_solution
    debug_results["pred_diff"] = pruned_diff
    debug_results["gendr"] = meta
    return out


def apply_gendr_to_results(
    results: Sequence[dict],
    dataset_name: str,
    log_dir: str,
    config: GenDRConfig,
    target_round: int | None = None,
) -> Tuple[List[dict], dict]:
    """
    Apply GenDR to a list of debugging results and return (new_results, summary).
    """
    oracle = UnitTestOracle(
        dataset_name=dataset_name,
        log_dir=log_dir,
        timeout_per_task=config.timeout_per_task,
        timeout=config.timeout,
    )

    processed = []
    summary = {
        "processed_items": 0,
        "with_debug_results": 0,
        "skipped_items": 0,
        "total_removed_blocks": 0,
        "total_removed_lines": 0,
        "oracle_calls": 0,
    }
    for item in results:
        out = apply_gendr_to_item(item, oracle=oracle, config=config, target_round=target_round)
        processed.append(out)
        if "debug_results" not in out:
            continue
        summary["with_debug_results"] += 1
        gmeta = out["debug_results"].get("gendr")
        if not isinstance(gmeta, dict):
            continue
        summary["processed_items"] += 1
        if gmeta.get("skip_reason"):
            summary["skipped_items"] += 1
        summary["total_removed_blocks"] += int(gmeta.get("n_removed_blocks", 0))
        summary["total_removed_lines"] += int(gmeta.get("n_removed_lines", 0))

    summary["oracle_calls"] = oracle.oracle_calls
    return processed, summary

#!/usr/bin/env python3
"""Build and optionally evaluate EPC hybrid-gated-composer outputs.

Selection rule per item:
  1. Use the smallest passing pruned explorer candidate, if one exists.
  2. Otherwise use the EPC composer output when EPC final_pass is true.
  3. Otherwise use the best failed explorer candidate as a partial fallback.
  4. Only if no usable partial candidate exists, fall back to the original buggy code.

This keeps the strong precision behavior of the best-explorer ablation while
letting the composer or partial candidates rescue cases where no individual
explorer passes.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config import DEFAULT_TOLERANCE_MULTILINE, DEFAULT_TOLERANCE_SINGLELINE
from epc_decision_only import _choose_best_partial_candidate
from evaluator import Evaluator
from utils import file_diff


def _passing_pruned_candidates(item: dict) -> list[dict]:
    candidates = item.get("epc", {}).get("pruned_candidates") or []
    passing = []
    for cand in candidates:
        meta = cand.get("gendr_meta") or {}
        code = cand.get("pruned_repair_code")
        if code is not None and meta.get("final_passed") is True:
            passing.append(cand)
    return sorted(
        passing,
        key=lambda c: (
            int(c.get("pruned_edit_size", 10**9)),
            str(c.get("explorer_id", "")),
            int(c.get("retry_round", 0)),
        ),
    )


def build_hybrid_results(results: list[dict]) -> tuple[list[dict], dict]:
    out = []
    counts: Counter[str] = Counter()
    rescue_task_ids = []
    partial_task_ids = []
    fallback_task_ids = []

    for item in results:
        row = copy.deepcopy(item)
        epc = row.get("epc") or {}
        original_debug = row.get("debug_results") or {}
        passing = _passing_pruned_candidates(row)
        chosen = passing[0] if passing else None
        partial = _choose_best_partial_candidate(epc.get("explorer_candidates") or [])

        if chosen is not None:
            source = "best_passing_pruned_explorer"
            final_code = chosen["pruned_repair_code"]
        elif epc.get("final_pass") is True and original_debug.get("solution") is not None:
            source = "composer_rescue"
            final_code = original_debug["solution"]
            rescue_task_ids.append(row.get("task_id"))
        elif partial is not None:
            source = "best_partial_failed_explorer"
            final_code = partial["raw_repair_code"]
            partial_task_ids.append(row.get("task_id"))
        else:
            source = "buggy_fallback"
            final_code = row.get("buggy_code", "")
            fallback_task_ids.append(row.get("task_id"))

        _, _, pred_diff = file_diff(row.get("buggy_code", ""), final_code, cleaned=True)
        row["debug_results"] = {
            "model": "epc_hybrid_gated_composer",
            "solution": final_code,
            "pred_diff": pred_diff,
        }
        row["epc_hybrid"] = {
            "source": source,
            "no_composer_status": (
                "best_passing_pruned_explorer" if chosen is not None else "no_passing_pruned_explorer"
            ),
            "composer_status": epc.get("status"),
            "composer_final_pass": epc.get("final_pass"),
            "chosen_best_explorer": _compact_candidate(chosen) if chosen is not None else None,
        }
        counts[source] += 1
        out.append(row)

    summary = {
        "n": len(out),
        "selected_best_explorer": counts["best_passing_pruned_explorer"],
        "composer_rescue": counts["composer_rescue"],
        "partial_fallback": counts["best_partial_failed_explorer"],
        "buggy_fallback": counts["buggy_fallback"],
        "rescue_task_ids": rescue_task_ids,
        "partial_task_ids": partial_task_ids,
        "fallback_task_ids": fallback_task_ids,
    }
    return out, summary


def _compact_candidate(candidate: dict | None) -> dict | None:
    if candidate is None:
        return None
    keys = [
        "explorer_id",
        "base_explorer_id",
        "explorer_role",
        "retry_round",
        "raw_edit_size",
        "pruned_edit_size",
        "gendr_meta",
    ]
    return {k: copy.deepcopy(candidate.get(k)) for k in keys if k in candidate}


def _mean_scores(scores: dict) -> dict:
    unit = scores.get("Unit score", {}) or {}
    sym = scores.get("Symbolic block scores", {}) or {}
    n = len(unit)
    if n == 0:
        return {"n": 0, "unit": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "n": n,
        "unit": sum(unit.values()) / n,
        "precision": sum(v["precision"] for v in sym.values()) / n,
        "recall": sum(v["recall"] for v in sym.values()) / n,
        "f1": sum(v["f1"] for v in sym.values()) / n,
    }


def evaluate_results(args: argparse.Namespace, results: list[dict]) -> dict:
    eval_args = argparse.Namespace(
        dataset_name=args.dataset_name,
        eval_result_dir=args.eval_result_dir,
        eval_model_name=args.eval_model_name,
        eval_set_name=args.eval_set_name,
        stride=args.stride,
        tolerance=args.tolerance,
        unit_test_timeout=args.unit_test_timeout,
    )
    evaluator = Evaluator(eval_args)

    original_verify = evaluator.handler.verify_unit_test

    def verify_with_timeout_per_task(*v_args, **v_kwargs):
        v_kwargs.setdefault("timeout_per_task", args.timeout_per_task)
        return original_verify(*v_args, **v_kwargs)

    evaluator.handler.verify_unit_test = verify_with_timeout_per_task
    return evaluator.run_evaluation(results=results, round=args.round)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--summary_file", default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval_result_dir", default="results")
    parser.add_argument("--eval_model_name", default="deepseek-chat_epc_hybrid_gated_composer_t60")
    parser.add_argument("--eval_set_name", required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--tolerance", type=int, default=None)
    parser.add_argument("--unit_test_timeout", type=int, default=1800)
    parser.add_argument("--timeout_per_task", type=int, default=60)
    args = parser.parse_args()

    if args.tolerance is None:
        args.tolerance = (
            DEFAULT_TOLERANCE_MULTILINE if args.mode == "multi" else DEFAULT_TOLERANCE_SINGLELINE
        )
    if args.summary_file is None:
        args.summary_file = os.path.splitext(args.output_file)[0] + ".summary.json"

    with open(args.input_file, "r") as f:
        results = json.load(f)

    hybrid, summary = build_hybrid_results(results)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(hybrid, f, indent=2)
    with open(args.summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[hybrid] wrote {args.output_file}")
    print(f"[hybrid] summary {json.dumps(summary, ensure_ascii=False)}")

    if args.evaluate:
        scores = evaluate_results(args, hybrid)
        means = _mean_scores(scores)
        print(
            "[hybrid][eval] "
            f"n={means['n']} unit={means['unit']:.4f} "
            f"precision={means['precision']:.4f} recall={means['recall']:.4f} "
            f"f1={means['f1']:.4f}"
        )


if __name__ == "__main__":
    main()

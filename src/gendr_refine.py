"""
Offline GenDR refiner for existing debug-results JSON files.

Typical usage:
    python src/gendr_refine.py \
      --dataset_name bigcodebench \
      --input_file results/bigcodebench/debug_results/deepseek-chat_on_bigcodebench_pdb_single_hard_round_1.json \
      --evaluate --compare_input
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import DEFAULT_TOLERANCE_MULTILINE, DEFAULT_TOLERANCE_SINGLELINE
from evaluator import Evaluator
from gendr import GenDRConfig, apply_gendr_to_results


def _derive_model_name(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    if "_on_" in stem:
        return stem.split("_on_", 1)[0]
    return stem


def _derive_eval_set_name(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    if "_on_" in stem and "_round_" in stem:
        return stem.split("_on_", 1)[1].rsplit("_round_", 1)[0]
    return stem


def _mean_unit_and_symbolic(scores: dict) -> tuple[float, float, float, float, int]:
    unit = scores.get("Unit score", {})
    sym = scores.get("Symbolic block scores", {})
    n = len(unit)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    unit_avg = sum(unit.values()) / n
    prec_avg = sum(v["precision"] for v in sym.values()) / n
    rec_avg = sum(v["recall"] for v in sym.values()) / n
    f1_avg = sum(v["f1"] for v in sym.values()) / n
    return unit_avg, prec_avg, rec_avg, f1_avg, n


def _evaluate_results(
    dataset_name: str,
    results: list,
    eval_result_dir: str,
    eval_model_name: str,
    eval_set_name: str,
    stride: int,
    tolerance: int,
    unit_test_timeout: int,
    timeout_per_task: int,
) -> dict:
    eval_args = argparse.Namespace(
        dataset_name=dataset_name,
        eval_result_dir=eval_result_dir,
        eval_model_name=eval_model_name,
        eval_set_name=eval_set_name,
        stride=stride,
        tolerance=tolerance,
        unit_test_timeout=unit_test_timeout,
    )
    evaluator = Evaluator(eval_args)

    original_verify = evaluator.handler.verify_unit_test

    def verify_with_timeout_per_task(*v_args, **v_kwargs):
        v_kwargs.setdefault("timeout_per_task", timeout_per_task)
        return original_verify(*v_args, **v_kwargs)

    evaluator.handler.verify_unit_test = verify_with_timeout_per_task
    return evaluator.run_evaluation(results=results, round=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True,
                        help="Existing debug_results JSON file.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSON path. Default: <input>_gendr_<strategy>_<granularity>.json")
    parser.add_argument("--target_round", type=int, default=None,
                        help="Only apply GenDR to items whose item['round'] equals this value.")

    parser.add_argument("--gendr_strategy", choices=["sequential", "independent", "hierarchical"],
                        default="sequential")
    parser.add_argument("--gendr_block_granularity", choices=["hunk", "line"], default="hunk")
    parser.add_argument("--gendr_max_blocks", type=int, default=80)
    parser.add_argument("--gendr_timeout_per_task", type=int, default=20)
    parser.add_argument("--gendr_timeout", type=int, default=1800)
    parser.add_argument("--gendr_allow_non_passing_base", action="store_true")

    parser.add_argument("--evaluate", action="store_true", help="Evaluate refined output with evaluator.py logic.")
    parser.add_argument("--compare_input", action="store_true",
                        help="When --evaluate is set, also evaluate the original input file for comparison.")
    parser.add_argument("--eval_result_dir", type=str, default="results")
    parser.add_argument("--eval_model_name", type=str, default=None)
    parser.add_argument("--eval_set_name", type=str, default=None)
    parser.add_argument("--unit_test_timeout", type=int, default=1800,
                        help="Timeout (seconds) for dataset unit-test evaluator subprocess.")
    parser.add_argument("--timeout_per_task", type=int, default=20,
                        help="Per-task timeout (seconds) for dataset unit-test evaluator calls.")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--tolerance", type=int, default=None)

    args = parser.parse_args()

    if args.tolerance is None:
        args.tolerance = (DEFAULT_TOLERANCE_MULTILINE if args.mode == "multi" else DEFAULT_TOLERANCE_SINGLELINE)

    with open(args.input_file, "r") as f:
        input_results = json.load(f)

    if args.output_file is None:
        stem, ext = os.path.splitext(args.input_file)
        args.output_file = f"{stem}_gendr_{args.gendr_strategy}_{args.gendr_block_granularity}{ext}"

    eval_model_base = args.eval_model_name or _derive_model_name(args.input_file)
    eval_set_name = args.eval_set_name or _derive_eval_set_name(args.input_file)

    output_dir = os.path.join("results", args.dataset_name, "gendr_log")
    gendr_log_dir = os.path.join(
        output_dir,
        f"{os.path.splitext(os.path.basename(args.output_file))[0]}_oracle",
    )
    config = GenDRConfig(
        strategy=args.gendr_strategy,
        block_granularity=args.gendr_block_granularity,
        max_blocks=args.gendr_max_blocks,
        timeout_per_task=args.gendr_timeout_per_task,
        timeout=args.gendr_timeout,
        only_when_fix_passes=not args.gendr_allow_non_passing_base,
    )

    refined_results, summary = apply_gendr_to_results(
        results=input_results,
        dataset_name=args.dataset_name,
        log_dir=gendr_log_dir,
        config=config,
        target_round=args.target_round,
    )

    with open(args.output_file, "w") as f:
        json.dump(refined_results, f, indent=2)

    print(f"[GenDR] Wrote refined results to: {args.output_file}")
    print(
        "[GenDR] Summary: "
        f"processed={summary['processed_items']} "
        f"with_debug_results={summary['with_debug_results']} "
        f"skipped={summary['skipped_items']} "
        f"removed_blocks={summary['total_removed_blocks']} "
        f"removed_lines={summary['total_removed_lines']} "
        f"oracle_calls={summary['oracle_calls']}"
    )

    if args.evaluate:
        if args.compare_input:
            baseline_name = f"{eval_model_base}_baseline"
            baseline_scores = _evaluate_results(
                dataset_name=args.dataset_name,
                results=input_results,
                eval_result_dir=args.eval_result_dir,
                eval_model_name=baseline_name,
                eval_set_name=eval_set_name,
                stride=args.stride,
                tolerance=args.tolerance,
                unit_test_timeout=args.unit_test_timeout,
                timeout_per_task=args.timeout_per_task,
            )
            b_unit, b_prec, b_rec, b_f1, b_n = _mean_unit_and_symbolic(baseline_scores)
            print(
                f"[GenDR][baseline] unit={b_unit:.3f} prec={b_prec:.3f} "
                f"rec={b_rec:.3f} f1={b_f1:.3f} (n={b_n})"
            )
        refined_name = f"{eval_model_base}_gendr_{args.gendr_strategy}_{args.gendr_block_granularity}"
        refined_scores = _evaluate_results(
            dataset_name=args.dataset_name,
            results=refined_results,
            eval_result_dir=args.eval_result_dir,
            eval_model_name=refined_name,
            eval_set_name=eval_set_name,
            stride=args.stride,
            tolerance=args.tolerance,
            unit_test_timeout=args.unit_test_timeout,
            timeout_per_task=args.timeout_per_task,
        )
        r_unit, r_prec, r_rec, r_f1, r_n = _mean_unit_and_symbolic(refined_scores)
        print(
            f"[GenDR][refined]  unit={r_unit:.3f} prec={r_prec:.3f} "
            f"rec={r_rec:.3f} f1={r_f1:.3f} (n={r_n})"
        )


if __name__ == "__main__":
    main()

"""
Version-A EPC (Explorer-Pruner-Composer) decision-only pipeline.

Implements:
  1) role-specific explorer generation
  2) passing-only filtering
  3) GenDR pruning per passing candidate
  4) atomic PatchBank + patch clustering
  5) decision-only composer (JSON plan only, no full-code output)
  6) deterministic patch application backend
  7) optional PDB evaluation via Evaluator

This module is intentionally standalone so it does not disrupt the existing
bug_correct.py workflow.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import dspy
import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api_config import resolve_api_key
from config import DEFAULT_TOLERANCE_MULTILINE, DEFAULT_TOLERANCE_SINGLELINE
from evaluator import Evaluator
from gendr import GenDRConfig, UnitTestOracle, apply_gendr_to_item, build_blocks
from module import Debugger
from utils import apply_diff, file_diff


JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ExplorerRole:
    explorer_id: str
    role_name: str
    prompt_bias: str
    debug_mode: str


DEFAULT_EXPLORER_ROLES: List[ExplorerRole] = [
    ExplorerRole(
        explorer_id="E1",
        role_name="Minimal Explorer",
        prompt_bias=(
            "Prioritize the smallest possible behavior-changing fix. Avoid broad rewrites."
        ),
        debug_mode="minimal",
    ),
    ExplorerRole(
        explorer_id="E2",
        role_name="Boundary Explorer",
        prompt_bias=(
            "Prioritize edge cases: empty input, boundaries, off-by-one, inclusive/exclusive conditions."
        ),
        debug_mode="minimal",
    ),
    ExplorerRole(
        explorer_id="E3",
        role_name="Data-flow Explorer",
        prompt_bias=(
            "Focus on assignments, state transitions, return values, and value propagation."
        ),
        debug_mode="minimal",
    ),
    ExplorerRole(
        explorer_id="E4",
        role_name="Control-flow Explorer",
        prompt_bias=(
            "Focus on branches, loop conditions, and early-return logic."
        ),
        debug_mode="minimal",
    ),
    ExplorerRole(
        explorer_id="E5",
        role_name="Freeform Explorer",
        prompt_bias=(
            "Fix the bug in any way that passes tests; minimality is not required."
        ),
        debug_mode="free",
    ),
]


def _base_task_id(task_id: str | None) -> str:
    if task_id is None:
        return ""
    return str(task_id).split("_", 1)[0]


def _select_items_diverse(data: List[dict], max_items: int) -> List[dict]:
    if max_items <= 0 or len(data) <= max_items:
        return data
    groups: Dict[str, List[dict]] = {}
    for item in data:
        groups.setdefault(_base_task_id(item.get("task_id")), []).append(item)

    selected: List[dict] = []
    layer = 0
    while len(selected) < max_items:
        added = False
        for _, items in groups.items():
            if layer < len(items):
                selected.append(items[layer])
                added = True
                if len(selected) >= max_items:
                    break
        if not added:
            break
        layer += 1
    return selected


def _call_lm_text(lm: Any, prompt: str) -> str:
    response = lm(prompt)
    if isinstance(response, list) and response:
        return str(response[0])
    if hasattr(response, "completions") and response.completions:
        completion = response.completions[0]
        if hasattr(completion, "text"):
            return str(completion.text)
        if hasattr(completion, "content"):
            return str(completion.content)
    return str(response)


def _safe_json_loads(raw: str) -> tuple[dict | None, str]:
    text = (raw or "").strip()
    if not text:
        return None, "empty composer output"

    candidates = [text]
    fence = JSON_FENCE_RE.search(text)
    if fence:
        candidates.insert(0, fence.group(1).strip())

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj, ""
        except json.JSONDecodeError:
            pass

    balanced = _extract_balanced_json_object(text)
    if balanced is not None:
        try:
            obj = json.loads(balanced)
            if isinstance(obj, dict):
                return obj, ""
        except json.JSONDecodeError:
            pass

    return None, "cannot parse composer JSON"


def _extract_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    in_string = False
    escaped = False
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return None


def _diff_entry_key(line_no: str, edit: dict) -> tuple:
    return (
        str(line_no),
        str(edit.get("type", "")),
        str(edit.get("original", "")),
        str(edit.get("modified", "")),
    )


def _patch_signature(patch: dict) -> tuple:
    entries = sorted(
        (_diff_entry_key(ln, edit) for ln, edit in patch["diff"].items()),
        key=lambda x: (int(x[0]), x[1]),
    )
    return tuple(entries)


def _dedup_patch_bank(patch_bank: List[dict]) -> List[dict]:
    seen = set()
    deduped = []
    for patch in patch_bank:
        sig = _patch_signature(patch)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(patch)
    return deduped


def _infer_edit_type(diff_block: dict) -> str:
    kinds = sorted({e.get("type", "Unknown") for e in diff_block.values()})
    if len(kinds) == 1:
        return kinds[0].lower()
    return "mixed_change"


def _infer_risk(edit_size: int) -> str:
    if edit_size <= 1:
        return "low"
    if edit_size <= 3:
        return "medium"
    return "high"


def _build_unified_diff(block: dict) -> str:
    entries = sorted(block["diff"].items(), key=lambda x: int(x[0]))
    old_count = sum(1 for _, e in entries if e["type"] in ("Modify", "Delete"))
    new_count = sum(1 for _, e in entries if e["type"] in ("Modify", "Add"))
    old_count = max(old_count, 1)
    new_count = max(new_count, 1)
    start = block["block_start"]
    lines = [f"@@ -{start},{old_count} +{start},{new_count} @@"]
    for _, e in entries:
        tp = e["type"]
        if tp == "Modify":
            lines.append(f"-{e['original']}")
            lines.append(f"+{e['modified']}")
        elif tp == "Delete":
            lines.append(f"-{e['original']}")
        elif tp == "Add":
            lines.append(f"+{e['modified']}")
    return "\n".join(lines)


def _build_ast_line_index(code: str) -> Dict[int, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}

    line_to_best: Dict[int, tuple[int, str]] = {}

    def visit(node: ast.AST, depth: int = 0) -> None:
        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", None)
        if isinstance(lineno, int):
            if end_lineno is None:
                end_lineno = lineno
            label = type(node).__name__
            for ln in range(lineno, int(end_lineno) + 1):
                prev = line_to_best.get(ln)
                if prev is None or depth >= prev[0]:
                    line_to_best[ln] = (depth, label)
        for child in ast.iter_child_nodes(node):
            visit(child, depth + 1)

    visit(tree, 0)
    return {ln: label for ln, (_, label) in line_to_best.items()}


def _infer_ast_locus(ast_index: Dict[int, str], span_start: int, span_end: int) -> str:
    labels = [ast_index[ln] for ln in range(span_start, span_end + 1) if ln in ast_index]
    if not labels:
        return f"line_span:{span_start}-{span_end}"
    top = Counter(labels).most_common(1)[0][0]
    return f"{top}@{span_start}-{span_end}"


def _build_atomic_patches(
    buggy_code: str,
    pruned_diff: Dict[str, dict],
    source_explorer: str,
    source_role: str,
    patch_granularity: str,
    patch_start_idx: int = 1,
) -> List[dict]:
    if not pruned_diff:
        return []

    blocks = build_blocks(pruned_diff, granularity=patch_granularity)
    buggy_lines = buggy_code.splitlines()
    ast_index = _build_ast_line_index(buggy_code)
    base_hash = hashlib.sha256(buggy_code.encode("utf-8")).hexdigest()
    patches: List[dict] = []

    for i, block in enumerate(blocks):
        patch_id = f"P{patch_start_idx + i}"
        start = int(block["block_start"])
        end = int(block["block_end"])
        entries = sorted(block["diff"].items(), key=lambda x: int(x[0]))
        old_text_lines = [e["original"] for _, e in entries if e["type"] in ("Modify", "Delete")]
        new_text_lines = [e["modified"] for _, e in entries if e["type"] in ("Modify", "Add")]
        context_before = buggy_lines[start - 2] if start - 2 >= 0 else ""
        context_after = buggy_lines[end] if end < len(buggy_lines) else ""
        patch = {
            "patch_id": patch_id,
            "source_explorer": source_explorer,
            "source_role": source_role,
            "base_code_hash": base_hash,
            "old_span": [start, end],
            "old_text": "\n".join(old_text_lines),
            "new_text": "\n".join(new_text_lines),
            "context_before": context_before,
            "context_after": context_after,
            "edit_type": _infer_edit_type(block["diff"]),
            "ast_locus": _infer_ast_locus(ast_index, start, end),
            "candidate_local_evidence": [
                "Source explorer candidate passed all tests.",
                "Candidate-local GenDR pruning retained this edit in the source candidate.",
            ],
            "edit_size": len(block["diff"]),
            "risk": _infer_risk(len(block["diff"])),
            "unified_diff": _build_unified_diff(block),
            "diff": copy.deepcopy(block["diff"]),
        }
        patches.append(patch)
    return patches


def _span_overlap(span_a: Sequence[int], span_b: Sequence[int]) -> bool:
    a1, a2 = int(span_a[0]), int(span_a[1])
    b1, b2 = int(span_b[0]), int(span_b[1])
    return not (a2 < b1 or b2 < a1)


def _cluster_patches(patch_bank: List[dict]) -> Tuple[List[dict], Dict[str, str]]:
    if not patch_bank:
        return [], {}

    n = len(patch_bank)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = patch_bank[i], patch_bank[j]
            same_ast = pi["ast_locus"] == pj["ast_locus"]
            overlap = _span_overlap(pi["old_span"], pj["old_span"])
            if same_ast or overlap:
                union(i, j)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    clusters = []
    patch_to_cluster: Dict[str, str] = {}
    cluster_idx = 1
    for _, idxs in sorted(groups.items(), key=lambda kv: min(kv[1])):
        cid = f"C{cluster_idx}"
        cluster_idx += 1
        patch_ids = [patch_bank[i]["patch_id"] for i in idxs]
        loci = [patch_bank[i]["ast_locus"] for i in idxs]
        top_locus = Counter(loci).most_common(1)[0][0]
        spans = [patch_bank[i]["old_span"] for i in idxs]
        span_start = min(s[0] for s in spans)
        span_end = max(s[1] for s in spans)
        cluster = {
            "cluster_id": cid,
            "semantic_locus": top_locus,
            "patch_ids": patch_ids,
            "cluster_note": (
                "Alternative or related edits for the same semantic/textual region. "
                "Prefer selecting at most one unless clearly complementary."
            ),
            "span": [span_start, span_end],
        }
        clusters.append(cluster)
        for pid in patch_ids:
            patch_to_cluster[pid] = cid
    return clusters, patch_to_cluster


def _compose_prompt_payload(
    task_id: str,
    task_prompt: str,
    buggy_code: str,
    patch_bank: List[dict],
    patch_clusters: List[dict],
) -> dict:
    prompt_patches = []
    for p in patch_bank:
        prompt_patches.append(
            {
                "patch_id": p["patch_id"],
                "source_explorer": p["source_explorer"],
                "source_role": p["source_role"],
                "old_span": p["old_span"],
                "edit_type": p["edit_type"],
                "ast_locus": p["ast_locus"],
                "evidence": p["candidate_local_evidence"],
                "edit_size": p["edit_size"],
                "risk": p["risk"],
                "unified_diff": p["unified_diff"],
            }
        )
    return {
        "task_id": task_id,
        "task_description": task_prompt,
        "buggy_code": buggy_code,
        "patch_bank": prompt_patches,
        "patch_clusters": patch_clusters,
    }


def _build_decision_only_prompt(payload: dict) -> str:
    schema = {
        "selected_patch_ids": ["P1", "P3"],
        "rejected_patch_ids": [{"patch_id": "P2", "reason": "alternative fix"}],
        "conflict_decisions": [
            {
                "cluster_id": "C1",
                "conflict_type": "same_ast_locus",
                "selected_patch_id": "P1",
                "rejected_patch_ids": ["P2"],
                "reason": "smaller compatible patch",
            }
        ],
        "apply_order": ["P1", "P3"],
        "local_resolutions": [],
    }
    return (
        "You are a Patch Composer for precise code repair.\n\n"
        "Your job is NOT to write the final code.\n"
        "Your job is to select and compose a minimal compatible set of atomic patches from PatchBank.\n\n"
        "Rules:\n"
        "- Output JSON only.\n"
        "- Do NOT output full repaired code.\n"
        "- Select patches only from PatchBank.\n"
        "- Do NOT invent new patches.\n"
        "- Avoid unrelated edits.\n"
        "- If multiple patches target the same semantic location, usually pick at most one.\n"
        "- If conflict needs local resolution, provide only minimal replacement code for target span.\n"
        "- GenDR evidence is candidate-local, not globally conclusive.\n\n"
        "Return JSON with this schema (keys must exist):\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "Input payload:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


def _heuristic_plan(patch_bank: List[dict], patch_clusters: List[dict]) -> dict:
    patch_by_id = {p["patch_id"]: p for p in patch_bank}
    selected_ids = []
    for cluster in patch_clusters:
        candidates = [patch_by_id[pid] for pid in cluster["patch_ids"] if pid in patch_by_id]
        if not candidates:
            continue
        best = sorted(candidates, key=lambda p: (p["edit_size"], p["patch_id"]))[0]
        selected_ids.append(best["patch_id"])
    selected_ids = list(dict.fromkeys(selected_ids))
    return {
        "selected_patch_ids": selected_ids,
        "rejected_patch_ids": [],
        "conflict_decisions": [],
        "apply_order": selected_ids[:],
        "local_resolutions": [],
    }


def _normalize_plan(
    plan_obj: dict | None,
    patch_bank: List[dict],
    patch_clusters: List[dict],
) -> tuple[dict, bool, List[str]]:
    valid = True
    errors: List[str] = []
    patch_ids = {p["patch_id"] for p in patch_bank}
    cluster_ids = {c["cluster_id"] for c in patch_clusters}

    if not isinstance(plan_obj, dict):
        valid = False
        errors.append("plan is not a JSON object")
        return _heuristic_plan(patch_bank, patch_clusters), valid, errors

    selected = plan_obj.get("selected_patch_ids", [])
    if not isinstance(selected, list):
        valid = False
        errors.append("selected_patch_ids must be list")
        selected = []
    selected = [str(x) for x in selected if str(x) in patch_ids]
    selected = list(dict.fromkeys(selected))

    apply_order = plan_obj.get("apply_order", selected)
    if not isinstance(apply_order, list):
        valid = False
        errors.append("apply_order must be list")
        apply_order = selected[:]
    apply_order = [str(x) for x in apply_order if str(x) in selected]
    for pid in selected:
        if pid not in apply_order:
            apply_order.append(pid)

    rejected_raw = plan_obj.get("rejected_patch_ids", [])
    rejected = []
    if isinstance(rejected_raw, list):
        for item in rejected_raw:
            if isinstance(item, dict) and "patch_id" in item:
                rejected.append(
                    {
                        "patch_id": str(item["patch_id"]),
                        "reason": str(item.get("reason", "")),
                    }
                )

    conflicts_raw = plan_obj.get("conflict_decisions", [])
    conflicts = []
    if isinstance(conflicts_raw, list):
        for item in conflicts_raw:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("cluster_id", ""))
            if cid and cid not in cluster_ids:
                valid = False
                errors.append(f"unknown cluster_id in conflict_decisions: {cid}")
                continue
            conflicts.append(
                {
                    "cluster_id": cid,
                    "conflict_type": str(item.get("conflict_type", "")),
                    "selected_patch_id": (
                        None
                        if item.get("selected_patch_id") is None
                        else str(item.get("selected_patch_id"))
                    ),
                    "rejected_patch_ids": [str(x) for x in item.get("rejected_patch_ids", [])],
                    "reason": str(item.get("reason", "")),
                }
            )

    local_raw = plan_obj.get("local_resolutions", [])
    local = []
    if isinstance(local_raw, list):
        for item in local_raw:
            if not isinstance(item, dict):
                continue
            span = item.get("target_old_span")
            if not (isinstance(span, list) and len(span) == 2):
                valid = False
                errors.append("local_resolution.target_old_span must be [start, end]")
                continue
            try:
                start, end = int(span[0]), int(span[1])
            except (TypeError, ValueError):
                valid = False
                errors.append("local_resolution.target_old_span has non-int values")
                continue
            if start > end:
                valid = False
                errors.append("local_resolution.target_old_span start>end")
                continue
            cid = str(item.get("cluster_id", ""))
            if cid and cid not in cluster_ids:
                valid = False
                errors.append(f"unknown cluster_id in local_resolutions: {cid}")
                continue
            local.append(
                {
                    "cluster_id": cid,
                    "target_old_span": [start, end],
                    "allowed_patch_ids": [str(x) for x in item.get("allowed_patch_ids", [])],
                    "resolved_code": str(item.get("resolved_code", "")),
                }
            )

    normalized = {
        "selected_patch_ids": selected,
        "rejected_patch_ids": rejected,
        "conflict_decisions": conflicts,
        "apply_order": apply_order,
        "local_resolutions": local,
    }
    return normalized, valid, errors


def _validate_patch_anchor(code: str, patch: dict) -> tuple[bool, str]:
    lines = code.splitlines()
    for line_no, edit in sorted(patch["diff"].items(), key=lambda x: int(x[0])):
        idx = int(line_no) - 1
        tp = edit["type"]
        if tp in ("Modify", "Delete"):
            if idx < 0 or idx >= len(lines):
                return False, f"line {line_no} out of range"
            if lines[idx].rstrip() != str(edit["original"]).rstrip():
                return False, (
                    f"anchor mismatch at line {line_no}: "
                    f"expected '{edit['original']}', got '{lines[idx]}'"
                )
        elif tp == "Add":
            if idx < 0 or idx > len(lines):
                return False, f"add index {line_no} out of range"
    return True, ""


def _apply_local_resolution(code: str, resolution: dict) -> tuple[str, bool, str]:
    lines = code.splitlines()
    start, end = resolution["target_old_span"]
    if start < 1 or end < 1 or start > len(lines) + 1 or end > len(lines):
        return code, False, f"local resolution span out of range: [{start}, {end}]"
    replacement = resolution["resolved_code"].splitlines()
    lines[start - 1: end] = replacement
    return "\n".join(lines), True, ""


def _detect_textual_conflicts(selected_patches: List[dict]) -> List[Tuple[str, str]]:
    conflicts = []
    for i in range(len(selected_patches)):
        for j in range(i + 1, len(selected_patches)):
            a, b = selected_patches[i], selected_patches[j]
            if _span_overlap(a["old_span"], b["old_span"]):
                conflicts.append((a["patch_id"], b["patch_id"]))
    return conflicts


def _safe_parse_python(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def _apply_plan_deterministically(
    buggy_code: str,
    patch_bank: List[dict],
    patch_to_cluster: Dict[str, str],
    patch_clusters: List[dict],
    plan: dict,
) -> dict:
    patch_by_id = {p["patch_id"]: p for p in patch_bank}
    selected_ids = [pid for pid in plan["selected_patch_ids"] if pid in patch_by_id]
    ordered_ids = [pid for pid in plan["apply_order"] if pid in selected_ids]
    for pid in selected_ids:
        if pid not in ordered_ids:
            ordered_ids.append(pid)
    selected_patches = [patch_by_id[pid] for pid in ordered_ids]

    conflicts = _detect_textual_conflicts(selected_patches)
    conflict_patch_ids = set()
    for a, b in conflicts:
        conflict_patch_ids.add(a)
        conflict_patch_ids.add(b)

    local_by_cluster = {}
    for res in plan["local_resolutions"]:
        cid = res["cluster_id"]
        if cid:
            local_by_cluster[cid] = res

    errors = []
    for pid in conflict_patch_ids:
        cid = patch_to_cluster.get(pid, "")
        if cid and cid not in local_by_cluster:
            errors.append(f"textual conflict in cluster {cid} without local resolution")

    non_conflicting = [p for p in selected_patches if p["patch_id"] not in conflict_patch_ids]
    code = buggy_code
    applied_patch_ids: List[str] = []
    skipped_patch_ids: List[str] = []

    for patch in sorted(non_conflicting, key=lambda p: (p["old_span"][0], p["old_span"][1]), reverse=True):
        ok, msg = _validate_patch_anchor(code, patch)
        if not ok:
            errors.append(f"patch {patch['patch_id']} anchor validation failed: {msg}")
            skipped_patch_ids.append(patch["patch_id"])
            continue
        code = apply_diff(code, patch["diff"], with_delta=True)
        applied_patch_ids.append(patch["patch_id"])

    cluster_by_id = {c["cluster_id"]: c for c in patch_clusters}
    sorted_resolutions = sorted(
        plan["local_resolutions"],
        key=lambda r: (r["target_old_span"][0], r["target_old_span"][1]),
        reverse=True,
    )
    applied_local = []
    for resolution in sorted_resolutions:
        cid = resolution["cluster_id"]
        if cid and cid in cluster_by_id:
            cspan = cluster_by_id[cid]["span"]
            rstart, rend = resolution["target_old_span"]
            if rstart < cspan[0] or rend > cspan[1]:
                errors.append(
                    f"local resolution span {resolution['target_old_span']} exceeds cluster span {cspan} ({cid})"
                )
                continue
        code, ok, msg = _apply_local_resolution(code, resolution)
        if not ok:
            errors.append(msg)
            continue
        applied_local.append(resolution)

    syntax_ok, syntax_err = _safe_parse_python(code)
    if not syntax_ok:
        errors.append(f"final syntax check failed: {syntax_err}")

    return {
        "final_code": code,
        "selected_patch_ids": selected_ids,
        "ordered_selected_patch_ids": ordered_ids,
        "applied_patch_ids": applied_patch_ids,
        "skipped_patch_ids": skipped_patch_ids,
        "textual_conflicts": conflicts,
        "applied_local_resolutions": applied_local,
        "errors": errors,
        "valid": len(errors) == 0,
    }


def _compute_out_of_bank_metrics(
    buggy_code: str,
    final_code: str,
    selected_patches: List[dict],
    local_resolutions: List[dict],
) -> dict:
    _, _, final_diff = file_diff(buggy_code, final_code, cleaned=True)
    selected_keys = set()
    for patch in selected_patches:
        for ln, edit in patch["diff"].items():
            selected_keys.add(_diff_entry_key(ln, edit))

    final_keys = {_diff_entry_key(ln, edit) for ln, edit in final_diff.items()}
    out_of_bank = final_keys - selected_keys

    total = len(final_keys)
    out_rate = (len(out_of_bank) / total) if total else 0.0

    local_spans = [tuple(r["target_old_span"]) for r in local_resolutions]
    local_oob = 0
    for ln, tp, orig, mod in out_of_bank:
        _ = tp, orig, mod
        line_i = int(ln)
        if any(start <= line_i <= end for start, end in local_spans):
            local_oob += 1
    conflict_local_rate = (local_oob / total) if total else 0.0

    return {
        "final_diff_size": total,
        "out_of_bank_edits": len(out_of_bank),
        "out_of_bank_edit_rate": out_rate,
        "conflict_local_out_of_bank_edits": local_oob,
        "conflict_local_out_of_bank_rate": conflict_local_rate,
        "final_diff": final_diff,
    }


def _candidate_edit_size(candidate: dict, key: str = "pruned_edit_size") -> int:
    try:
        return int(candidate.get(key))
    except (TypeError, ValueError):
        return 10**9


def _choose_best_passing_pruned_candidate(pruned_candidates: List[dict]) -> dict | None:
    passing = [
        c for c in pruned_candidates
        if c.get("gendr_meta", {}).get("final_passed") is True
    ]
    if not passing:
        return None
    return sorted(
        passing,
        key=lambda c: (_candidate_edit_size(c), str(c.get("explorer_id", ""))),
    )[0]


def _candidate_test_ratio(candidate: dict) -> float:
    try:
        total = int(candidate.get("num_tests_total") or 0)
        passed = int(candidate.get("num_tests_passed") or 0)
    except (TypeError, ValueError):
        return 0.0
    if total <= 0:
        return 0.0
    return passed / total


def _block_signature(block: dict) -> tuple:
    return tuple(
        sorted(
            (_diff_entry_key(ln, edit) for ln, edit in block.get("diff", {}).items()),
            key=lambda x: (int(x[0]), x[1]),
        )
    )


def _failed_candidate_block_support(
    candidates: List[dict],
    patch_granularity: str,
) -> tuple[Counter, Dict[tuple, List[str]]]:
    support: Counter = Counter()
    explorer_ids_by_sig: Dict[tuple, List[str]] = defaultdict(list)
    for candidate in candidates:
        if candidate.get("unit_pass") or not candidate.get("raw_diff"):
            continue
        seen_in_candidate = set()
        for block in build_blocks(candidate.get("raw_diff", {}), granularity=patch_granularity):
            sig = _block_signature(block)
            if not sig or sig in seen_in_candidate:
                continue
            seen_in_candidate.add(sig)
            support[sig] += 1
            explorer_id = str(candidate.get("explorer_id", ""))
            if explorer_id and explorer_id not in explorer_ids_by_sig[sig]:
                explorer_ids_by_sig[sig].append(explorer_id)
    return support, explorer_ids_by_sig


def _candidate_consensus_density(
    candidate: dict,
    support: Counter,
    patch_granularity: str,
) -> float:
    blocks = build_blocks(candidate.get("raw_diff", {}), granularity=patch_granularity)
    if not blocks:
        return 0.0
    return sum(float(support.get(_block_signature(block), 0)) for block in blocks) / len(blocks)


def _choose_best_partial_candidate(
    candidates: List[dict],
    patch_granularity: str = "line",
) -> dict | None:
    eligible = [
        c for c in candidates
        if not c.get("unit_pass")
        and str(c.get("raw_repair_code", "")).strip()
        and c.get("raw_diff")
    ]
    if not eligible:
        return None
    support, _ = _failed_candidate_block_support(eligible, patch_granularity=patch_granularity)
    return sorted(
        eligible,
        key=lambda c: (
            -_candidate_test_ratio(c),
            _candidate_edit_size(c, "raw_edit_size"),
            -_candidate_consensus_density(c, support, patch_granularity),
            int(c.get("retry_round", 0) or 0),
            str(c.get("explorer_id", "")),
        ),
    )[0]


def _build_partial_patch_bank_from_failed_candidates(
    buggy_code: str,
    candidates: List[dict],
    patch_granularity: str,
    min_support: int = 2,
    patch_start_idx: int = 1,
) -> List[dict]:
    failed = [
        c for c in candidates
        if not c.get("unit_pass") and c.get("raw_diff")
    ]
    if not failed:
        return []

    support, explorer_ids_by_sig = _failed_candidate_block_support(
        failed,
        patch_granularity=patch_granularity,
    )
    threshold = max(1, int(min_support))
    patches: List[dict] = []
    seen_signatures = set()
    patch_idx = patch_start_idx

    for candidate in failed:
        for block in build_blocks(candidate.get("raw_diff", {}), granularity=patch_granularity):
            sig = _block_signature(block)
            if not sig or sig in seen_signatures or support.get(sig, 0) < threshold:
                continue
            seen_signatures.add(sig)
            built = _build_atomic_patches(
                buggy_code=buggy_code,
                pruned_diff=copy.deepcopy(block["diff"]),
                source_explorer=str(candidate.get("explorer_id", "")),
                source_role=str(candidate.get("explorer_role", "")),
                patch_granularity=patch_granularity,
                patch_start_idx=patch_idx,
            )
            for patch in built:
                patch["candidate_local_evidence"] = [
                    (
                        f"This edit appeared in {support[sig]} failed candidates: "
                        f"{', '.join(explorer_ids_by_sig.get(sig, []))}."
                    ),
                    (
                        "No complete explorer candidate passed all unit tests; "
                        "treat this as partial local evidence, not a globally passing repair."
                    ),
                ]
                patch["partial_support_count"] = support[sig]
                patch["partial_support_explorer_ids"] = explorer_ids_by_sig.get(sig, [])
                patch["source_candidate_passed"] = False
                patches.append(patch)
                patch_idx += 1

    return patches


def _apply_final_test_fallback(
    task_id: str,
    final_code: str,
    final_status: str,
    pruned_candidates: List[dict],
    oracle: UnitTestOracle,
) -> tuple[str, str, bool, dict]:
    pre_fallback_pass = bool(oracle.check_one(task_id, final_code))
    fallback_meta = {
        "used": False,
        "attempted": False,
        "reason": "",
        "pre_fallback_status": final_status,
        "pre_fallback_final_pass": pre_fallback_pass,
    }
    if pre_fallback_pass:
        return final_code, final_status, True, fallback_meta

    fallback_candidate = _choose_best_passing_pruned_candidate(pruned_candidates)
    if not fallback_candidate:
        fallback_meta["reason"] = "no_passing_pruned_candidate"
        return final_code, final_status, False, fallback_meta

    fallback_code = fallback_candidate.get("pruned_repair_code", "")
    fallback_meta.update(
        {
            "attempted": True,
            "reason": "final_test_failure",
            "fallback_explorer_id": fallback_candidate.get("explorer_id"),
            "fallback_pruned_edit_size": fallback_candidate.get("pruned_edit_size"),
        }
    )
    fallback_pass = bool(oracle.check_one(task_id, fallback_code))
    fallback_meta["fallback_final_pass"] = fallback_pass
    if not fallback_pass:
        return final_code, final_status, False, fallback_meta

    fallback_meta["used"] = True
    return (
        fallback_code,
        "fallback_best_pruned_due_to_final_test_failure",
        True,
        fallback_meta,
    )


def _resolve_patch_granularity(explicit_value: str | None, mode: str) -> str:
    if explicit_value:
        return explicit_value
    return "hunk" if mode == "multi" else "line"


def _feedback_retry_mode(debug_mode: str) -> str:
    if debug_mode.startswith("free"):
        return "free_with_feedback"
    return "minimal_with_feedback"


def _format_failed_attempts_for_feedback(
    candidates: List[dict],
    max_attempts: int = 5,
    max_feedback_chars: int = 2000,
) -> str:
    failed = [c for c in candidates if not c.get("unit_pass")]
    if not failed:
        return ""
    failed = sorted(failed, key=lambda c: (_candidate_edit_size(c, "raw_edit_size"), str(c.get("explorer_id", ""))))
    chunks = [
        "Previous repair attempts failed the unit tests. Use the feedback below to make a more targeted fix."
    ]
    for idx, cand in enumerate(failed[:max_attempts], start=1):
        feedback = str(cand.get("unit_feedback") or "No detailed unit-test feedback was captured.")
        if len(feedback) > max_feedback_chars:
            feedback = feedback[:max_feedback_chars] + "\n...[truncated]"
        raw_diff = cand.get("raw_diff", {})
        chunks.append(
            "\n".join(
                [
                    f"Attempt {idx}: {cand.get('explorer_id')} ({cand.get('explorer_role')})",
                    f"Predicted diff: {json.dumps(raw_diff, ensure_ascii=False)}",
                    f"Unit-test feedback: {feedback}",
                ]
            )
        )
    return "\n\n".join(chunks)


def _make_candidate_entry(
    role: ExplorerRole,
    solution: str,
    raw_response: str,
    buggy_code: str,
    retry_round: int = 0,
) -> dict:
    _, _, raw_diff = file_diff(buggy_code, solution, cleaned=True)
    explorer_id = role.explorer_id if retry_round <= 0 else f"{role.explorer_id}F{retry_round}"
    return {
        "explorer_id": explorer_id,
        "base_explorer_id": role.explorer_id,
        "explorer_role": role.role_name,
        "prompt_bias": role.prompt_bias,
        "retry_round": retry_round,
        "raw_response": raw_response,
        "raw_repair_code": solution,
        "raw_diff": raw_diff,
        "raw_edit_size": len(raw_diff),
        "raw_diff_hunks": build_blocks(raw_diff, granularity="hunk") if raw_diff else [],
    }


def _evaluate_candidates_with_oracle(
    task_id: str,
    candidates: List[dict],
    oracle: UnitTestOracle,
) -> None:
    if hasattr(oracle, "check_many_with_feedback"):
        results = oracle.check_many_with_feedback(task_id, [c["raw_repair_code"] for c in candidates])
        for c, result in zip(candidates, results):
            c["unit_pass"] = bool(result.get("passed", False))
            c["unit_feedback"] = str(result.get("feedback", ""))
            c["num_tests_passed"] = 1 if c["unit_pass"] else 0
            c["num_tests_total"] = 1
        return

    pass_flags = oracle.check_many(task_id, [c["raw_repair_code"] for c in candidates])
    for c, passed in zip(candidates, pass_flags):
        c["unit_pass"] = bool(passed)
        c["unit_feedback"] = ""
        c["num_tests_passed"] = 1 if passed else 0
        c["num_tests_total"] = 1


def _build_explorer_prompt(task_prompt: str, role: ExplorerRole, test_cases: str | None) -> str:
    bias = (
        f"[Explorer Role]\n"
        f"- id: {role.explorer_id}\n"
        f"- name: {role.role_name}\n"
        f"- guidance: {role.prompt_bias}\n\n"
        "Follow the role guidance while fixing the buggy code."
    )
    if test_cases:
        return (
            f"{task_prompt}\n\n{bias}\n\n"
            "Optional unit tests (context only):\n"
            f"```python\n{test_cases}\n```"
        )
    return f"{task_prompt}\n\n{bias}"


def _create_lm(model_name: str, model_api_file: str | None, temperature: float, max_tokens: int):
    api_key = resolve_api_key(model_name, model_api_file)
    if model_name.split("/")[0] == "together_ai":
        return dspy.LM(
            model_name,
            api_key=api_key,
            api_base="https://api.together.xyz/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            num_retries=3,
        )
    return dspy.LM(
        model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        num_retries=3,
    )


def _process_one_item(
    item: dict,
    explorer_roles: List[ExplorerRole],
    debugger: Debugger,
    composer_lm: Any | None,
    oracle: UnitTestOracle,
    gendr_config: GenDRConfig,
    patch_granularity: str,
    max_patch_bank_size: int,
    dry_run: bool,
    feedback_retry_rounds: int = 0,
) -> dict:
    out = copy.deepcopy(item)
    task_id = str(out.get("task_id", ""))
    task_prompt = out.get("task_prompt", "")
    buggy_code = out.get("buggy_code", "")
    test_cases = out.get("test")

    candidates = []
    for role in explorer_roles:
        if dry_run:
            solution = out.get("gt_solution", buggy_code)
            raw_response = "dry_run"
        else:
            role_prompt = _build_explorer_prompt(task_prompt, role, test_cases)
            try:
                pred = debugger(
                    task_prompt=role_prompt,
                    buggy_code=buggy_code,
                    test_cases=None,
                    failures=None,
                    mode=role.debug_mode,
                )
                solution = pred.solution or ""
                raw_response = solution
            except Exception as e:
                solution = ""
                raw_response = f"ERROR: {e}"

        candidates.append(_make_candidate_entry(role, solution, raw_response, buggy_code))

    _evaluate_candidates_with_oracle(task_id, candidates, oracle)

    initial_passing_candidates_count = len([c for c in candidates if c.get("unit_pass")])
    feedback_retry_used = 0
    if not any(c.get("unit_pass") for c in candidates) and feedback_retry_rounds > 0 and not dry_run:
        for retry_round in range(1, feedback_retry_rounds + 1):
            feedback = _format_failed_attempts_for_feedback(candidates)
            if not feedback:
                break
            retry_candidates = []
            for role in explorer_roles:
                role_prompt = _build_explorer_prompt(task_prompt, role, test_cases)
                try:
                    pred = debugger(
                        task_prompt=role_prompt,
                        buggy_code=buggy_code,
                        test_cases=None,
                        failures=feedback,
                        mode=_feedback_retry_mode(role.debug_mode),
                    )
                    solution = pred.solution or ""
                    raw_response = solution
                except Exception as e:
                    solution = ""
                    raw_response = f"ERROR: {e}"
                retry_candidates.append(
                    _make_candidate_entry(
                        role,
                        solution,
                        raw_response,
                        buggy_code,
                        retry_round=retry_round,
                    )
                )
            _evaluate_candidates_with_oracle(task_id, retry_candidates, oracle)
            candidates.extend(retry_candidates)
            feedback_retry_used = retry_round
            if any(c.get("unit_pass") for c in retry_candidates):
                break

    passing = [c for c in candidates if c["unit_pass"]]
    best_partial_candidate = _choose_best_partial_candidate(
        candidates,
        patch_granularity=patch_granularity,
    )
    if not passing:
        pruned_candidates = []
        patch_bank = _build_partial_patch_bank_from_failed_candidates(
            buggy_code=buggy_code,
            candidates=candidates,
            patch_granularity=patch_granularity,
            min_support=2,
        )
        patch_bank = _dedup_patch_bank(patch_bank)
        if max_patch_bank_size > 0 and len(patch_bank) > max_patch_bank_size:
            patch_bank = sorted(
                patch_bank,
                key=lambda p: (-int(p.get("partial_support_count", 0)), p["edit_size"], p["patch_id"]),
            )[:max_patch_bank_size]
        patch_clusters, patch_to_cluster = _cluster_patches(patch_bank)
        composer_json_valid = True
        composer_errors: List[str] = []
        if patch_bank:
            payload = _compose_prompt_payload(
                task_id=task_id,
                task_prompt=task_prompt,
                buggy_code=buggy_code,
                patch_bank=patch_bank,
                patch_clusters=patch_clusters,
            )
            prompt = _build_decision_only_prompt(payload)
            if dry_run or composer_lm is None:
                composer_plan = _heuristic_plan(patch_bank, patch_clusters)
                composer_raw = json.dumps(composer_plan, ensure_ascii=False, indent=2)
            else:
                composer_raw = _call_lm_text(composer_lm, prompt)
                plan_obj, parse_err = _safe_json_loads(composer_raw)
                composer_plan, composer_json_valid, composer_errors = _normalize_plan(
                    plan_obj,
                    patch_bank=patch_bank,
                    patch_clusters=patch_clusters,
                )
                if parse_err:
                    composer_json_valid = False
                    composer_errors.insert(0, parse_err)
                if not composer_json_valid and not composer_plan["selected_patch_ids"]:
                    composer_plan = _heuristic_plan(patch_bank, patch_clusters)

            apply_meta = _apply_plan_deterministically(
                buggy_code=buggy_code,
                patch_bank=patch_bank,
                patch_to_cluster=patch_to_cluster,
                patch_clusters=patch_clusters,
                plan=composer_plan,
            )
            if apply_meta["valid"]:
                final_code = apply_meta["final_code"]
                final_status = "partial_patch_bank_no_passing_explorer"
            elif best_partial_candidate is not None:
                final_code = best_partial_candidate["raw_repair_code"]
                final_status = "fallback_best_partial_due_to_invalid_partial_plan"
            else:
                final_code = buggy_code
                final_status = "no_passing_explorer_candidate"
        else:
            composer_raw = ""
            composer_plan = _heuristic_plan(patch_bank, patch_clusters)
            if best_partial_candidate is not None:
                final_code = best_partial_candidate["raw_repair_code"]
                final_status = "fallback_best_partial_due_to_no_partial_patch_bank"
            else:
                final_code = buggy_code
                final_status = "no_passing_explorer_candidate"
            apply_meta = {
                "valid": True,
                "errors": [],
                "final_code": final_code,
                "selected_patch_ids": [],
                "ordered_selected_patch_ids": [],
                "applied_patch_ids": [],
                "skipped_patch_ids": [],
                "textual_conflicts": [],
                "applied_local_resolutions": [],
            }
    else:
        passing = sorted(passing, key=lambda c: (c["raw_edit_size"], c["explorer_id"]))
        best_single_explorer = passing[0]

        pruned_candidates = []
        patch_bank = []
        patch_idx = 1
        for cand in passing:
            gendr_input = {
                "task_id": task_id,
                "buggy_code": buggy_code,
                "debug_results": {
                    "solution": cand["raw_repair_code"],
                    "pred_diff": cand["raw_diff"],
                },
            }
            gendr_out = apply_gendr_to_item(gendr_input, oracle=oracle, config=gendr_config)
            gendr_dbg = gendr_out["debug_results"]
            pruned_solution = gendr_dbg["solution"]
            pruned_diff = gendr_dbg["pred_diff"]
            gmeta = gendr_dbg.get("gendr", {})

            pruned_entry = copy.deepcopy(cand)
            pruned_entry["pruned_repair_code"] = pruned_solution
            pruned_entry["pruned_diff"] = pruned_diff
            pruned_entry["pruned_edit_size"] = len(pruned_diff)
            pruned_entry["gendr_meta"] = gmeta
            pruned_candidates.append(pruned_entry)

            patches = _build_atomic_patches(
                buggy_code=buggy_code,
                pruned_diff=pruned_diff,
                source_explorer=cand["explorer_id"],
                source_role=cand["explorer_role"],
                patch_granularity=patch_granularity,
                patch_start_idx=patch_idx,
            )
            patch_idx += len(patches)
            patch_bank.extend(patches)

        pruned_candidates = sorted(
            pruned_candidates,
            key=lambda c: (c["pruned_edit_size"], c["explorer_id"]),
        )
        best_single_pruned = pruned_candidates[0] if pruned_candidates else best_single_explorer

        patch_bank = _dedup_patch_bank(patch_bank)
        if max_patch_bank_size > 0 and len(patch_bank) > max_patch_bank_size:
            patch_bank = sorted(
                patch_bank,
                key=lambda p: (p["edit_size"], p["patch_id"]),
            )[:max_patch_bank_size]

        patch_clusters, patch_to_cluster = _cluster_patches(patch_bank)

        if not patch_bank:
            final_code = best_single_pruned.get("pruned_repair_code", best_single_explorer["raw_repair_code"])
            final_status = "no_patch_after_pruning"
            composer_raw = ""
            composer_plan = _heuristic_plan(patch_bank, patch_clusters)
            composer_json_valid = True
            composer_errors = []
            apply_meta = {
                "valid": True,
                "errors": [],
                "final_code": final_code,
                "selected_patch_ids": [],
                "ordered_selected_patch_ids": [],
                "applied_patch_ids": [],
                "skipped_patch_ids": [],
                "textual_conflicts": [],
                "applied_local_resolutions": [],
            }
        else:
            payload = _compose_prompt_payload(
                task_id=task_id,
                task_prompt=task_prompt,
                buggy_code=buggy_code,
                patch_bank=patch_bank,
                patch_clusters=patch_clusters,
            )
            prompt = _build_decision_only_prompt(payload)
            if dry_run:
                composer_plan = _heuristic_plan(patch_bank, patch_clusters)
                composer_raw = json.dumps(composer_plan, ensure_ascii=False, indent=2)
                composer_json_valid = True
                composer_errors = []
            else:
                composer_raw = _call_lm_text(composer_lm, prompt)
                plan_obj, parse_err = _safe_json_loads(composer_raw)
                composer_plan, composer_json_valid, composer_errors = _normalize_plan(
                    plan_obj,
                    patch_bank=patch_bank,
                    patch_clusters=patch_clusters,
                )
                if parse_err:
                    composer_json_valid = False
                    composer_errors.insert(0, parse_err)
                if not composer_json_valid and not composer_plan["selected_patch_ids"]:
                    composer_plan = _heuristic_plan(patch_bank, patch_clusters)

            apply_meta = _apply_plan_deterministically(
                buggy_code=buggy_code,
                patch_bank=patch_bank,
                patch_to_cluster=patch_to_cluster,
                patch_clusters=patch_clusters,
                plan=composer_plan,
            )

            if apply_meta["valid"]:
                final_code = apply_meta["final_code"]
                final_status = "ok"
            else:
                final_code = best_single_pruned.get("pruned_repair_code", best_single_explorer["raw_repair_code"])
                final_status = "fallback_best_single_pruned_due_to_invalid_plan"

    final_code, final_status, final_pass, fallback_meta = _apply_final_test_fallback(
        task_id=task_id,
        final_code=final_code,
        final_status=final_status,
        pruned_candidates=pruned_candidates,
        oracle=oracle,
    )
    metric_selected_patches = [
        p for p in patch_bank if p["patch_id"] in set(composer_plan["selected_patch_ids"])
    ]
    if fallback_meta.get("used"):
        fallback_explorer_id = fallback_meta.get("fallback_explorer_id")
        fallback_candidate = next(
            (c for c in pruned_candidates if c.get("explorer_id") == fallback_explorer_id),
            None,
        )
        if fallback_candidate:
            metric_selected_patches = _build_atomic_patches(
                buggy_code=buggy_code,
                pruned_diff=fallback_candidate.get("pruned_diff", {}),
                source_explorer=str(fallback_candidate.get("explorer_id", "")),
                source_role=str(fallback_candidate.get("explorer_role", "")),
                patch_granularity=patch_granularity,
            )
    metrics = _compute_out_of_bank_metrics(
        buggy_code=buggy_code,
        final_code=final_code,
        selected_patches=metric_selected_patches,
        local_resolutions=composer_plan.get("local_resolutions", []),
    )
    final_diff = metrics["final_diff"]
    metrics.pop("final_diff", None)

    out["debug_results"] = {
        "model": "epc_decision_only",
        "solution": final_code,
        "pred_diff": final_diff,
    }
    out["epc"] = {
        "status": final_status,
        "final_pass": final_pass,
        "fallback_meta": fallback_meta,
        "initial_passing_candidates_count": initial_passing_candidates_count,
        "feedback_retry_rounds_used": feedback_retry_used,
        "final_selection_source": "pruned_fallback" if fallback_meta.get("used") else "composer_or_initial_fallback",
        "explorer_candidates": candidates,
        "passing_candidates_count": len([c for c in candidates if c.get("unit_pass")]),
        "partial_candidates_count": len(
            [c for c in candidates if not c.get("unit_pass") and c.get("raw_diff")]
        ),
        "best_partial_candidate": (
            {
                "explorer_id": best_partial_candidate.get("explorer_id"),
                "explorer_role": best_partial_candidate.get("explorer_role"),
                "retry_round": best_partial_candidate.get("retry_round"),
                "raw_edit_size": best_partial_candidate.get("raw_edit_size"),
                "num_tests_passed": best_partial_candidate.get("num_tests_passed"),
                "num_tests_total": best_partial_candidate.get("num_tests_total"),
            }
            if best_partial_candidate is not None
            else None
        ),
        "pruned_candidates": pruned_candidates,
        "patch_bank_size": len(patch_bank),
        "patch_bank": patch_bank,
        "patch_clusters": patch_clusters,
        "composer_json_valid": composer_json_valid,
        "composer_errors": composer_errors,
        "composer_raw_output": composer_raw,
        "composer_plan": composer_plan,
        "apply_meta": apply_meta,
        "out_of_bank_metrics": metrics,
    }
    return out


def _resolve_input_file(dataset_name: str, input_file: str) -> str:
    if os.path.isabs(input_file) and os.path.exists(input_file):
        return input_file
    if os.path.exists(input_file):
        return input_file
    candidate = os.path.join("results", dataset_name, "bug_data", input_file)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Cannot locate input file: {input_file}")


def _derive_eval_set_name(input_path: str) -> str:
    return os.path.splitext(os.path.basename(input_path))[0]


def _mean_unit_and_symbolic(scores: dict) -> dict:
    unit = scores.get("Unit score", {})
    sym = scores.get("Symbolic block scores", {})
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True, help="Explorer model name.")
    parser.add_argument(
        "--composer_model_name",
        type=str,
        default=None,
        help="Composer model name. Defaults to --model_name.",
    )
    parser.add_argument("--model_api_file", type=str, default=None)
    parser.add_argument("--composer_model_api_file", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default="")
    parser.add_argument("--eval_set_name", type=str, default=None)

    parser.add_argument("--k_explorers", type=int, default=5, help="Number of explorer roles to use.")
    parser.add_argument("--max_items", type=int, default=50, help="<=0 means all items.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=8000)
    parser.add_argument("--max_patch_bank_size", type=int, default=80, help="<=0 means unlimited.")
    parser.add_argument(
        "--patch_granularity",
        choices=["hunk", "line"],
        default=None,
        help="Default: hunk for --mode multi, line for --mode single.",
    )
    parser.add_argument(
        "--feedback_retry_rounds",
        type=int,
        default=1,
        help="Extra explorer retry rounds with unit-test feedback when no initial explorer passes.",
    )

    parser.add_argument("--gendr_strategy", choices=["sequential", "independent", "hierarchical"], default="sequential")
    parser.add_argument("--gendr_block_granularity", choices=["hunk", "line"], default="hunk")
    parser.add_argument("--gendr_max_blocks", type=int, default=80)
    parser.add_argument("--gendr_timeout_per_task", type=int, default=20)
    parser.add_argument("--gendr_timeout", type=int, default=1800)
    parser.add_argument("--gendr_allow_non_passing_base", action="store_true")

    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--eval_result_dir", type=str, default="results")
    parser.add_argument("--unit_test_timeout", type=int, default=1800)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--tolerance", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Skip LLM calls and use deterministic dry-run behavior.")

    args = parser.parse_args()
    if args.tolerance is None:
        args.tolerance = (
            DEFAULT_TOLERANCE_MULTILINE if args.mode == "multi" else DEFAULT_TOLERANCE_SINGLELINE
        )
    args.patch_granularity = _resolve_patch_granularity(args.patch_granularity, args.mode)

    input_path = _resolve_input_file(args.dataset_name, args.input_file)
    with open(input_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input file must be a list of task dicts.")
    if args.max_items > 0:
        data = _select_items_diverse(data, args.max_items)
    print(f"[EPC] Loaded {len(data)} items from {input_path}")

    eval_set_name = args.eval_set_name or _derive_eval_set_name(input_path)
    model_short = args.model_name.split("/")[-1]
    composer_model_name = args.composer_model_name or args.model_name
    composer_model_short = composer_model_name.split("/")[-1]
    output_tag = f"{args.output_prefix}{model_short}_epc_decision_only_on_{eval_set_name}_round_1"

    output_dir = os.path.join("results", args.dataset_name, "debug_results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{output_tag}.json")

    epc_log_dir = os.path.join("results", args.dataset_name, "epc_log", output_tag)
    os.makedirs(epc_log_dir, exist_ok=True)

    if args.dry_run:
        explorer_lm = None
        composer_lm = None
        debugger = Debugger()
        print("[EPC] DRY RUN: no model API calls")
    else:
        explorer_lm = _create_lm(
            model_name=args.model_name,
            model_api_file=args.model_api_file,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        dspy.settings.configure(lm=explorer_lm)
        debugger = Debugger()
        composer_lm = _create_lm(
            model_name=composer_model_name,
            model_api_file=args.composer_model_api_file or args.model_api_file,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        print(f"[EPC] Explorer model: {args.model_name}")
        print(f"[EPC] Composer model: {composer_model_name}")

    gendr_config = GenDRConfig(
        strategy=args.gendr_strategy,
        block_granularity=args.gendr_block_granularity,
        max_blocks=args.gendr_max_blocks,
        timeout_per_task=args.gendr_timeout_per_task,
        timeout=args.gendr_timeout,
        only_when_fix_passes=not args.gendr_allow_non_passing_base,
    )

    oracle = UnitTestOracle(
        dataset_name=args.dataset_name,
        log_dir=os.path.join(epc_log_dir, "oracle"),
        timeout_per_task=args.gendr_timeout_per_task,
        timeout=args.gendr_timeout,
    )

    explorer_roles = DEFAULT_EXPLORER_ROLES[: max(1, min(args.k_explorers, len(DEFAULT_EXPLORER_ROLES)))]
    print(f"[EPC] Using explorer roles: {[r.explorer_id for r in explorer_roles]}")
    print(f"[EPC] Patch granularity: {args.patch_granularity}")
    print(f"[EPC] Feedback retry rounds: {args.feedback_retry_rounds}")

    results = []
    status_counter = Counter()
    final_pass_count = 0
    pre_fallback_final_pass_count = 0
    final_fallback_used_count = 0
    feedback_retry_item_count = 0
    initial_no_passing_explorer_count = 0
    no_passing_explorer_count = 0
    composer_apply_valid_count = 0
    composer_final_fail_with_passing_candidate_count = 0
    json_valid_count = 0
    patch_util_values = []
    out_of_bank_values = []
    conflict_local_oob_values = []

    for item in tqdm.tqdm(data, desc="EPC decision-only"):
        processed = _process_one_item(
            item=item,
            explorer_roles=explorer_roles,
            debugger=debugger,
            composer_lm=composer_lm,
            oracle=oracle,
            gendr_config=gendr_config,
            patch_granularity=args.patch_granularity,
            max_patch_bank_size=args.max_patch_bank_size,
            dry_run=args.dry_run,
            feedback_retry_rounds=max(0, args.feedback_retry_rounds),
        )
        results.append(processed)
        epc_meta = processed.get("epc", {})
        status_counter[epc_meta.get("status", "unknown")] += 1
        final_pass = bool(epc_meta.get("final_pass", False))
        fallback_meta = epc_meta.get("fallback_meta", {})
        pre_fallback_pass = bool(fallback_meta.get("pre_fallback_final_pass", final_pass))
        passing_count = int(epc_meta.get("passing_candidates_count", 0))
        initial_passing_count = int(epc_meta.get("initial_passing_candidates_count", passing_count))
        final_pass_count += int(final_pass)
        pre_fallback_final_pass_count += int(pre_fallback_pass)
        final_fallback_used_count += int(bool(fallback_meta.get("used", False)))
        feedback_retry_item_count += int(int(epc_meta.get("feedback_retry_rounds_used", 0)) > 0)
        initial_no_passing_explorer_count += int(initial_passing_count == 0)
        no_passing_explorer_count += int(passing_count == 0)
        composer_apply_valid_count += int(bool(epc_meta.get("apply_meta", {}).get("valid", False)))
        composer_final_fail_with_passing_candidate_count += int(passing_count > 0 and not pre_fallback_pass)
        json_valid_count += int(bool(epc_meta.get("composer_json_valid", False)))
        pb_size = int(epc_meta.get("patch_bank_size", 0))
        selected_size = len(epc_meta.get("composer_plan", {}).get("selected_patch_ids", []))
        if pb_size > 0:
            patch_util_values.append(selected_size / pb_size)
        oob = epc_meta.get("out_of_bank_metrics", {})
        out_of_bank_values.append(float(oob.get("out_of_bank_edit_rate", 0.0)))
        conflict_local_oob_values.append(float(oob.get("conflict_local_out_of_bank_rate", 0.0)))

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[EPC] Wrote decision-only results: {output_file}")

    summary = {
        "dataset_name": args.dataset_name,
        "eval_set_name": eval_set_name,
        "explorer_model": args.model_name,
        "composer_model": composer_model_name,
        "patch_granularity": args.patch_granularity,
        "feedback_retry_rounds": max(0, args.feedback_retry_rounds),
        "n_items": len(results),
        "status_counts": dict(status_counter),
        "composition_success_rate": (final_pass_count / len(results)) if results else 0.0,
        "pre_fallback_composition_success_rate": (
            pre_fallback_final_pass_count / len(results)
        ) if results else 0.0,
        "final_fallback_used_count": final_fallback_used_count,
        "final_fallback_used_rate": (final_fallback_used_count / len(results)) if results else 0.0,
        "feedback_retry_item_count": feedback_retry_item_count,
        "feedback_retry_item_rate": (feedback_retry_item_count / len(results)) if results else 0.0,
        "initial_no_passing_explorer_count": initial_no_passing_explorer_count,
        "initial_no_passing_explorer_rate": (
            initial_no_passing_explorer_count / len(results)
        ) if results else 0.0,
        "no_passing_explorer_count": no_passing_explorer_count,
        "no_passing_explorer_rate": (no_passing_explorer_count / len(results)) if results else 0.0,
        "composer_apply_valid_rate": (composer_apply_valid_count / len(results)) if results else 0.0,
        "composer_final_fail_with_passing_candidate_count": composer_final_fail_with_passing_candidate_count,
        "json_valid_rate": (json_valid_count / len(results)) if results else 0.0,
        "patch_bank_utilization_avg": (sum(patch_util_values) / len(patch_util_values)) if patch_util_values else 0.0,
        "out_of_bank_edit_rate_avg": (sum(out_of_bank_values) / len(out_of_bank_values)) if out_of_bank_values else 0.0,
        "conflict_local_out_of_bank_rate_avg": (
            sum(conflict_local_oob_values) / len(conflict_local_oob_values)
            if conflict_local_oob_values
            else 0.0
        ),
        "oracle_calls": oracle.oracle_calls,
        "output_file": output_file,
    }
    summary_file = os.path.join(epc_log_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[EPC] Summary file: {summary_file}")
    print(
        "[EPC][summary] "
        f"n={summary['n_items']} "
        f"comp_success={summary['composition_success_rate']:.3f} "
        f"json_valid={summary['json_valid_rate']:.3f} "
        f"out_of_bank={summary['out_of_bank_edit_rate_avg']:.3f}"
    )

    if not args.no_eval:
        eval_args = SimpleNamespace(
            dataset_name=args.dataset_name,
            eval_result_dir=args.eval_result_dir,
            eval_model_name=f"{composer_model_short}_epc_decision_only",
            eval_set_name=eval_set_name,
            stride=args.stride,
            tolerance=args.tolerance,
            unit_test_timeout=args.unit_test_timeout,
        )
        evaluator = Evaluator(eval_args)
        scores = evaluator.run_evaluation(results=results, round=1)
        means = _mean_unit_and_symbolic(scores)
        eval_summary_file = os.path.join(epc_log_dir, "eval_means.json")
        with open(eval_summary_file, "w") as f:
            json.dump(means, f, indent=2)
        print(
            "[EPC][eval] "
            f"unit={means['unit']:.3f} "
            f"prec={means['precision']:.3f} "
            f"rec={means['recall']:.3f} "
            f"f1={means['f1']:.3f} "
            f"(n={means['n']})"
        )


if __name__ == "__main__":
    main()

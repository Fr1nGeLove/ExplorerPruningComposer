# EPC Partial Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve useful local edits from failed EPC explorer candidates and avoid zero-information buggy fallback when no complete candidate passes unit tests.

**Architecture:** Add failed-candidate partial patch extraction to `src/epc_decision_only.py`, then expose a no-composer hybrid fallback that can choose the best partial explorer candidate from existing traces. Keep the default path conservative enough to preserve current passing-candidate behavior.

**Tech Stack:** Python, unittest, existing EPC/GenDR utilities.

---

### Task 1: Partial Patch Bank

**Files:**
- Modify: `tests/test_epc_decision_only.py`
- Modify: `src/epc_decision_only.py`

- [ ] **Step 1: Write failing tests** for building deduplicated partial patches from failed candidates and choosing the best partial candidate.
- [ ] **Step 2: Run:** `python -m unittest tests.test_epc_decision_only -v`
- [ ] **Step 3: Implement helper functions** in `src/epc_decision_only.py`.
- [ ] **Step 4: Re-run:** `python -m unittest tests.test_epc_decision_only -v`

### Task 2: Hybrid Partial Fallback

**Files:**
- Modify: `tests/test_epc_decision_only.py`
- Modify: `scripts/epc_hybrid_gated_eval.py`

- [ ] **Step 1: Write failing test** that no-pass hybrid selection uses a partial failed explorer instead of buggy code.
- [ ] **Step 2: Run:** `python -m unittest tests.test_epc_decision_only -v`
- [ ] **Step 3: Implement fallback selection in `scripts/epc_hybrid_gated_eval.py`.**
- [ ] **Step 4: Re-run unit tests and evaluate the existing BigCodeBench PDB-SINGLE-HARD first100 trace.**

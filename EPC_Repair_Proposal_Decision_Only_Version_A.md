# Proposal: Explorer–Pruner–Decision-Only Composer for Precise Code Repair

## 0. Version Scope

This proposal describes **Version A** of the Explorer–Pruner–Composer framework.

The key design change in this version is:

> The Composer does **not** output the final repaired code.  
> It only outputs a structured patch-selection plan.  
> The final code is produced by a deterministic backend that applies selected patches to the original buggy code.

This design decouples:

```text
LLM reasoning ability:
  select patches, reject patches, resolve conflicts

from

token-level code generation:
  rewriting or copying the whole program
```

The goal is to avoid the **autoregressive rewriting trap**: when a model is asked to output the full final code, it may unintentionally reformat, rename, or rewrite unrelated parts of the program, even if it selected the right patches.

---

## 1. Core Research Question

Given:

```text
1. A buggy program C_b
2. Multiple explorer-generated repair candidates
3. GenDR-pruned atomic patches extracted from those candidates
```

Can a strong code model act as a **discrete patch-level decision maker** and select a minimal compatible set of patches, without directly generating the full repaired program?

In short:

> Can we transform code repair from open-ended code generation into symbolic patch selection?

---

## 2. High-Level Pipeline

```text
Input:
  - Task description x
  - Original buggy code C_b
  - Unit tests U
  - K explorer prompts or agents

Output:
  - Patch selection plan
  - Deterministically composed final code
  - Composition metadata
  - Evaluation results
```

Pipeline:

```text
Phase 1: Explorer Generation
  Multiple explorers generate diverse repair candidates.

Phase 2: GenDR Pruning
  Each passing candidate is minimized by GenDR.
  The remaining edits are converted into atomic patches.

Phase 3: Decision-Only Pure-Prompt Composer
  The model receives a structured PatchBank.
  It outputs only patch IDs, rejection reasons, conflict decisions, and optional local conflict resolutions.

Phase 4: Deterministic Patch Application
  A Python backend applies the selected patches to C_b.
  Non-conflicting patches are applied mechanically.

Phase 5: Evaluation
  The final code is evaluated with pass/unit and precision/edit metrics.
```

---

## 3. Main Hypothesis

The main hypothesis is:

> Pruner-certified PatchBanks turn precise code repair into a discrete decision problem.  
> Strong code models are better at selecting and combining symbolic patch options than at generating an entire repaired program without over-editing.

This tests a stronger claim than standard prompting:

```text
Do not merely ask the model to "write code carefully."
Instead, discretize candidate edits into a symbolic PatchBank
and ask the model to make patch-level decisions.
```

---

## 4. Phase 1: Explorer Generation

### 4.1 Explorer Roles

Use multiple explorers to generate diverse repair candidates.

Recommended initial setting: `K = 5`.

| Explorer | Prompt Bias | Purpose |
|---|---|---|
| E1: Minimal Explorer | Make the smallest possible fix | High precision candidate |
| E2: Boundary Explorer | Focus on edge cases and boundary conditions | Off-by-one / empty input / corner cases |
| E3: Data-flow Explorer | Focus on assignments, state updates, returns | Data-flow bugs |
| E4: Control-flow Explorer | Focus on branches, loops, early returns | Control-flow bugs |
| E5: Freeform Explorer | Fix the code without minimal-edit constraint | High recall candidate |

The system should allow ablations:

```text
K ∈ {3, 5, 8}
same-prompt sampling vs role-specific explorers
single-model explorers vs multi-model explorers
```

### 4.2 Explorer Output

Each explorer returns a complete repaired program:

```text
P_i = Explorer_i(C_b, x)
```

For each candidate, save:

```json
{
  "explorer_id": "E2",
  "explorer_role": "Boundary Explorer",
  "raw_repair_code": "...",
  "unit_pass": true,
  "num_tests_passed": 20,
  "num_tests_total": 20,
  "raw_edit_size": 8,
  "raw_diff_hunks": []
}
```

### 4.3 Candidate Filtering

For Version A, start with the cleanest setting:

```text
Keep only candidates that pass all available tests.
```

If no candidate passes all tests, mark the example as:

```text
no_passing_explorer_candidate
```

Later versions can incorporate partial candidates, but Version A should first evaluate composition over pruner-certified passing candidates.

---

## 5. Phase 2: GenDR Pruning

Each passing explorer candidate is independently minimized by GenDR:

```text
A_i = GenDR(C_b, P_i, U)
```

Where:

```text
C_b = original buggy code
P_i = explorer repair
U = unit tests
A_i = pruned repair / minimal patch core
```

### 5.1 Atomic Patch Extraction

After GenDR pruning, convert the remaining edits into atomic patches.

Each patch must be represented relative to the original buggy code `C_b`, not relative to the explorer output.

This avoids coordinate drift when multiple patches are composed.

Recommended object:

```json
{
  "patch_id": "P3",
  "source_explorer": "E2",
  "source_role": "Boundary Explorer",

  "base_code_hash": "sha256(C_b)",

  "old_span": [11, 11],
  "old_text": "    while left < right:",
  "new_text": "    while left <= right:",

  "context_before": "    mid = (left + right) // 2",
  "context_after": "        if arr[mid] == target:",

  "edit_type": "condition_change",
  "ast_locus": "While.test",
  "defs": [],
  "uses": ["left", "right"],

  "candidate_local_evidence": [
    "The source explorer repair passed all tests.",
    "Within that source repair, GenDR attempted to revert this edit and tests failed, so the edit was necessary in that candidate."
  ],

  "edit_size": 1,
  "risk": "medium"
}
```

Important: GenDR evidence is **candidate-local**, not globally conclusive.

Prompt wording should make this explicit:

```text
GenDR evidence means the edit was necessary inside its source candidate repair.
If multiple patches fix the same semantic locus, select the smallest compatible one.
Do not select all patches merely because they all have local necessity evidence.
```

---

## 6. Phase 3: Decision-Only Pure-Prompt Composer

### 6.1 Composer Objective

The Composer receives:

```text
1. Task description
2. Original buggy code
3. PatchBank
4. Patch clusters
5. Candidate-local GenDR evidence
```

It outputs:

```text
1. selected_patch_ids
2. rejected_patch_ids with reasons
3. conflict_decisions
4. apply_order
5. optional local_resolutions only for overlapping conflicts
```

It must **not** output the full final repaired code.

Primary objective:

```text
Select the smallest compatible set of PatchBank patches likely to produce a correct repair.
```

Secondary objective:

```text
Avoid unrelated edits by construction.
```

---

### 6.2 Why Decision-Only?

The previous pure-prompt design asked the Composer to output:

```text
Final Code:
  output the complete repaired program
```

This is risky because full-code generation can trigger autoregressive rewriting:

```text
The model may select the right patches,
but while copying the rest of the code,
it may rename variables, reformat statements, add guards, or rewrite unrelated logic.
```

Decision-Only Composer avoids this by making the model output only symbolic decisions.

Then a deterministic backend applies the selected patches.

For non-conflicting patches:

```text
out-of-bank edits = 0 by construction
```

This makes the experiment cleaner:

```text
Measured ability = patch-level reasoning
Not measured = full-program copying fidelity
```

---

### 6.3 Composer Prompt Template

Use the following prompt template.

```text
You are a Patch Composer for precise code repair.

Your job is NOT to write the final code.
Your job is to select and compose a minimal compatible set of atomic patches from the provided PatchBank.

You are given:
1. The task description.
2. The original buggy code.
3. A PatchBank. Each patch was extracted from an explorer repair and pruned by GenDR.
4. Patch clusters indicating patches that affect the same semantic location.
5. Candidate-local evidence for each patch.

Rules:
- Output JSON only.
- Do not output full repaired code.
- Select patches only from the PatchBank.
- Do not invent new patches.
- Do not introduce unrelated edits.
- If multiple patches modify the same semantic location, usually select at most one.
- If patches modify disjoint locations and appear complementary, you may select them together.
- If two patches overlap textually or semantically, explain the conflict.
- Prefer the smallest set of patches that best repairs the program.
- GenDR evidence is candidate-local, not global. Do not select all locally necessary patches if they are alternative fixes for the same issue.
- If a conflict requires local resolution, output only the minimal replacement code for the conflict span, not the whole program.

Return JSON with the following schema:
{
  "selected_patch_ids": ["P1", "P3"],
  "rejected_patch_ids": [
    {
      "patch_id": "P2",
      "reason": "Alternative to P1 but larger."
    }
  ],
  "conflict_decisions": [
    {
      "cluster_id": "C1",
      "conflict_type": "same_ast_locus",
      "selected_patch_id": "P1",
      "rejected_patch_ids": ["P2"],
      "reason": "P1 is the smallest compatible fix for this condition."
    }
  ],
  "apply_order": ["P1", "P3"],
  "local_resolutions": []
}
```

---

### 6.4 PatchBank Formatting

Each patch should be shown using both structured metadata and a unified diff.

Unified diff is recommended because code models have strong priors from GitHub pull requests and code review data.

Example:

```diff
Patch ID: P3
Source: E2 — Boundary Explorer
Cluster: C1 — binary_search_loop_condition
Edit Type: condition_change
AST Locus: While.test
Evidence:
  - Source explorer repair passed all tests.
  - Candidate-local GenDR evidence: reverting this edit inside the source repair caused tests to fail.
Edit Size: 1
Risk: medium

@@ -10,5 +10,5 @@
     mid = (left + right) // 2
-    while left < right:
+    while left <= right:
         if arr[mid] == target:
             return mid
```

For patches affecting the same semantic location, group them into a cluster:

```text
Patch Cluster C1: binary_search_loop_condition
Semantic Locus: While.test

Patches in this cluster:
  - P3
  - P7
  - P9

Instruction:
  These patches affect the same semantic location.
  Select at most one unless there is a clear reason to combine them.
```

---

### 6.5 Composer Input Schema

Recommended input object:

```json
{
  "task_id": "example_001",
  "task_description": "...",
  "buggy_code": "...",
  "patch_bank": [
    {
      "patch_id": "P1",
      "source_explorer": "E1",
      "source_role": "Minimal Explorer",
      "old_span": [8, 8],
      "old_text": "if x < n:",
      "new_text": "if x <= n:",
      "edit_type": "operator_change",
      "ast_locus": "If.test",
      "evidence": [
        "Source explorer repair passed all tests.",
        "Candidate-local GenDR evidence: reverting this edit caused tests to fail."
      ],
      "edit_size": 1,
      "unified_diff": "@@ -8,3 +8,3 @@\n- if x < n:\n+ if x <= n:"
    }
  ],
  "patch_clusters": [
    {
      "cluster_id": "C1",
      "semantic_locus": "If.test at line 8",
      "patch_ids": ["P1", "P4", "P8"],
      "cluster_note": "Alternative condition fixes."
    }
  ]
}
```

---

### 6.6 Composer Output Schema

The model must output JSON only.

Example without conflict:

```json
{
  "selected_patch_ids": ["P1", "P6"],
  "rejected_patch_ids": [
    {
      "patch_id": "P4",
      "reason": "Alternative to P1 but larger."
    },
    {
      "patch_id": "P8",
      "reason": "Redundant with P1."
    }
  ],
  "conflict_decisions": [
    {
      "cluster_id": "C1",
      "conflict_type": "same_ast_locus",
      "selected_patch_id": "P1",
      "rejected_patch_ids": ["P4", "P8"],
      "reason": "P1 is the smallest condition fix."
    }
  ],
  "apply_order": ["P1", "P6"],
  "local_resolutions": []
}
```

Example with local conflict resolution:

```json
{
  "selected_patch_ids": ["P2", "P5"],
  "rejected_patch_ids": [],
  "conflict_decisions": [
    {
      "cluster_id": "C3",
      "conflict_type": "overlapping_text_span",
      "selected_patch_id": null,
      "rejected_patch_ids": [],
      "requires_local_resolution": true,
      "reason": "P2 and P5 modify the same return statement but fix different aspects."
    }
  ],
  "apply_order": ["P2", "P5"],
  "local_resolutions": [
    {
      "cluster_id": "C3",
      "target_old_span": [18, 20],
      "allowed_patch_ids": ["P2", "P5"],
      "resolved_code": "    return result if result is not None else []"
    }
  ]
}
```

Local resolution rules:

```text
- Only allowed for overlapping conflict spans.
- Must be limited to the target_old_span.
- Must not rewrite unrelated code.
- Must be validated by the backend before application.
```

---

## 7. Phase 4: Deterministic Patch Application

### 7.1 Backend Responsibilities

The backend receives:

```text
C_b
PatchBank
Composer JSON decision plan
```

It then:

```text
1. Validates selected_patch_ids.
2. Validates that selected patches exist in PatchBank.
3. Detects textual and semantic conflicts.
4. Applies non-conflicting patches mechanically.
5. Applies local_resolutions only within allowed conflict spans.
6. Parses the resulting code.
7. Runs tests.
8. Computes out-of-bank edit metrics.
```

### 7.2 Patch Application Rules

Recommended rules:

```text
1. All patches are relative to original C_b.
2. Verify base_code_hash before applying.
3. Verify old_text matches C_b at old_span or within a small context window.
4. Apply non-overlapping patches from bottom to top to avoid line-number drift.
5. Do not automatically merge overlapping patches.
6. If selected patches overlap and no local_resolution is provided, reject the composition plan.
7. If local_resolution modifies outside target_old_span, reject it.
```

### 7.3 Deterministic Apply Pseudocode

```python
def apply_composer_plan(C_buggy, patch_bank, plan):
    validate_json_schema(plan)
    selected = [patch_bank[pid] for pid in plan["selected_patch_ids"]]

    conflicts = detect_conflicts(selected)

    if conflicts:
        validate_conflict_decisions(conflicts, plan["conflict_decisions"])
        validate_local_resolutions(conflicts, plan["local_resolutions"])

    non_conflicting = remove_conflicting_patches(selected, conflicts)

    code = C_buggy

    # Apply from bottom to top to avoid span drift.
    for patch in sorted(non_conflicting, key=lambda p: p.old_span[0], reverse=True):
        code = apply_single_patch(code, patch)

    # Apply local resolutions only inside validated spans.
    for resolution in plan["local_resolutions"]:
        code = apply_local_resolution(code, resolution)

    return code
```

---

## 8. Evaluation for Version A

Version A should answer:

> Can a strong code model perform patch-level composition through prompting alone, if code generation is physically removed from the output channel?

### 8.1 Main Baselines

Compare:

| Method | Description |
|---|---|
| Best Single Explorer | Select one passing explorer output with smallest edit size |
| Best Single Explorer + GenDR | Apply GenDR to the best single explorer |
| Explorer + Pruner | Independently pruned explorer patches; choose best pruned candidate |
| Free Synthesizer | Give all explorer outputs to model and ask it to write final code freely |
| Old Pure-Prompt Composer | Model selects patches and outputs full final code |
| Decision-Only Composer | Model outputs only patch-selection JSON; backend applies patches |

The most important comparisons are:

```text
Decision-Only Composer vs Free Synthesizer
Decision-Only Composer vs Old Pure-Prompt Composer
Decision-Only Composer vs Best Single Explorer + GenDR
```

---

## 9. Metrics

### 9.1 PDB Metrics

```text
Unit
Precision
Recall
F1
```

### 9.2 PRepair-style Metrics

```text
pass@1/5/10/20
edit_1.0@1/5/10/20
edit_1.5@1/5/10/20
edit_2.0@1/5/10/20
average edit cost
```

### 9.3 Composer-Specific Metrics

#### Out-of-Bank Edit Rate

```text
out_of_bank_edit_rate =
  number of final edits not attributable to selected PatchBank patches
  / total number of final edits
```

For non-conflicting deterministic application:

```text
out_of_bank_edit_rate = 0 by construction
```

#### Conflict-Local Out-of-Bank Rate

```text
conflict_local_out_of_bank_rate =
  number of generated edits inside conflict spans
  / total final edits
```

#### PatchBank Utilization

```text
#selected patches / #available patches
```

#### Composition Success Rate

```text
#examples where composed final code passes tests
/ #examples evaluated
```

#### Conflict Resolution Validity

```text
#valid local resolutions / #local resolutions proposed
```

#### JSON Validity Rate

```text
#valid composer JSON outputs / #composer calls
```

This is important because the Composer is required to output machine-actionable decisions.

---

## 10. Expected Outcomes

### Positive Outcome A

```text
Decision-Only Composer improves recall/F1 over best single pruned candidate.
```

Interpretation:

> The model successfully combines complementary atomic patches from different explorers.

### Positive Outcome B

```text
Decision-Only Composer has much lower out-of-bank edit rate than Free Synthesizer or Old Pure-Prompt Composer.
```

Interpretation:

> The PatchBank decision interface prevents over-editing by construction.

### Positive Outcome C

```text
Decision-Only Composer helps more on multi-bug examples than on single-bug examples.
```

Interpretation:

> Composition is most valuable when different explorers discover different bugs.

### Negative Outcome A

```text
Decision-Only Composer rarely improves over the best single pruned candidate.
```

Interpretation:

> Either most useful patches are already contained in one candidate, or the model fails to identify complementary patches.

Mitigation:

```text
Move to Version B: algorithmic or beam-search Composer.
```

### Negative Outcome B

```text
Decision-Only Composer selects conflicting patches or invalid patch IDs.
```

Mitigation:

```text
Improve PatchBank clustering and add stricter JSON validation / repair.
```

---

## 11. Implementation Checklist

### 11.1 Explorer Layer

- [ ] Implement role-specific explorer prompts.
- [ ] Generate K candidate repairs.
- [ ] Run unit tests on each candidate.
- [ ] Extract raw diffs and edit sizes.
- [ ] Save candidate metadata.

### 11.2 Pruner Layer

- [ ] Run GenDR on each passing candidate.
- [ ] Extract remaining hunks.
- [ ] Normalize atomic patches relative to original `C_b`.
- [ ] Record candidate-local GenDR evidence.
- [ ] Build PatchBank.
- [ ] Cluster patches by textual span and AST locus.

### 11.3 Composer Layer

- [ ] Serialize task description, buggy code, PatchBank, and clusters.
- [ ] Use unified diff format with context lines.
- [ ] Prompt model to output JSON only.
- [ ] Parse composer JSON.
- [ ] Validate selected patch IDs.
- [ ] Validate conflict decisions and local resolutions.

### 11.4 Patch Application Layer

- [ ] Implement deterministic patch apply.
- [ ] Apply non-overlapping patches bottom-to-top.
- [ ] Reject invalid overlapping selections unless local_resolution is provided.
- [ ] Validate local_resolution span.
- [ ] Parse resulting code.
- [ ] Run tests.

### 11.5 Evaluation Layer

- [ ] Compute PDB metrics.
- [ ] Compute PRepair-style metrics if using PRepair benchmark.
- [ ] Compute out-of-bank edit rate.
- [ ] Compute JSON validity rate.
- [ ] Compare against Free Synthesizer and Old Pure-Prompt Composer.
- [ ] Save logs for case studies.

---

## 12. Suggested Files / Modules

```text
explorers/
  prompts.py
  generate_candidates.py

pruner/
  gendr.py
  atomic_patch.py
  patch_bank.py
  patch_clustering.py

composer/
  decision_only_prompt.py
  decision_only_composer.py
  parse_composer_json.py
  validate_composer_plan.py

patch_apply/
  apply_patch.py
  conflict_detection.py
  local_resolution.py

evaluation/
  run_tests.py
  pdb_metrics.py
  prepair_metrics.py
  edit_metrics.py
  out_of_bank_metrics.py

experiments/
  run_version_a_decision_only.py
  ablation_composer_interfaces.py
  analyze_composer_failures.py
```

---

## 13. End-to-End Pseudocode

```python
def run_version_a_decision_only(task, explorers, tests, composer_model):
    C_b = task.buggy_code
    x = task.description

    # Phase 1: Explorer generation
    candidates = []
    for explorer in explorers:
        repair = explorer.generate(x, C_b)
        result = run_tests(repair, tests)
        candidates.append({
            "explorer": explorer.name,
            "role": explorer.role,
            "repair": repair,
            "test_result": result,
            "edit_size": compute_edit_size(C_b, repair)
        })

    passing_candidates = [
        c for c in candidates
        if c["test_result"].pass_all
    ]

    if len(passing_candidates) == 0:
        return {
            "status": "no_passing_explorer_candidate",
            "candidates": candidates
        }

    # Phase 2: GenDR pruning and PatchBank construction
    patch_bank = []
    for c in passing_candidates:
        pruned_code, trace = gendr_prune(C_b, c["repair"], tests)

        atomic_patches = extract_atomic_patches_relative_to_base(
            buggy_code=C_b,
            pruned_code=pruned_code,
            trace=trace,
            source_explorer=c["explorer"],
            source_role=c["role"]
        )

        patch_bank.extend(atomic_patches)

    patch_clusters = cluster_patches(patch_bank)

    # Phase 3: Decision-only composition
    prompt = build_decision_only_prompt(
        task_description=x,
        buggy_code=C_b,
        patch_bank=patch_bank,
        patch_clusters=patch_clusters
    )

    raw_output = composer_model.generate(prompt)
    plan = parse_and_validate_composer_json(raw_output, patch_bank, patch_clusters)

    # Phase 4: Deterministic patch application
    final_code = apply_composer_plan(
        C_buggy=C_b,
        patch_bank=patch_bank,
        plan=plan
    )

    # Phase 5: Evaluation
    final_test_result = run_tests(final_code, tests)

    out_of_bank_rate = compute_out_of_bank_edit_rate(
        C_buggy=C_b,
        final_code=final_code,
        selected_patches=[
            p for p in patch_bank
            if p.patch_id in plan["selected_patch_ids"]
        ],
        local_resolutions=plan.get("local_resolutions", [])
    )

    return {
        "status": "ok",
        "final_code": final_code,
        "composer_plan": plan,
        "final_test_result": final_test_result,
        "out_of_bank_edit_rate": out_of_bank_rate,
        "patch_bank": patch_bank,
        "patch_clusters": patch_clusters,
        "candidates": candidates
    }
```

---

## 14. Case Studies to Save

### Case 1: Composer combines complementary patches

```text
Explorer E1 finds bug A.
Explorer E2 finds bug B.
Decision-Only Composer selects both patches.
Deterministic backend applies both.
Final code passes with small edit size.
```

### Case 2: Composer rejects larger alternative

```text
P1 and P4 modify the same condition.
P1 is one-line minimal fix.
P4 includes extra guard or refactor.
Composer selects P1 and rejects P4.
```

### Case 3: Free Synthesizer over-edits

```text
Given the same PatchBank or explorer outputs,
Free Synthesizer writes full code and changes unrelated lines.
Decision-Only Composer avoids those changes because it only selects patch IDs.
```

### Case 4: Old Pure-Prompt Composer suffers autoregressive trap

```text
The model selects correct patches but introduces unrelated edits while outputting the full code.
Decision-Only Composer avoids this failure by not producing final code.
```

### Case 5: Conflict resolution needed

```text
Two patches overlap the same return statement.
Composer requests local_resolution for only that span.
Backend validates the resolution and rejects any broader rewrite.
```

### 14.1 Empirical Case Studies from BigCodeBench t240

The following three cases are selected from the unified `t240` BigCodeBench evaluation.
They illustrate the main qualitative difference between EPC hybrid partial and
Agentic+GenDR: both methods can pass unit tests, but EPC more often preserves the
minimal ground-truth edit structure.

#### Case A: Single-hard one-line repair where both methods pass tests

Task: `BigCodeBench/0_2`

Setting: `PDB-SINGLE-HARD first100`

Bug type: `Assignment / Variable Initialization`

| Method | Unit | Precision | Recall | F1 | Predicted edit count |
|---|---:|---:|---:|---:|---:|
| EPC hybrid partial | 1 | 1.0000 | 1.0000 | 1.0000 | 1 |
| Agentic+GenDR | 1 | 0.0000 | 0.0000 | 0.0000 | 5 |

Ground-truth edit:

```diff
-    permutations = itertools.permutations(numbers)
+    permutations = list(itertools.permutations(numbers))
```

EPC hybrid partial predicts exactly the ground-truth edit:

```diff
-    permutations = itertools.permutations(numbers)
+    permutations = list(itertools.permutations(numbers))
```

Agentic+GenDR also passes the unit tests, but it does not repair the source of
the bug. Instead, it adds an alternate counter-based implementation:

```diff
+    count = 0
+        count += 1
-    avg_sum_diffs = sum_diffs / len(permutations)
+    if count == 0:
+        return 0.0
+    avg_sum_diffs = sum_diffs / count
```

Interpretation:

```text
This is a clean example where unit accuracy is insufficient.
Both outputs pass tests, but Agentic+GenDR solves the observed behavior through
a different local rewrite. EPC hybrid partial repairs the exact root cause with
one edit, so it receives full symbolic credit.
```

#### Case B: Multi-edit repair where Agentic+GenDR over-edits while still passing tests

Task: `BigCodeBench/1028_2`

Setting: `PDB-MULTI 100`

Bug type: `Build/Package/Merge / Dependency Version Conflicts`

| Method | Unit | Precision | Recall | F1 | Predicted edit count |
|---|---:|---:|---:|---:|---:|
| EPC hybrid partial | 1 | 1.0000 | 1.0000 | 1.0000 | 2 |
| Agentic+GenDR | 1 | 0.0000 | 0.0000 | 0.0000 | 19 |

Ground-truth edit:

```diff
-                    dist_name, _, _ = platform.linux_distribution()
-                    command = ["vmstat", "1", "1"] if dist_name == "Ubuntu" else ["top", "-b", "-n1"]
+                    # Unix/Linux command for CPU usage
+                    command = ["top", "-b", "-n1"]
```

EPC hybrid partial selects a compact patch that is semantically aligned with the
ground-truth block:

```diff
-                    dist_name, _, _ = platform.linux_distribution()
+                    command = ["top", "-b", "-n1"]
-                    command = ["vmstat", "1", "1"] if dist_name == "Ubuntu" else ["top", "-b", "-n1"]
```

Agentic+GenDR passes the tests, but it performs a broader local rewrite of the
CPU parsing logic:

```diff
-                    dist_name, _, _ = platform.linux_distribution()
+                    command = ["vmstat", "1", "1"]
-                cpu_usage_line = (
-                    output.decode("utf-8").split("\n")[2]
-                    if platform.system() == "Windows"
-                    else output.decode("utf-8").split("\n")[2]
-                )
-                cpu_usage = (
-                    cpu_usage_line.split(",")[-1].strip().replace('"', "")
-                    if platform.system() == "Windows"
-                    else cpu_usage_line.split(":")[1].split(",")[0].strip()
-                )
+                lines = output.decode("utf-8").split("\n")
+                if platform.system() == "Windows":
+                    # Filter non-empty lines that contain the CPU usage data
+                    cpu_usage_line = [line for line in lines if line and not line.startswith('"')]
+                    ...
+                else:
+                    # Filter lines that contain "Cpu(s)" for Linux output
+                    ...
```

Interpretation:

```text
This case shows the advantage of patch-level selection in multi-edit repair.
Agentic+GenDR can find a passing behavior, but it introduces many extra edits
around output parsing. EPC hybrid partial keeps the repair localized to the
dependency-sensitive command selection block.
```

#### Case C: Partial fallback recovers exact local edits from a non-passing candidate

Task: `BigCodeBench/14_40`

Setting: `PDB-SINGLE-HARD first100`

EPC source: `best_partial_failed_explorer`

| Method | Unit | Precision | Recall | F1 | Predicted edit count |
|---|---:|---:|---:|---:|---:|
| EPC hybrid partial | 0 | 1.0000 | 1.0000 | 1.0000 | 3 |
| Agentic+GenDR | 0 | 0.0000 | 0.0000 | 0.0000 | 6 |

Ground-truth edit:

```diff
-    project_dir = os.path.basename(config.get('Project', 'directory'))
+    project_dir = config.get('Project', 'directory')
-    project_dir = os.path.dirname(project_dir)
-    shutil.make_archive(base_name=os.path.splitext(archive_file)[0], format='gztar', root_dir=project_dir)
+    shutil.make_archive(base_name=os.path.splitext(archive_file)[0], format='zip', root_dir=project_dir)
```

EPC hybrid partial predicts the exact same three local edits:

```diff
-    project_dir = os.path.basename(config.get('Project', 'directory'))
+    project_dir = config.get('Project', 'directory')
-    project_dir = os.path.dirname(project_dir)
-    shutil.make_archive(base_name=os.path.splitext(archive_file)[0], format='gztar', root_dir=project_dir)
+    shutil.make_archive(base_name=os.path.splitext(archive_file)[0], format='zip', root_dir=project_dir)
```

Agentic+GenDR edits nearby code, but each edit is broader or shifted relative to
the ground-truth repair:

```diff
-    project_dir = os.path.basename(config.get('Project', 'directory'))
+    if not os.path.isfile(config_file_path):
+        raise FileNotFoundError(f'Config file {config_file_path} does not exist.')
+    project_dir = config.get('Project', 'directory')
-    project_dir = os.path.dirname(project_dir)
+    archive_file = os.path.join(archieve_dir, os.path.basename(project_dir) + '.zip')
-    archive_file = f'{archieve_dir}/{os.path.basename(project_dir)}.zip'
-    shutil.make_archive(base_name=os.path.splitext(archive_file)[0], format='gztar', root_dir=project_dir)
+    shutil.make_archive(base_name=os.path.splitext(archive_file)[0], format='zip', root_dir=os.path.dirname(project_dir), base_dir=os.path.basename(project_dir))
```

Interpretation:

```text
This case isolates the value of partial fallback.
Even though no complete explorer candidate passed the tests, the failed
candidate contained the exact ground-truth local edits. The old hybrid would
fall back to the buggy program and receive zero symbolic credit; partial
fallback preserves these local signals and recovers full edit-level F1.
This is not presented as a passing repair case. It demonstrates that failed
explorer candidates can still contain high-quality localized evidence.
```

---

## 15. Main Claim for Version A

If Version A works, the paper can claim:

> Pruner-certified PatchBanks transform precise code repair from open-ended program generation into discrete patch-level decision making. By preventing the model from outputting full code and using deterministic patch application, Decision-Only Composer avoids autoregressive rewriting and reduces over-editing by construction.

A concise slogan:

```text
Generate broadly. Prune locally. Decide symbolically. Apply deterministically.
```

---

## 16. Immediate Next Steps

1. Modify Composer prompt to output JSON only.
2. Remove `Final Code` from Composer output.
3. Implement deterministic patch application from selected patch IDs.
4. Implement out-of-bank edit rate.
5. Run a small pilot:
   ```text
   50 PDB examples
   K = 5 explorers
   passing candidates only
   ```
6. Compare:
   ```text
   Best Single Explorer
   Best Single Explorer + GenDR
   Free Synthesizer
   Old Pure-Prompt Composer
   Decision-Only Composer
   ```
7. Inspect all cases where:
   ```text
   Decision-Only Composer fails but Free Synthesizer passes
   Free Synthesizer over-edits but Decision-Only Composer stays precise
   Composer selects conflicting patches
   JSON output is invalid
   ```

---

## 17. Notes for GPT-5.3-Codex Implementation

Focus implementation on the clean Version A path:

```text
Explorer outputs -> GenDR -> PatchBank -> Decision JSON -> deterministic patch apply -> evaluation
```

Do not implement beam search or algorithmic Composer yet.

Do not let the Composer output full code.

The first milestone should be:

```text
A working end-to-end pipeline on 50 examples
with valid Composer JSON and deterministic patch application.
```

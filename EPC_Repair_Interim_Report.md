# EPC Precise Repair 阶段性汇报

## 1. 摘要

本阶段工作的核心目标是验证一个假设：精确代码修复不应只被建模为“让模型重新生成一份能过测试的程序”，而应尽量转化为“在多个候选修复中选择、裁剪、组合最小必要 patch”的离散决策问题。

当前实现围绕 Explorer-Pruner-Composer, EPC, 构建了一条更偏符号化的修复流程：多个 explorer 生成候选修复，GenDR 将候选修复裁剪成更小的 patch core，Decision-Only Composer 只输出 patch 选择计划，最终代码由确定性 backend 生成。为了更充分利用 failed explorer 中的局部正确信号，当前还加入了 EPC hybrid partial 选择策略：当没有 passing pruned explorer 或 composer rescue 时，允许回退到最佳 failed explorer 的局部修复。

目前最清晰的结果来自 BigCodeBench t240。EPC hybrid partial 在 BCB PDB-SINGLE-HARD first100 上达到 Unit 0.7400, F1 0.8435，高于 Agentic3+GenDR 的 Unit 0.7300, F1 0.7823；在 BCB PDB-MULTI 100 上达到 Unit 0.9459, F1 0.8862，Unit 与 Agentic3+GenDR rerun 持平，但 symbolic F1 高出约 0.105。这个现象支持当前叙事：Agentic 方法能找到 passing behavior，但更容易 over-edit；EPC 更倾向于保留 ground-truth edit structure。

## 2. 动机

现有代码修复评测常以 unit pass 作为主指标。Unit pass 可以回答“代码是否通过测试”，但不能回答“模型是否修复了真正的 bug”。在 PDB 这类 precise debugging benchmark 中，这个区别很关键：一个方法可能通过重写周边逻辑、增加特判、替换实现方式来通过测试，但它并没有恢复 ground-truth 局部修改。

这带来两个问题。

第一，unit-only evaluation 会奖励 behavior-equivalent rewrite，而不是 precise repair。对真实维护场景而言，越小、越局部、越贴近 root cause 的 patch 越容易审查、回滚和组合。

第二，直接要求 LLM 输出完整 repaired program 会暴露 autoregressive rewriting trap。即使模型已经“知道”正确修复点，在生成完整代码时仍可能顺手重排、重写、增加 guard 或改变无关逻辑，造成 symbolic precision 下降。

因此，本项目的核心动机是把修复任务从开放式代码生成转化为 patch-level selection：让模型负责判断哪些候选 patch 应被采用，而不是让模型自由重写整段程序。

## 3. 当前方法

### 3.1 Explorer Generation

EPC 首先使用多个角色化 explorer 产生候选修复。当前默认使用 5 个 explorer role：

| Explorer | Prompt bias | 目的 |
|---|---|---|
| E1 Minimal | 最小行为修改 | 提供高精度候选 |
| E2 Boundary | 边界条件和 corner cases | 捕获 off-by-one / empty input 等问题 |
| E3 Data-flow | 赋值、状态流、返回值 | 捕获数据流错误 |
| E4 Control-flow | 分支、循环、early return | 捕获控制流错误 |
| E5 Freeform | 不强制 minimal | 提供高召回候选 |

每个 explorer 输出一份完整 repaired program。系统记录 raw diff、unit pass、edit size、explorer role 等信息。

### 3.2 GenDR Pruning

对 passing explorer candidate，GenDR 会尝试逐块删除 candidate 中的修改，只保留删除后会导致测试失败的必要编辑块。当前支持 hunk 和 line 两种粒度；实验中 multi setting 主要使用 hunk 粒度。

GenDR 的作用是将“可能很大的候选修复”压缩为更小的 patch core，使后续 composer 面对的不是完整代码，而是 pruner-certified patch bank。

### 3.3 Decision-Only Composer

Decision-Only Composer 不输出 final code。它只接收 PatchBank 和 patch clusters，并输出结构化 JSON plan，包括：

| 字段 | 含义 |
|---|---|
| selected_patch_ids | 选择哪些 patch |
| rejected_patch_ids | 拒绝哪些 patch |
| conflict_decisions | 冲突 patch 的选择依据 |
| local_resolutions | 必要时仅在冲突局部生成 resolution |
| confidence / rationale | 决策依据 |

最终 repaired code 由 deterministic backend 根据 plan 应用 patch。这样可以把模型能力限制在“选择和冲突决策”上，避免 full-code generation 引入 out-of-bank edits。

### 3.4 EPC Hybrid Partial

当前效果最好的评估策略是 EPC hybrid partial。它不是让 composer 重写代码，而是在 EPC 产物上做确定性选择：

| 优先级 | 选择来源 | 说明 |
|---:|---|---|
| 1 | best passing pruned explorer | 选择 edit size 最小的 passing GenDR-pruned explorer |
| 2 | composer rescue | 如果 composer final_pass=True，则使用 composer 组合结果 |
| 3 | best partial failed explorer | 如果没有 passing candidate，则用 label-free 排序选择 failed explorer：优先考虑测试通过比例、raw edit size、failed-candidate 间的 edit consensus density、retry round 和 explorer id |
| 4 | buggy fallback | 如果没有可用局部信号，则回退 buggy code |

Partial fallback 的直觉是：failed candidate 不一定整体可用，但其局部 diff 仍可能包含真实修复信号。当前选择逻辑不读取 `gt_diff`、symbolic precision/recall/F1 或 ground-truth edit；PDB 的 symbolic metrics 只在最终评测阶段用于衡量这种“局部修复信号”是否对齐 ground truth。因此，当前 partial fallback 不是基于标签挑选候选，但它仍然是 test-oracle-aided：候选生成和 GenDR/selection 会使用可用 unit tests 的 pass/fail 信号。

### 3.5 Agentic3+GenDR Baseline

Agentic3 baseline 使用多轮 bug correction。每轮失败样本会带上上一轮 failed attempt 和可选错误反馈进入下一轮。GenDR final 则对最终 round 的修复结果进行离线 diff pruning。它代表更传统的“生成完整修复 + 后处理裁剪”路线。

与 EPC 相比，Agentic3+GenDR 更依赖 full-code generation，因此更容易出现通过测试但 symbolic mismatch 的情况。

## 4. 实现状态

当前代码中已经实现以下组件：

| 模块 | 作用 |
|---|---|
| `src/epc_decision_only.py` | EPC 主流程：explorer generation, PatchBank, decision-only composer, deterministic apply, metadata |
| `src/gendr.py` | GenDR block construction, unit-test oracle, hunk/line pruning |
| `scripts/epc_hybrid_gated_eval.py` | EPC hybrid / hybrid partial 选择和评估 |
| `src/bug_correct.py` | Agentic multi-round baseline，支持 `--enable_gendr` |
| `src/gendr_refine.py` | 对已有 debug result 离线运行 GenDR final |
| `dataset/livecodebench/install/lcb_runner/benchmarks/code_generation.py` | LCB 本地多 jsonl cache loader |

LCB loader 也做了一个实用修复：现在 `LCB_CODEGEN_RELEASE_FILE` 可以指向单个 jsonl，也可以指向一个目录。若指向 `/root/rivermind-data/livecodebench`，evaluator 会扫描目录下 6 个 `test*.jsonl`，避免每次 oracle 调用都回退到 HuggingFace streaming。

## 5. 部分实验结果

### 5.1 BigCodeBench PDB-SINGLE-HARD first100, t240

| Method | n | Unit | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| EPC hybrid | 100 | 0.7400 | 0.6589 | 0.6742 | 0.6528 |
| EPC hybrid partial | 100 | 0.7400 | 0.8486 | 0.8775 | 0.8435 |
| Agentic3+GenDR final | 100 | 0.7300 | 0.7767 | 0.8283 | 0.7823 |

主要观察：EPC hybrid partial 的 Unit 与 EPC hybrid 相同，但 symbolic F1 大幅提升。这说明 partial fallback 不一定提升 unit pass，却能恢复更多 ground-truth edit structure。相对 Agentic3+GenDR，EPC hybrid partial 的 Unit 略高，F1 高约 0.061。

EPC hybrid partial 的来源分解：

| Source | Count |
|---|---:|
| best passing pruned explorer | 68 |
| composer rescue | 6 |
| best partial failed explorer | 26 |
| buggy fallback | 0 |

这表明 single-hard setting 中有相当多样本没有完整 passing explorer，但 failed explorer 仍携带可用的局部 patch 信号。

### 5.2 BigCodeBench PDB-MULTI 100, t240

| Method | n | Unit | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| EPC hybrid | 37 | 0.7027 | 0.8730 | 0.9054 | 0.8862 |
| EPC hybrid partial | 37 | 0.9459 | 0.8730 | 0.9054 | 0.8862 |
| Agentic3+GenDR rerun | 37 | 0.9459 | 0.7572 | 0.8649 | 0.7813 |

主要观察：EPC hybrid partial 和 Agentic3+GenDR rerun 的 Unit 都是 0.9459，但 EPC symbolic F1 更高。这是目前最支持论文叙事的一组结果：在 functional correctness 持平时，EPC 保留了更精确的 edit-level repair。

EPC hybrid partial 的来源分解：

| Source | Count |
|---|---:|
| best passing pruned explorer | 34 |
| composer rescue | 2 |
| best partial failed explorer | 1 |
| buggy fallback | 0 |

Multi setting 中 passing pruned explorer 已经覆盖大部分样本。EPC 的优势主要来自选择较小、较局部的 pruned candidate，而不是依赖大规模 fallback。

### 5.3 LiveCodeBench 当前已有结果

LCB-multi_100 的 EPC hybrid partial 和 Agentic3+GenDR final 正在补跑前置依赖。当前已有结果如下：

| Setting | Method | n | Unit | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| LCB multi_100 | baseline | 100 | 0.0600 | 0.4086 | 0.9600 | 0.4891 |
| LCB multi_100 | baseline+GenDR | 100 | 0.0600 | 0.4223 | 0.9550 | 0.5033 |
| LCB single first100 | EPC hybrid t60 | 100 | 0.9400 | 0.7866 | 0.8200 | 0.7991 |
| LCB single first100 | Agentic3+GenDR | 100 | 0.8600 | 0.7591 | 0.7908 | 0.7662 |

这些结果只作为阶段性参考。LCB multi_100 的正式 EPC-vs-Agentic 对比需要等待当前本地 cache 修复后的完整补跑结果。

## 6. Case Study

### Case A: 同样过测试，但 EPC 精确修复 root cause

Task: `BigCodeBench/0_2`

Setting: BCB PDB-SINGLE-HARD first100

| Method | Unit | Precision | Recall | F1 | Predicted edits |
|---|---:|---:|---:|---:|---:|
| EPC hybrid partial | 1 | 1.0000 | 1.0000 | 1.0000 | 1 |
| Agentic3+GenDR | 1 | 0.0000 | 0.0000 | 0.0000 | 5 |

Ground truth 是将 iterator 显式 materialize：

```diff
-    permutations = itertools.permutations(numbers)
+    permutations = list(itertools.permutations(numbers))
```

EPC hybrid partial 预测完全相同的单行修改。Agentic3+GenDR 也通过 unit tests，但它通过增加 counter 逻辑绕开了 `len(permutations)` 的问题，因此 symbolic F1 为 0。

这个 case 说明 unit pass 无法区分 root-cause repair 和 behavior workaround。EPC 的 patch-level selection 更容易保留原始局部 bug 位置。

### Case B: Multi-edit 中 Agentic 过度重写

Task: `BigCodeBench/1028_2`

Setting: BCB PDB-MULTI 100

| Method | Unit | Precision | Recall | F1 | Predicted edits |
|---|---:|---:|---:|---:|---:|
| EPC hybrid partial | 1 | 1.0000 | 1.0000 | 1.0000 | 2 |
| Agentic3+GenDR | 1 | 0.0000 | 0.0000 | 0.0000 | 19 |

Ground truth 只需要移除 deprecated `platform.linux_distribution()` 分支，并统一使用 `top`：

```diff
-                    dist_name, _, _ = platform.linux_distribution()
-                    command = ["vmstat", "1", "1"] if dist_name == "Ubuntu" else ["top", "-b", "-n1"]
+                    # Unix/Linux command for CPU usage
+                    command = ["top", "-b", "-n1"]
```

EPC hybrid partial 选择了紧凑 patch，修复集中在 command selection。Agentic3+GenDR 虽然过测试，但重写了 CPU usage parsing 周边逻辑，产生 19 个 predicted edits。

这个 case 支持 EPC 的核心优势：不是更会“写一段能跑的代码”，而是更会“选择必要的局部修复”。

### Case C: Failed explorer 中仍有可用局部证据

Task: `BigCodeBench/14_40`

Setting: BCB PDB-SINGLE-HARD first100

| Method | Unit | Precision | Recall | F1 | Predicted edits |
|---|---:|---:|---:|---:|---:|
| EPC hybrid partial | 0 | 1.0000 | 1.0000 | 1.0000 | 3 |
| Agentic3+GenDR | 0 | 0.0000 | 0.0000 | 0.0000 | 6 |

EPC hybrid partial 预测了和 ground truth 完全一致的三个局部 edits，包括使用原始 project directory、删除错误的 dirname 操作，以及将 archive format 从 `gztar` 改为 `zip`。

需要特别说明：这个 case 不是 functional success case。它的 Unit 为 0，是因为 ground-truth solution 本身在当前 BigCodeBench evaluator 中也会因为环境文件缺失而失败。因此它只说明 partial fallback 可以恢复精确 symbolic edits，不能被表述为 passing repair。

## 7. 当前结论

阶段性结论可以概括为三点。

第一，EPC hybrid partial 在 BigCodeBench 上已经显示出比 Agentic3+GenDR 更好的 edit-level precision，尤其是在 Unit 持平时 symbolic F1 更高。

第二，partial fallback 是一个重要补充。它说明 failed explorer 不能简单丢弃，因为其中可能包含 ground-truth 局部 patch。对于 PDB 这类 precise repair 任务，这种局部证据有独立价值。

第三，Decision-Only Composer 的研究方向仍然成立，但当前最强结果主要来自 pruned explorer selection 与 partial fallback。后续需要进一步隔离 composer 本身的贡献，例如统计 composer rescue 的成功率、out-of-bank edit rate、以及 conflict resolution validity。

## 8. 下一步计划

近期最值得做的实验是补齐 LCB-multi_100 的正式对比：

| Dataset | Method | 状态 |
|---|---|---|
| LCB multi_100 | EPC hybrid partial | 本地 multi-jsonl cache 已修复，待完整跑完 |
| LCB multi_100 | Agentic3+GenDR final | 本地 multi-jsonl cache 已修复，待完整跑完 |
| BCB single-hard100 | EPC vs Agentic case study | 已有 2-3 个强 case |
| BCB multi100 | EPC vs Agentic case study | 已有强 over-edit case |

同时建议增加几个诊断表：

| 诊断项 | 目的 |
|---|---|
| Source breakdown | 区分 best explorer, composer rescue, partial fallback 的贡献 |
| Unit-pass but F1-low cases | 定位 Agentic over-edit 或 behavior workaround |
| F1-high but Unit-fail cases | 排查 evaluator/environment issue 与 symbolic-only recovery |
| Out-of-bank edit rate | 衡量 Decision-Only Composer 是否真的避免自由重写 |

## 9. 可用于汇报的一句话版本

EPC 的核心不是让模型更努力地重写代码，而是先生成多样候选、用测试裁剪成局部 patch，再把最终修复限制为离散 patch selection；初步结果显示，在 BigCodeBench 上这种方式能在保持或匹配 unit pass 的同时显著减少 over-edit，更接近 ground-truth precise repair。

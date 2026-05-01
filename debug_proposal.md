# GenDR: Generate-Diff-Revert — 基于测试引导的逐块撤销实现精准调试

## 1. 问题动机 (Motivation)

大语言模型（LLM）在代码修复中表现出色，但存在一个被广泛观察到的严重问题：**over-editing（过度编辑）**。当给定一段含有bug的代码时，前沿模型（如 GPT-5.1-Codex、DeepSeek-V3.2-Thinking）倾向于重写大量代码甚至整个程序来"修复"问题，而非精确定位并最小化修改 [1]。这一行为在 Precise Debugging Benchmark (PDB) 上被量化：尽管模型的单元测试通过率超过 76%，但 edit-level precision 仅为 39%–45% [1]。PRepair 进一步证实，随着 GRPO 训练的推进，模型的修复正确率提升，但 edit cost 甚至超过 0.6，表明模型并未学会定位 bug，而是通过大规模修改"碰运气" [2]。

现有解决方案分为两类，各有不足：

- **训练时方法**（如 PRepair 的 EA-GRPO [2]）：通过 edit-aware reward 在训练中引导模型减少 over-editing。但这需要额外的训练数据、RL 训练管线和计算资源，且只适用于可以微调的开源模型，无法应用于闭源 API 模型（GPT、Claude 等）。
- **Prompting 方法**（如 minimal debugging prompt [1]）：仅通过提示要求模型"尽量少改"。PDB 的实验表明，这有一定效果，但 precision 仍远不够理想——最好的模型也仅达 72%。
- **迭代/Agentic 方法** [1]：通过多轮交互或提供测试反馈来让模型改进。但 PDB 的实验明确显示：迭代和 agentic 策略可以提升 unit-test 分数和 recall，但 **precision 几乎不变甚至下降**。更多的反馈往往导致更大范围的重写。

**核心洞察**：既然 LLM 天然倾向于"多写"而非"精确写"，与其在生成阶段约束模型（这在训练和推理中都很难做到），不如接受模型的高 Recall 输出，然后在后处理阶段通过测试引导的系统化裁剪来恢复 Precision。这正是本研究的出发点。

## 2. 核心想法 (Core Idea)

我们提出 **GenDR (Generate-Diff-Revert)**，一个模型无关的、推理时的后处理框架，将精准调试分解为三个阶段：

**Phase 1: Generative Fixing（生成修复）** — 让 LLM 正常生成修复代码。此阶段不对模型施加任何"最小编辑"约束，允许模型充分发挥其代码生成能力，产出高 Recall 但低 Precision 的修复方案。这是一个"发散"阶段。

**Phase 2: Trace-Guided Diffing（差异提取）** — 对生成的修复代码与原始 buggy 代码进行结构化 Diff，提取出模型实际修改的所有代码块（hunks/blocks）。每个 block 被视为一个独立的"修改单元"。

**Phase 3: Test-Guided Revert（测试引导撤销）** — 这是本方法的核心创新。对每个 diff block，尝试将其撤销（revert）回原始 buggy 代码的对应片段，然后运行单元测试：

- 如果测试**依然通过** → 该 block 是 over-editing，采纳撤销（成功裁剪）。
- 如果测试**失败** → 该 block 是 essential edit，保留修改。

最终输出仅包含那些被测试验证为"必要"的修改，从而在保持 Recall 的同时大幅提升 Precision。

**关键优势**：

1. **模型无关（Model-agnostic）**：适用于任何 LLM，包括闭源 API 模型，无需训练或微调。
2. **与 PDB 的 `essentialU` 概念的理论对齐**：PDB 论文定义了 `essentialU` 函数来寻找最小必要编辑子集 [1]，但仅将其用于评估指标计算。我们将其核心思想提升为一种实际的推理时方法。
3. **可组合性**：GenDR 可以叠加在任何现有方法之上——无论是 prompt engineering、iterative debugging、还是 EA-GRPO 训练的模型，都可以在最后加一层 GenDR 来进一步提升 precision。

## 3. 方法概述 (Method Overview)

### 3.1 整体框架

```
Input: buggy code C_b, task description x, unit tests U
Output: precise repair C_precise

┌──────────────────────────────────────────────────────┐
│ Phase 1: Generative Fixing                           │
│   C_fix = LLM(C_b, x)                              │
│   [可选] 验证 F_U(C_fix) = 1，否则 fallback        │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│ Phase 2: Trace-Guided Diffing                        │
│   D = StructuredDiff(C_b, C_fix)                    │
│   blocks = {b_1, b_2, ..., b_n}  // 提取修改块      │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│ Phase 3: Test-Guided Revert (逐块撤销)               │
│   C_current = C_fix                                  │
│   for each block b_i in blocks:                      │
│     C_candidate = Revert(C_current, b_i)            │
│     if F_U(C_candidate) == 1:                       │
│       C_current = C_candidate  // 裁剪成功          │
│     else:                                            │
│       keep b_i in C_current     // 保留必要修改      │
│   C_precise = C_current                              │
└──────────────────────────────────────────────────────┘
```

### 3.2 Phase 1: Generative Fixing 的设计选择

- **不施加 minimal-edit 约束**：与直觉相反，我们在生成阶段刻意不要求模型 "make minimal changes"。PDB 论文已经表明 [1]，这种约束的效果有限且不稳定。相反，我们希望模型自由发挥以最大化 Recall（修复尽可能多的 bug）。
- **Multi-sample 策略**：可以采样多个修复候选（temperature > 0），选择通过测试的那个。如果多个候选都通过测试，可选择 diff 最小的那个作为 Phase 2 的输入。
- **可选的 Thinking model**：对于复杂的 multi-bug 场景，可以使用 thinking model（如 DeepSeek-V3.2-Thinking）以提升初始修复的 Recall。

### 3.3 Phase 2: Structured Diffing

- **行级 Diff**：使用标准 unified diff 算法（如 Python `difflib`），提取所有修改的行。
- **Hunk 分组**：将连续修改的行聚合为 hunk（代码块）。相邻的修改行（间隔 ≤ context_gap 行）合并为同一 hunk。
- **Block 粒度选择**：
    - **Fine-grained（行级）**：每个修改行独立处理。最精细但测试执行次数最多。
    - **Hunk-level（块级）**：每个 hunk 整体处理。平衡精度与效率。
    - **Adaptive（自适应）**：先尝试 hunk-level revert，如果失败则拆分为行级逐行测试。

### 3.4 Phase 3: Test-Guided Revert 的核心算法

#### 3.4.1 基础版本：Sequential Revert（顺序撤销）

最简单的策略：依次尝试撤销每个 block。

```python
def sequential_revert(C_fix, C_b, blocks, test_suite):
    C_current = C_fix
    essential_blocks = []
    for block in blocks:
        C_candidate = revert_block(C_current, C_b, block)
        if run_tests(C_candidate, test_suite):
            C_current = C_candidate  # 撤销成功
        else:
            essential_blocks.append(block)  # 保留
    return C_current, essential_blocks
```

**问题**：撤销顺序可能影响结果。如果 block A 和 block B 之间存在依赖关系（撤销 A 后 B 变得不必要，或反过来），不同顺序可能产生不同的最终结果。

#### 3.4.2 改进版本：Independent Revert（独立撤销）

对每个 block 独立判断其必要性（以完整的 C_fix 为基准，而非逐步更新的 C_current）：

```python
def independent_revert(C_fix, C_b, blocks, test_suite):
    essential_mask = []
    for block in blocks:
        C_candidate = revert_single_block(C_fix, C_b, block)
        essential_mask.append(not run_tests(C_candidate, test_suite))
    # 仅保留 essential blocks
    C_precise = apply_essential_blocks(C_b, C_fix, blocks, essential_mask)
    # 最终验证
    if run_tests(C_precise, test_suite):
        return C_precise
    else:
        return C_fix  # fallback
```

**优点**：每个 block 的判断不受其他 block 撤销与否的影响，结果确定性更强。 **缺点**：如果多个 block 之间存在联合依赖（单独撤销任何一个都通过测试，但同时撤销就失败），可能裁剪过多导致最终验证失败。

#### 3.4.3 高级版本：Hierarchical Revert（层级撤销）

结合两种策略的优势：

```python
def hierarchical_revert(C_fix, C_b, blocks, test_suite):
    # Step 1: Independent pass — 识别明确 essential 的 blocks
    definitely_essential = []
    maybe_removable = []
    for block in blocks:
        C_candidate = revert_single_block(C_fix, C_b, block)
        if run_tests(C_candidate, test_suite):
            maybe_removable.append(block)
        else:
            definitely_essential.append(block)

    # Step 2: 对 maybe_removable 进行贪心 sequential pass
    C_current = apply_only_blocks(C_b, C_fix, definitely_essential)
    # 先添加所有 maybe_removable，然后逐个尝试移除
    C_current = apply_blocks(C_current, C_fix, maybe_removable)
    for block in maybe_removable:
        C_candidate = revert_block(C_current, C_b, block)
        if run_tests(C_candidate, test_suite):
            C_current = C_candidate

    return C_current
```

#### 3.4.4 最优版本：Binary-Search Revert（二分搜索撤销）

借鉴经典的 Delta Debugging 思想，使用二分搜索来减少测试执行次数：

```python
def binary_search_revert(C_fix, C_b, blocks, test_suite):
    """
    尝试一次性撤销一半 blocks；
    如果测试通过 → 这一半都是 over-editing，递归处理另一半
    如果测试失败 → 拆分为更小的组，递归
    """
    if len(blocks) == 1:
        C_candidate = revert_block(C_fix, C_b, blocks[0])
        if run_tests(C_candidate, test_suite):
            return [], [blocks[0]]  # removable
        else:
            return [blocks[0]], []  # essential
    
    mid = len(blocks) // 2
    left, right = blocks[:mid], blocks[mid:]
    
    # 尝试撤销左半部分
    C_revert_left = revert_blocks(C_fix, C_b, left)
    if run_tests(C_revert_left, test_suite):
        # 左半全部可裁剪，递归处理右半
        essential_right, removable_right = binary_search_revert(
            C_revert_left, C_b, right, test_suite)
        return essential_right, left + removable_right
    
    # 类似处理右半...（省略完整逻辑）
    # 如果两半都不能整体撤销，分别递归
```

**复杂度**：最优情况 O(log n)（大部分 blocks 可裁剪），最差情况 O(n log n)。相比 Sequential 的 O(n)，在 over-editing 严重（大部分 blocks 可裁剪）时效率更高。

### 3.5 可选增强模块

#### 3.5.1 Multi-Candidate Ensemble

从 Phase 1 采样 K 个修复候选 {C_fix^1, ..., C_fix^K}，对每个独立运行 Phase 2-3，得到 K 个精简修复。选择修改量最小且通过测试的那个。

#### 3.5.2 Semantic-Aware Block Merging

在 Phase 2 的 diff 提取中，利用 AST（抽象语法树）信息将语义相关的修改合并为一个 block。例如，修改一个函数签名和对应的函数体通常应作为一个整体来考虑。

#### 3.5.3 Coverage-Guided Prioritization

利用测试覆盖率信息对 blocks 排序：覆盖率变化小的 block 优先尝试撤销（更可能是 over-editing）。

## 4. 理论直觉 / 为什么这个方法应该有效 (Why It Should Work)

### 4.1 与 PDB essentialU 的理论联系

PDB 论文 [1] 在其评估框架中定义了 `essentialU` 函数，用于在模型的 predicted edits 中搜索通过测试所需的最小编辑子集。公式为：

$$ (|\hat{E}*i|)*\epsilon = \min(|\text{essential}_\mathcal{U}(\text{map}(E_i))|, |E_i| + \epsilon) $$

我们的 Phase 3 本质上就是在推理时执行 `essentialU` 的一个近似版本。区别在于：

- PDB 的 `essentialU` 是在**已知 ground-truth bug 位置**的前提下，对每个 bug 的 predicted edits 分别搜索最小子集——这只能用于评估。
- 我们的方法在**不知道 ground-truth**的前提下，对模型的全部 diff blocks 进行搜索——这可以用于实际推理。

### 4.2 为什么 over-editing 大部分是可逆的

PDB 论文 [1] 的一个关键发现支持我们的方法：模型的 over-editing 主要是**语义无关的重写**（如变量重命名、代码重组、添加不必要的注释等），而非引入新的功能或修改已有的正确逻辑。这意味着：

- 大多数 over-editing blocks 在被撤销后，程序仍能通过测试。
- 真正的 bug fix 通常集中在少数几个 blocks 中。

这与 PDB 的数据一致：模型 edit precision 平均在 40%-70%，意味着 30%-60% 的编辑是不必要的，可以被安全撤销。

### 4.3 为什么后处理优于约束生成

训练时或提示时的约束（如 EA-GRPO、minimal-edit prompt）试图在**生成过程中**同时优化正确性和精确性。但这两个目标之间存在张力 [2]：

- 减少编辑量可能降低 bug 被修复的概率（损害 Recall）。
- EA-GRPO 需要设置 accuracy threshold α 来平衡——只有当组内正确率 ≥ α 时才施加 edit penalty [2]。

GenDR 将两个目标**解耦**：Phase 1 专注于正确性（最大化 Recall），Phase 3 专注于精确性（最大化 Precision），不存在目标冲突。

### 4.4 复杂度分析

设修复后的 diff 包含 n 个 blocks：

- Sequential Revert: n 次测试执行
- Independent Revert: n 次测试执行 + 1 次最终验证
- Binary-Search Revert: O(k log(n/k)) 次测试执行（k 为 essential blocks 数量）
- 对于典型场景（n=10-20 blocks，k=2-4 essential），测试执行次数在 10-30 之间，每次测试执行时间在毫秒到秒级，总时间开销可接受。

## 5. 与用户背景的匹配度 (Feasibility for You)

### 技能匹配

- **核心实现**：Phase 1 仅需调用 LLM API；Phase 2 使用标准 diff 工具（Python difflib / unified diff）；Phase 3 是简单的循环 + 测试执行。整体实现复杂度低。
- **实验基础设施**：PDB 已开源代码和数据集 [1]，可以直接复用其评估管线（precision、recall、unit-test score）。PRepair 也公开了其 benchmark [2]。

### 计算资源需求

- **LLM 推理**：Phase 1 的生成与现有方法相同，不增加成本。Phase 3 的测试执行不需要 GPU。
- **测试执行**：主要额外成本是 Phase 3 的多次测试运行。对于 PDB-SINGLE-HARD（Python 程序），每次测试在 CPU 上通常 < 1 秒。即使 n=20 blocks，总额外开销也仅 ~20 秒。
- **无需训练**：与 PRepair 不同，不需要 RL 训练的 GPU 资源。

### 实现时间估计

- Phase 1-2 实现：1-2 天
- Phase 3 基础版本：2-3 天
- Phase 3 高级版本（Binary-Search、Hierarchical）：3-5 天
- 实验复现与评估：1-2 周
- 总计：约 3-4 周

## 6. 预期贡献 (Expected Contributions)

1. **提出 GenDR 框架**：第一个系统化的、模型无关的推理时 precise debugging 后处理方法。将 PDB 的 `essentialU` 评估概念提升为实际可用的推理时方法。
2. **Precision 大幅提升**：预期在 PDB-SINGLE-HARD 上将 GPT-5.1-Codex 的 precision 从 39.7% 提升至 60%+（具体取决于测试引导裁剪的效果），同时保持 Recall 不变或仅略微下降。
3. **与训练方法的互补性验证**：展示 GenDR 可以叠加在 PRepair 训练后的模型上，进一步提升 precision（即 EA-GRPO + GenDR > 任一单独使用），证明推理时方法与训练时方法的正交性。
4. **多种 Revert 策略的系统对比**：提供 Sequential / Independent / Hierarchical / Binary-Search 四种策略在 precision-recall-efficiency 三维空间中的完整 trade-off 分析。
5. **实际推理开销分析**：证明 GenDR 的额外时间成本（测试执行）在实际场景中可接受（秒级），使其成为一种"free lunch"式的精度提升。

## 7. 实验计划 (Suggested Experiments)

### 7.1 数据集

|     数据集      |               来源               |     规模      |          Bug 类型          |       用途        |
| :-------------: | :------------------------------: | :-----------: | :------------------------: | :---------------: |
| PDB-SINGLE-HARD | BigCodeBench + LiveCodeBench [1] |     5,751     |    单行 bug，1-4个/程序    |     主要评估      |
|    PDB-MULTI    | BigCodeBench + LiveCodeBench [1] |      256      | 多行 block bug，1-3个/程序 |     泛化验证      |
|  HumanEval-Fix  |          HumanEval [2]           |      164      |      手工注入逻辑 bug      | 跨 benchmark 验证 |
|   DebugBench    |       Real-world bugs [1]        | 40 (filtered) |          真实 bug          |   真实场景验证    |

### 7.2 Baselines

|        方法         |    类别     |                  描述                  |
| :-----------------: | :---------: | :------------------------------------: |
|     Vanilla LLM     | Single-shot |    标准单次生成（freeform prompt）     |
| Minimal-edit Prompt |  Prompting  |          明确要求最小编辑 [1]          |
| Iterative Debugging | Multi-turn  |            3 轮迭代修复 [1]            |
|  Agentic Debugging  |    Agent    |       含测试反馈的 3 轮修复 [1]        |
|  PRepair (EA-GRPO)  |  Training   | Edit-aware RL 训练 [2]（在开源模型上） |
|     Claude-Code     |    Agent    |        端到端 agentic 系统 [1]         |

### 7.3 评估指标

- **Edit-level Precision** (ε-relaxed, ε=2 for single, ε=1 for multi) [1]
- **Bug-level Recall** [1]
- **Unit-test Pass@1** [1]
- **fix_p@1** (p=1, 1.5, 2) [2]
- **测试执行次数**：衡量 Phase 3 的计算开销
- **Wall-clock 时间**：端到端总时间（含 LLM 推理 + 测试执行）

### 7.4 实验设计

#### 实验 1：主要效果验证（Table 1 of our paper）

- **目标**：验证 GenDR 在不同模型上的 precision 提升。
- **设置**：在 PDB-SINGLE-HARD 上，对 PDB 论文中的 9 个模型分别应用 GenDR（使用 Sequential Revert），对比 vanilla、minimal-edit prompt、iterative、agentic baselines。
- **预期结果**：所有模型的 precision 显著提升；Recall 保持或仅略微下降。"Pass-oriented" 模型（GPT-5.1-Codex、DeepSeek）的提升最为显著。

#### 实验 2：Revert 策略对比（Table 2 / Figure 2）

- **目标**：对比四种 Revert 策略。
- **设置**：在 PDB-SINGLE-HARD 上，固定使用 GPT-5.1-Codex 和 Claude-Sonnet-4.5 两个代表性模型，对比 Sequential / Independent / Hierarchical / Binary-Search。
- **指标**：Precision、Recall、测试执行次数、Wall-clock 时间。
- **预期结果**：Hierarchical 在 precision-efficiency trade-off 上最优；Binary-Search 在 over-editing 严重的模型上效率最高。

#### 实验 3：Block 粒度消融（Figure 3）

- **目标**：分析 Phase 2 的 block 粒度对效果的影响。
- **设置**：对比 line-level / hunk-level / adaptive 三种粒度。
- **预期结果**：Line-level 精度最高但开销最大；Adaptive 在大多数场景下最优。

#### 实验 4：与 PRepair 的互补性（Table 3）

- **目标**：验证 GenDR 与训练时方法的互补性。
- **设置**：在 HumanEval-Fix 上，对 Qwen2.5-Coder-7B 模型对比：
    - Base model + GenDR
    - EA-GRPO model (PRepair)
    - EA-GRPO model + GenDR
- **指标**：fix_1@1, fix_1.5@1, pass@1。
- **预期结果**：EA-GRPO + GenDR 取得最高 fix_p@1，证明两种方法的正交互补性。

#### 实验 5：Multi-bug 场景分析（Figure 4）

- **目标**：分析 GenDR 在不同 bug 数量下的表现。
- **设置**：在 PDB-SINGLE-HARD 上，按 bug count (k=1,2,3,4) 分组，对比有/无 GenDR 的 precision 和 recall。
- **预期结果**：GenDR 在 k 较大时提升更显著（因为更多 bug 意味着更多 over-editing）。

#### 实验 6：PDB-MULTI 和 DebugBench 泛化（Table 4）

- **目标**：验证 GenDR 在多行 bug 和真实 bug 场景下的有效性。
- **设置**：在 PDB-MULTI 和 DebugBench 上评估 GenDR。
- **预期结果**：GenDR 在多行 bug 场景下同样有效，但提升幅度可能略小（多行 bug 本身更复杂，over-editing 的可逆性可能降低）。

#### 实验 7：Multi-Candidate Ensemble（Table 5）

- **目标**：验证多候选采样与 GenDR 的结合效果。
- **设置**：采样 K=1,3,5,10 个候选，对每个运行 GenDR，选最精简的。
- **预期结果**：K=5 是 precision 提升和推理成本的最佳平衡点。

#### 实验 8：时间开销分析（Figure 5）

- **目标**：证明 GenDR 的时间开销可接受。
- **设置**：统计 Phase 3 的测试执行次数、平均/中位/P95 执行时间。按 diff block 数量分组展示。
- **预期结果**：中位时间 < 5 秒，P95 < 30 秒。

### 7.5 验证核心 claim 的最小实验

如果资源有限，最小验证实验为：

1. 选择 GPT-5.1-Codex（precision 最低的模型）和 Claude-Sonnet-4.5（precision 最高的模型）。
2. 在 PDB-SINGLE-HARD 的 500 个样本子集上运行。
3. 实现 Sequential Revert（最简单的版本）。
4. 对比 Vanilla vs. GenDR 的 Precision、Recall、Unit-test score。

预计 1-2 天即可完成，可快速验证核心假设。

## 8. 相关工作定位 (Related Work Positioning)

### 与 PDB [1] 的关系

PDB 提出了 precise debugging 的评估框架和 `essentialU` 概念，但其结论是"iterative 和 agentic 策略无法改善 precision"。我们的工作直接回应这一挑战：**不是所有推理时策略都无效，关键在于策略的设计**。GenDR 将 `essentialU` 从评估指标转化为推理时方法，提供了 PDB 论文所呼吁的"rethink post-training pipelines"的一种替代路径——无需改变训练，仅改变推理。

### 与 PRepair [2] 的关系

PRepair 通过 EA-GRPO 在训练阶段解决 over-editing 问题。GenDR 在推理阶段解决同一问题。两者互补而非竞争：PRepair 减少模型生成 over-editing 的倾向（减少 Phase 3 需要裁剪的工作量），GenDR 在生成之后进一步裁剪残余的 over-editing。

### 与 Delta Debugging 的关系

经典的 Delta Debugging（Zeller, 1999）用于最小化导致失败的输入。我们借用了其二分搜索的思想，但应用场景不同：Delta Debugging 是"最小化导致失败的修改"，我们是"最小化使修复生效的修改"——目标函数相反。

### 与 REFINE [3] 的关系

REFINE 提出了 context-aware patch refinement，通过 LLM 对 patch 进行后处理。但 REFINE 仍依赖 LLM 来"决定"哪些修改是必要的——模型本身就倾向于 over-edit，用同一个模型来裁剪不一定可靠。GenDR 使用**测试**而非**模型**来判断必要性，提供了更可靠的裁剪信号。

## 9. 风险与开放问题 (Risks & Open Questions)

### 风险 1：Block 间依赖导致过度裁剪

如果两个 over-editing blocks 之间存在意外的正向依赖（A 和 B 单独撤销都通过测试，但同时撤销导致失败），Sequential Revert 和 Independent Revert 可能产生不同结果。

- **缓解**：Hierarchical Revert 通过最终验证步骤捕获此类情况；可加入"最终完整测试 + 回退"保障。

### 风险 2：测试套件不够充分

如果单元测试覆盖率低，某些 essential edit 可能被错误地判定为 over-editing（测试没覆盖到相关功能）。

- **缓解**：PDB 的 benchmark 设计保证了测试与 bug 的对应关系；在实际应用中，可结合模型的 confidence 或 LLM 判断作为辅助信号。

### 风险 3：Recall 下降

理论上 GenDR 不应降低 Recall（Phase 3 只撤销通过测试的修改），但如果 block 分割不当（将一个 essential edit 拆分到两个 blocks），可能导致单独撤销时通过测试但整体失去修复效果。

- **缓解**：使用 AST-aware block splitting；最终完整测试验证。

### 风险 4：时间开销在大型项目中不可接受

对于大型代码库（测试运行时间长），Phase 3 的多次测试执行可能太慢。

- **缓解**：Binary-Search 减少执行次数；可并行执行独立的 revert 测试；可结合 test selection 只运行相关测试。

### 开放问题

1. **最优 block 粒度**如何自动选择？是否可以根据 diff 的结构特征（如 AST 层级、变量依赖图）自适应决定？
2. **能否利用 LLM 的 attention 或 logits** 来为 block 排序提供先验（哪些 block 更可能是 over-editing）？
3. **GenDR 能否扩展到非 Python 语言？** 对于编译型语言（如 Java、C++），每次测试需要重新编译，时间开销更大。
4. **与 speculative editing [2] 的协同**：GenDR 产出的 precise patch 可以反过来提升 speculative decoding 的 acceptance rate，形成正循环。

## 10. 参考文献 (References)

- [1] Wang Bill Zhu, Miaosen Chai, Shangshang Wang, Yejia Liu, Song Bian, Honghua Dong, Willie Neiswanger, Robin Jia, "Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?", 2026. https://cdn.atominnolab.com/wisdoc/markdowns/20260424-6b7c2f09-3575-4430-9ffe-e317eb613110.md
- [2] Changxin Ke, Rui Zhang, Jiaming Guo, Yuanbo Wen, Li Ding, Shuo Wang, Xuyuan Zhu, Xiong Peng, Di Huang, Zidong Du, Xing Hu, Qi Guo, Yunji Chen, "QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization", arXiv 2604.05963, 2026. https://arxiv.org/abs/2604.05963v1
- [3] "REFINE: Enhancing Program Repair Agents through Context-Aware Patch Refinement", 2025.
- [4] Andreas Zeller, "Yesterday, my program worked. Today, it does not. Why?", ESEC/FSE, 1999.
- [5] Samuel Benton, Mengshi Zhang, Xia Li, Lingming Zhang, "Self-boosted automated program repair", ICSE, 2023.
- [6] Chunqiu Steven Xia, Yuxiang Wei, Yifeng Ding, Lingming Zhang, "Automated repair of programs from large language models", ICSE, 2023.
- [7] Chunqiu Steven Xia, Matteo Paltenghi, Jia Le Tian, Michael Pradel, Lingming Zhang, "Revisiting unnaturalness for automated program repair in the era of large language models", 2025.
- [8] Chen et al., "Evaluating Large Language Models Trained on Code", 2021.
- [9] Zhuo et al., "BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions", 2024.
- [10] Zhu et al., "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code", 2024.
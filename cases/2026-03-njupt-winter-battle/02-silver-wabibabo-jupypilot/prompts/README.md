# prompts 目录说明

本目录下的 `*_v1-3.md` 为 **Prompt 迭代展示与多轮模式所用**的提示词模板。  
**赛题① Prompt Engineering**：迭代过程（v1→v2→v3）、约束、Few-shot、System Role + Output Format、思维链（CoT）的落地位置；逐条对照见项目根目录 [赛题符合性-实现映射.md](../赛题符合性-实现映射.md)。

## 谁在什么时候读这些文件？

| 运行路径 | 是否读本目录 md | 说明 |
|----------|------------------|------|
| **Direct 单轮模式** | **不读** | code_patch / testgen / refactor / refactor_diff 在 RAG 有内容时**优先**走单轮；提示词全部在 `jupypilot/orchestrator/tool_loop.py`（`_SIMPLE_TASK_PROMPTS` 及约 405、418 行的 system 拼接） |
| **Tool-loop 多轮模式** | **读** | direct 未走或失败、以及 scaffold 任务走多轮；`PromptRegistry` 按任务名加载对应 `*_v1-3.md`，取「## v3（当前采用）」段落；`ContextBuilder` 拼装 system |
| **memory_summary** | **读** | `memory.py` 拼 system 时，任务内容来自 `memory_summary_v1-3.md` |
| **md 缺失/读失败** | 不读，用代码默认 | `prompt_registry.py` 内 `_DEFAULT_SYSTEM_BASE`、`_DEFAULT_TASK_PROMPTS`（含 refactor_diff）作为回退 |

## 文件与任务映射

- `system_base_v1-3.md` → 系统基础角色与协议（多轮 system 首段）
- `code_qa_v1-3.md`、`code_patch_v1-3.md`、`scaffold_v1-3.md`、`testgen_v1-3.md`、`refactor_v1-3.md` → 各任务多轮用
- `memory_summary_v1-3.md` → 历史递归摘要任务
- `persona_newbie_tutor.md` → 全局人设（`[PERSONA]` 段）：面向编程新手/小白的耐心代码导师，仅在多轮模式下由 `ContextBuilder` 注入，**只影响 `final.content` 内的讲解风格，不改变 JSON envelope 契约与工具调用行为**

修改本目录只影响**多轮**与 **memory_summary**；若希望改单轮表现，需改 `tool_loop.py`。

# 提示词优化说明：Gemini / GPT 风格

本目录下 5 个任务的 prompt（code_qa、code_patch、testgen、refactor、scaffold）及 system_base 已按 **Google Gemini API** 与 **OpenAI 官方文档** 的提示词最佳实践做了对齐优化，便于模型更稳定地遵循格式、减少无效输出。

## 参考来源

- **OpenAI**：[Prompt engineering](https://platform.openai.com/docs/guides/prompt-engineering)、[Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)、Few-shot 与 Message 结构（Identity / Instructions / Examples / Context）
- **Gemini**：[Prompt design strategies](https://ai.google.dev/gemini-api/docs/prompting-strategies)、[Function calling](https://ai.google.dev/gemini-api/docs/function-calling)、代码生成场景的「清晰指令 + 结构化输出」

## 采用的通用原则

| 原则 | 做法 |
|------|------|
| **Identity 先行** | 首句明确角色与任务目标（You are… / Your task is…），再写步骤与约束。 |
| **Instructions 结构化** | 用 `###` 或 `[Section]` 分段（工具顺序、思维链、输出格式），避免大段不分段。 |
| **输出格式具体化** | 用「字段 + 类型 + 约束」描述（类似 JSON Schema）；要求「恰好 1 个」「3–5 条」等可验证表述，避免「尽量」「适当」。 |
| **Few-shot 与格式一致** | 示例与最终要求格式完全一致（含标题、代码块、JSON 键名），便于模型模仿。 |
| **工具描述** | 每个工具一句话说明「何时用、必填/可选参数」，贴近 Gemini 的 function declaration 风格。 |

## 各任务落地方式

- **code_qa / code_patch / testgen**：Identity 句 + 工具顺序/步骤 + 输出结构（技术分析 + 导师讲解）+ 1 个完整 Few-shot。
- **refactor / scaffold**：Identity + 步骤 + **JSON 结构说明（字段与类型）** + 1 个完整 JSON 示例；保留 `findings`/`files` + `mentor_deep_dive` 双部分。
- **system_base**：顶部一句 Identity；工具列表补充「用途/参数」简短说明；输出协议保持「每行单 JSON、可多工具」不变。

当前采用的仍是各文件中的 **v3（当前采用）** 段落；若后续增加 v4，可将上述优化单独放入 v4 便于 A/B 对比。

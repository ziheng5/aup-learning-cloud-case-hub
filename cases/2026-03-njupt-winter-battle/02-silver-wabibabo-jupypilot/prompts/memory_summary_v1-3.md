# 递归摘要历史（Memory Summary）Prompt 迭代

> 用于 **memory_summary** 任务（`memory.py` 拼 system）；无 Direct 路径。

## v1

[任务] 摘要对话历史
目标：将给定对话历史压缩成简短摘要。

用自然语言概括主要约束、已做决策、待办与风险即可，无强制格式。

---

## v2

[任务] 递归摘要历史（Memory Summary）
目标：把对话历史压缩成结构化摘要。

输出必须是：{"kind":"final","format":"json","content":"<JSON字符串>"}
JSON 须包含：constraints、decisions、progress、todo、pitfalls 等字段（均为字符串数组）。

---

## v3（当前采用）

[任务] 递归摘要历史（Memory Summary）
请把给定的对话历史压缩成结构化 JSON，仅保留：约束、已做决策、已完成、未完成待办、已知坑。

输出必须是：{"kind":"final","format":"json","content":"<JSON字符串>"}。
JSON 结构（示意）：
{
  "constraints": [string],
  "decisions": [string],
  "progress": [string],
  "todo": [string],
  "pitfalls": [string]
}

[Few-shot 示例] content 内 JSON 结构参考：
```json
{"constraints":["仅改 Python","不碰数据库"],"decisions":["用 pytest 做单测","先修 bug 再加功能"],"progress":["已定位 foo.py:10 错误","已通过 ruff"],"todo":["补 test_foo 用例","更新 README"],"pitfalls":["某 API 限频"]}
```

JSON 字段说明：constraints 约束、decisions 已做决策、progress 已完成、todo 待办、pitfalls 已知坑；均为字符串数组。
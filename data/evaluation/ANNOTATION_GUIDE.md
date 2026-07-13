# 人工评测标注规范

每条问题至少由一名标注者核对原始 PDF，并填写正确论文、章节、页码、内容块和答案要点。

## 状态

- `silver`：程序从结构化分析证据生成，尚未经过人工复核，不得作为最终人工指标。
- `human_reviewed`：标注者已打开 PDF，确认问题可回答且所有证据定位正确。
- `rejected`：问题歧义、证据错误、无法回答或需要改写。

## 复核要求

1. 问题不能直接泄露论文 ID 或答案。
2. `relevant_paper_ids` 必须包含所有可支持答案的论文。
3. `relevant_pages` 使用 PDF 物理页码。
4. `relevant_block_ids` 必须与 `paper_blocks.jsonl` 一致。
5. `expected_answer_points` 只记录原文可支持的关键点。
6. 两名标注者冲突时由第三人裁决。

正式发布指标只能使用 `annotation_status=human_reviewed` 的记录。

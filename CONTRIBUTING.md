# 开发规范

- Python 版本固定为 3.12，依赖使用 uv 管理。
- 业务代码放在 `src/paper_research`，测试放在 `tests`。
- API 只接收/返回 Pydantic schema，不直接暴露 ORM 对象。
- 数据库操作集中在 repository 层；提交前运行 `ruff check .` 和 `pytest`。
- 新功能需包含正常路径及至少一个失败路径测试。
- 提交信息建议采用 `feat:`、`fix:`、`test:`、`docs:` 等前缀。

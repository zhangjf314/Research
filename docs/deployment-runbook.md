# 部署运行手册

## 启动

```powershell
Copy-Item .env.example .env
.\scripts\start.ps1 -Build
```

启动脚本会构建 API，启动 PostgreSQL、Qdrant、Redis、Nginx，并执行 Alembic 迁移。

## 检查

```powershell
python scripts\check_environment.py
docker compose ps
Invoke-RestMethod http://localhost/api/v1/health
```

## 备份

- PostgreSQL：使用 `pg_dump` 备份论文元数据。
- Qdrant：使用 Snapshot API 备份集合。
- `app_data`：备份原始 PDF、解析 JSON、页面资产和报告。

## 故障定位

- API 失败：使用响应头和错误体中的 `x-request-id` / `request_id` 对齐日志。
- PostgreSQL 或 Qdrant 不可用：查看 `/api/v1/health` 的分项状态。
- OCR 失败：检查 Tesseract 和 `TESSDATA_PREFIX`。
- 外部搜索限流：查看缓存目录和 429 重试日志；不要取消 arXiv 请求节流。

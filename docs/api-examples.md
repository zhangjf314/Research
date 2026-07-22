# API Examples

Minimal PowerShell examples for the public API.

## Upload and index a PDF

```powershell
$upload = curl.exe -sS `
  -F "file=@paper.pdf;type=application/pdf" `
  http://localhost/api/v1/papers/upload |
  ConvertFrom-Json

$paperId = $upload.paper.id

Invoke-RestMethod `
  -Method Post `
  "http://localhost/api/v1/papers/$paperId/index"
```

## Ask a paper-scoped question

```powershell
$qa = @{
  question = "What is the main method proposed by this paper?"
  paper_ids = @($paperId)
  top_k = 5
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  http://localhost/api/v1/qa `
  -ContentType application/json `
  -Body $qa
```

## Health and capabilities

```powershell
Invoke-RestMethod http://localhost/api/v1/health
Invoke-RestMethod http://localhost/api/v1/capabilities
```

Do not include API keys in requests, logs, screenshots, or committed files.

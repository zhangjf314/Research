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

## Search provider import

```powershell
$search = Invoke-RestMethod `
  -Method Post `
  -ContentType "application/json" `
  -Body (@{ query = "retrieval augmented generation"; limit = 5 } | ConvertTo-Json) `
  "http://localhost/api/v1/search/papers"

$candidate = $search.candidates |
  Where-Object { $_.pdf_url } |
  Select-Object -First 1

$paper = Invoke-RestMethod `
  -Method Post `
  -ContentType "application/json" `
  -Body ($candidate | ConvertTo-Json -Depth 8) `
  "http://localhost/api/v1/search/import"
```

The API does not accept arbitrary external PDF URLs. Import candidates must come
from the configured search providers.

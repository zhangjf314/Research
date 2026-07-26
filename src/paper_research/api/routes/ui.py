# ruff: noqa: E501
import html
import json
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from paper_research.analysis.types import PaperAnalysis
from paper_research.api.markdown import render_markdown
from paper_research.config import get_settings
from paper_research.db import get_db
from paper_research.evaluation.report_catalog import REPORTS, report_by_id
from paper_research.repositories.paper import PaperRepository

router = APIRouter()
DbSession = Annotated[Session, Depends(get_db)]


class MarkdownRenderRequest(BaseModel):
    markdown: str = Field(max_length=500_000)


def page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>"
        "body{font-family:system-ui;max-width:1100px;margin:40px auto;padding:0 24px;"
        "color:#172033}h1,h2{color:#173b67}.card{border:1px solid #dbe3ee;border-radius:12px;"
        "padding:18px;margin:14px 0;background:#fff}.muted{color:#64748b}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #dbe3ee;padding:8px}"
        "pre{white-space:pre-wrap;background:#f5f7fa;padding:16px;border-radius:10px}"
        ".report-card{padding:0;overflow:hidden}.report-toolbar{display:flex;"
        "justify-content:space-between;align-items:center;gap:16px;padding:14px 18px;"
        "border-bottom:1px solid #dbe3ee;background:#f8fafc}.report-toolbar div{display:flex;"
        "gap:8px}.report-toolbar button{padding:7px 11px;font-size:.9rem}.markdown-body{"
        "padding:22px;line-height:1.75;overflow-wrap:anywhere}.markdown-body h1{margin-top:0;"
        "padding-bottom:.4rem;border-bottom:1px solid #dbe3ee}.markdown-body h2{margin-top:2rem;"
        "padding-bottom:.25rem;border-bottom:1px solid #edf2f7}.markdown-body h3{margin-top:1.5rem}"
        ".markdown-body p,.markdown-body li{line-height:1.75}"
        ".markdown-body blockquote{margin:1rem 0;"
        "padding:.5rem 1rem;color:#475569;border-left:4px solid #94a3b8;background:#f8fafc}"
        ".markdown-body pre{overflow-x:auto;white-space:pre;background:#0f172a;color:#e2e8f0}"
        ".markdown-body code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}"
        ".markdown-body :not(pre)>code{padding:.15rem .35rem;border-radius:5px;background:#eef2f7;"
        "color:#be123c}.markdown-body table{display:block;overflow-x:auto;width:max-content;"
        "max-width:100%}.markdown-body a{text-decoration:underline}.report-raw{margin:0 18px 18px;"
        "max-height:480px;overflow:auto}"
        "nav{display:flex;gap:18px;padding:14px 0;border-bottom:1px solid #dbe3ee;"
        "margin-bottom:28px}a{color:#155eaa;text-decoration:none}"
        "input,textarea,button{font:inherit;padding:10px;border:1px solid #cbd5e1;"
        "border-radius:8px}"
        "button{background:#155eaa;color:white;cursor:pointer}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}"
        "</style></head><body><nav><a href='/api/v1/ui'>Dashboard</a>"
        "<a href='/api/v1/ui/library'>Library</a><a href='/api/v1/ui/search'>Search</a>"
        "<a href='/api/v1/ui/research'>Deep Research</a>"
        "<a href='/api/v1/ui/evaluation'>Evaluation</a>"
        "<a href='/api/v1/ui/gold-review'>Gold Review</a><a href='/docs'>API Docs</a>"
        f"</nav>{body}</body></html>"
    )


@router.post("/render-markdown", response_class=HTMLResponse)
def render_markdown_fragment(payload: MarkdownRenderRequest) -> HTMLResponse:
    return HTMLResponse(render_markdown(payload.markdown))


@router.get("", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    cards = "".join(
        f"<section class='card'><h2>{title}</h2><p>{description}</p>"
        f"<a href='{url}'>Open</a></section>"
        for title, description, url in (
            ("Paper Library", "Inspect parse, index, and analysis status.", "/api/v1/ui/library"),
            ("External Search", "Search arXiv and Semantic Scholar.", "/api/v1/ui/search"),
            ("Deep Research", "Run the budgeted evidence workflow.", "/api/v1/ui/research"),
            ("Evaluation", "Read reproducible RC audit reports.", "/api/v1/ui/evaluation"),
        )
    )
    return page(
        "PaperResearch Agent",
        "<h1>PaperResearch Agent</h1>"
        "<p class='muted'>Paper RAG and evidence-oriented research assistant.</p>"
        f"<div class='grid'>{cards}</div>",
    )


@router.get("/library", response_class=HTMLResponse)
def library_page(db: DbSession) -> HTMLResponse:
    return page(
        "Paper Library",
        """
        <h1>Paper Library</h1>
        <section class='card'>
          <h2>Upload local PDF</h2>
          <input id='paper-file' type='file' accept='application/pdf'>
          <label><input id='auto-index' type='checkbox' checked> Auto-index after upload</label>
          <button id='upload-paper' type='button' onclick='uploadPaper()'>Upload PDF</button>
          <p id='upload-status' class='muted'></p>
        </section>
        <section class='card'>
          <select id='paper-filter' onchange='loadLibrary()'>
            <option value='all'>All</option><option value='ready'>Ready</option>
            <option value='not-indexed'>Not indexed</option>
            <option value='missing-metadata'>Missing metadata</option>
            <option value='upload'>Upload</option><option value='external_search'>External search</option>
          </select>
          <input id='paper-query' placeholder='Search title' oninput='loadLibrary()'>
          <button type='button' onclick='loadLibrary()'>Refresh</button>
          <p id='missing-metadata-count' class='muted'>Missing metadata: loading...</p>
        </section>
        <table><thead><tr><th>Title</th><th>Authors</th><th>Year</th><th>Source</th>
        <th>Parse</th><th>Index</th><th>Created</th><th>Metadata</th><th>Actions</th>
        </tr></thead><tbody id='library-rows'></tbody></table>
        <section class='card' id='metadata-editor' hidden>
          <h2>Edit Metadata</h2><input id='edit-id' hidden>
          <label>Title <input id='edit-title' size='70'></label><br>
          <label>Authors <input id='edit-authors' size='70' placeholder='Semicolon separated'></label><br>
          <label>Year <input id='edit-year' type='number' min='1900' max='2100'></label><br>
          <label>Venue <input id='edit-venue' size='50'></label><br>
          <label>DOI <input id='edit-doi' size='50'></label><br>
          <label>arXiv ID <input id='edit-arxiv' size='30'></label><br>
          <button type='button' onclick='saveMetadata()'>Save</button>
          <button type='button' onclick='hideMetadataEditor()'>Cancel</button>
          <p id='edit-status' class='muted'></p>
        </section>
        <script>
        const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
        async function loadLibrary(){
          const filter=document.getElementById('paper-filter').value;
          const params=new URLSearchParams({limit:'100'});
          const q=document.getElementById('paper-query').value.trim();
          if(q) params.set('q', q);
          if(filter==='not-indexed') params.set('not_indexed','true');
          if(filter==='missing-metadata') params.set('missing_metadata','true');
          if(filter==='upload'||filter==='external_search') params.set('source_type', filter);
          const response=await fetch('/api/v1/papers?'+params.toString());
          const papers=await response.json();
          const rows=(papers||[]).filter(p=>filter!=='ready'||p.index_status==='READY').map(p=>{
            const authors=(p.authors||[]).join('; ');
            const meta=(p.year?'':'Missing year')+(authors?'':' Missing authors');
            const payload=esc(JSON.stringify(p));
            return `<tr><td>${esc(p.title)}</td><td>${esc(authors||'—')}</td><td>${esc(p.year||'Missing year')}</td>`+
              `<td>${esc(p.source_type)}</td><td>${esc(p.parse_status)}</td><td>${esc(p.index_status)}</td>`+
              `<td>${esc(p.created_at)}</td><td>${esc(meta||'OK')}</td>`+
              `<td><a href='/api/v1/ui/papers/${esc(p.id)}'>Open</a> <a href='/api/v1/papers/${esc(p.id)}/pdf'>Open PDF</a> `+
              `<button type='button' data-paper='${payload}' onclick='editMetadata(this.dataset.paper)'>Edit Metadata</button> `+
              `<button type='button' onclick='enrichMetadata("${esc(p.id)}")'>Enrich Metadata</button> `+
              `<button type='button' onclick='indexPaper("${esc(p.id)}")'>Index/Reindex</button></td></tr>`;
          }).join('');
          document.getElementById('library-rows').innerHTML=rows||'<tr><td colspan="9">No papers match the current filter.</td></tr>';
          const missing=(papers||[]).filter(p=>!p.year || !(p.authors||[]).length).length;
          document.getElementById('missing-metadata-count').textContent=`Missing metadata: ${missing}`;
        }
        async function uploadPaper(){
          const file=document.getElementById('paper-file').files[0]; const status=document.getElementById('upload-status');
          if(!file){status.textContent='Choose a PDF first.';return}
          const form=new FormData(); form.append('file', file, file.name); status.textContent='Uploading...';
          try{const response=await fetch('/api/v1/papers/upload',{method:'POST',body:form}); const data=await response.json();
            if(!response.ok) throw new Error(data.detail||`HTTP ${response.status}`);
            status.textContent=`Uploaded ${file.name}; duplicate=${data.duplicate}; paper=${data.paper.id}; parse=${data.paper.parse_status}`;
            if(document.getElementById('auto-index').checked && !data.duplicate){await indexPaper(data.paper.id);}
            await loadLibrary();
          }catch(error){status.textContent='Upload failed: '+(error.message||error);}
        }
        async function indexPaper(id){
          const response=await fetch(`/api/v1/papers/${id}/index`,{method:'POST'});
          if(!response.ok){const data=await response.json(); alert(data.detail||`Index failed ${response.status}`);}
          await loadLibrary();
        }
        function editMetadata(serialized){
          const p=JSON.parse(serialized); document.getElementById('metadata-editor').hidden=false;
          document.getElementById('edit-id').value=p.id; document.getElementById('edit-title').value=p.title||'';
          document.getElementById('edit-authors').value=(p.authors||[]).join('; ');
          document.getElementById('edit-year').value=p.year||''; document.getElementById('edit-venue').value=p.venue||'';
          document.getElementById('edit-doi').value=p.doi||''; document.getElementById('edit-arxiv').value=p.arxiv_id||'';
        }
        function hideMetadataEditor(){document.getElementById('metadata-editor').hidden=true;}
        async function saveMetadata(){
          const id=document.getElementById('edit-id').value; const payload={};
          const title=document.getElementById('edit-title').value.trim(); if(title) payload.title=title;
          payload.authors=document.getElementById('edit-authors').value.split(';').map(x=>x.trim()).filter(Boolean);
          const year=document.getElementById('edit-year').value; payload.year=year?Number(year):null;
          payload.venue=document.getElementById('edit-venue').value.trim()||null;
          payload.doi=document.getElementById('edit-doi').value.trim()||null;
          payload.arxiv_id=document.getElementById('edit-arxiv').value.trim()||null;
          const response=await fetch(`/api/v1/papers/${id}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
          const data=await response.json(); document.getElementById('edit-status').textContent=response.ok?'Saved':(data.detail||`HTTP ${response.status}`);
          if(response.ok){hideMetadataEditor(); await loadLibrary();}
        }
        async function enrichMetadata(id){
          const status=document.getElementById('upload-status'); status.textContent='Enriching metadata...';
          try{
            const response=await fetch(`/api/v1/papers/${id}/enrich-metadata`,{method:'POST'});
            const data=await response.json();
            const changes=Object.entries(data.changes||{}).map(([k,v])=>`${k}: ${v.old||'鈥?} -> ${v.new||'鈥?'}`).join('; ');
            status.textContent=response.ok ? `Metadata ${data.status}: ${changes||'no changes'}` : (data.detail||`HTTP ${response.status}`);
            await loadLibrary();
          }catch(error){status.textContent='Metadata enrichment failed: '+(error.message||error);}
        }
        loadLibrary();
        </script>""",
    )


@router.get("/search", response_class=HTMLResponse)
def search_page() -> HTMLResponse:
    return page(
        "Paper Search",
        """
        <h1>External Paper Search</h1>
        <div class='card'><input id='query' size='60' placeholder='Search arXiv and Semantic Scholar'>
        <button onclick='searchPapers()'>Search</button></div><div id='results'></div>
        <script>
        const esc=s=>String(s??'').replace(/[&<>"']/g,c=>
          ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
        async function searchPapers(){
          const q=document.getElementById('query').value;
          const r=await fetch('/api/v1/search/papers',{method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({query:q,limit:10,open_access_only:false})});
          const d=await r.json(); const items=d.candidates||[];
          document.getElementById('results').innerHTML=items.map(x=>
            `<section class="card"><h2>${esc(x.title)}</h2><p>${esc(x.abstract)}</p>`+
            `<p class="muted">${esc((x.authors||[]).join('; '))}</p>`+
            `<p>${esc(x.year)} | ${esc(x.venue)} | ${esc(x.source)} | `+
            `${esc(x.doi||x.arxiv_id||'no identifier')}</p>`+
            `<p>Open Access: ${esc(x.open_access)} | PDF: ${x.pdf_url?'available':'No downloadable PDF'}</p>`+
            (x.pdf_url?`<button onclick='importPaper(this.dataset.candidate,false)' data-candidate='${esc(JSON.stringify(x))}'>Import PDF</button> `+
            `<button onclick='importPaper(this.dataset.candidate,true)' data-candidate='${esc(JSON.stringify(x))}'>Import and Index</button>`:
            `<button disabled>No downloadable PDF</button>`)+`</section>`).join('');
        }
        async function importPaper(serialized, autoIndex){
          const candidate=JSON.parse(serialized);
          const response=await fetch('/api/v1/search/import',{method:'POST',
            headers:{'Content-Type':'application/json'},body:JSON.stringify(candidate)});
          const paper=await response.json();
          if(!response.ok){alert(paper.detail||`Import failed ${response.status}`);return;}
          if(autoIndex){await fetch(`/api/v1/papers/${paper.id}/index`,{method:'POST'});}
          alert(`Imported ${paper.title}`);
        }</script>""",
    )


@router.get("/research", response_class=HTMLResponse)
def research_page() -> HTMLResponse:
    return page(
        "Deep Research",
        """
        <h1>Deep Research</h1><div class='card'>
        <textarea id='query' rows='4' style='width:95%'
          placeholder='例如：比较不同 RAG 方法的技术路线、实验结果与局限'></textarea><br>
        <button id='run-research' type='button' onclick='runResearch()'>Run</button>
        <button type='button' onclick='fillExampleQuery()'>填入示例</button>
        <p id='research-status' class='muted'></p></div>
        <section class='card report-card'>
          <div class='report-toolbar'>
            <strong>Research Report</strong>
            <div>
              <button type='button' onclick='copyReport()'>Copy Markdown</button>
              <button type='button' onclick='toggleRaw()'>View Raw</button>
            </div>
          </div>
          <article id='report' class='markdown-body'><p class='muted'>Waiting</p></article>
          <pre id='report-raw' class='report-raw' hidden></pre>
        </section>
        <script>
        let currentReportMarkdown = "";
        async function renderMarkdown(markdown) {
          const response = await fetch('/api/v1/ui/render-markdown', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({markdown}),
          });
          if (!response.ok) {
            throw new Error(`Markdown rendering failed: ${response.status}`);
          }
          return await response.text();
        }
        async function runResearch(){
          const report = document.getElementById('report');
          const raw = document.getElementById('report-raw');
          const status = document.getElementById('research-status');
          const button = document.getElementById('run-research');
          const query = document.getElementById('query').value.trim();
          if (query.length < 3) {
            status.textContent = '请输入至少 3 个字符的研究问题。';
            return;
          }
          button.disabled = true;
          status.textContent = 'Running...';
          report.innerHTML = "<p class='muted'>Generating report...</p>";
          raw.hidden = true;
          try {
            const response = await fetch('/api/v1/research/deep',{method:'POST',
              headers:{'Content-Type':'application/json'},body:JSON.stringify({
              query,allow_external_search:false})});
            const data = await response.json();
            if (!response.ok) {
              const message = typeof data.detail === 'string'
                ? data.detail : `Research failed: HTTP ${response.status}`;
              throw new Error(message);
            }
            const completed = data.status === 'COMPLETED' &&
              data.succeeded === true &&
              data.report_available === true &&
              typeof data.report === 'string' &&
              data.report.trim().length > 0;
            if (data.status === 'PAUSED') {
              currentReportMarkdown = '';
              raw.textContent = JSON.stringify(data, null, 2);
              report.textContent = `Research paused\n\nTask: ${data.task_id || ''}\nReason: ${data.stop_reason || 'paused'}`;
              status.textContent = 'Paused';
              return;
            }
            if (!data.succeeded) {
              currentReportMarkdown = '';
              raw.textContent = JSON.stringify(data, null, 2);
              const usage = data.model_usage || {};
              report.textContent = [
                'Research failed',
                '',
                `Status: ${data.status || data.error_code || 'UNKNOWN'}`,
                `Task: ${data.task_id || ''}`,
                `Reason: ${data.stop_reason || ''}`,
                `Attempts: ${data.request_attempt_count ?? 0}`,
                `Tokens: ${usage.total_tokens ?? 0}`,
                `Estimated cost: ${usage.estimated_cost_usd ?? 0}`,
              ].join('\n');
              status.textContent = 'Failed';
              return;
            }
            currentReportMarkdown = typeof data.report === 'string' ? data.report : '';
            if (!completed) {
              throw new Error('Research response contract error: completed task has no report.');
            }
            const safeHtml = await renderMarkdown(currentReportMarkdown);
            report.innerHTML = safeHtml;
            raw.textContent = currentReportMarkdown;
            status.textContent = `Completed · task ${data.task_id || 'unknown'} · `
              + `${data.status || 'unknown'}`;
          } catch (error) {
            currentReportMarkdown = '';
            report.textContent = error instanceof Error ? error.message : 'Research failed.';
            raw.textContent = '';
            status.textContent = 'Failed';
          } finally {
            button.disabled = false;
          }
        }
        async function copyReport() {
          const status = document.getElementById('research-status');
          if (!currentReportMarkdown) {
            status.textContent = 'No report to copy.';
            return;
          }
          try {
            await navigator.clipboard.writeText(currentReportMarkdown);
            status.textContent = 'Markdown copied.';
          } catch (error) {
            status.textContent = 'Copy failed. Use View Raw and copy manually.';
          }
        }
        function toggleRaw() {
          const raw = document.getElementById('report-raw');
          raw.hidden = !raw.hidden;
        }
        function fillExampleQuery() {
          document.getElementById('query').value =
            'RAG 方法的主要技术路线、实验结果和局限分别是什么？';
        }
        </script>""",
    )


@router.get("/papers/{paper_id}", response_class=HTMLResponse)
def paper_detail_page(paper_id: uuid.UUID, db: DbSession) -> HTMLResponse:
    paper = PaperRepository(db).get(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="paper not found")
    analysis_path = get_settings().parsed_papers_dir / str(paper_id) / "paper_analysis.json"
    analysis = (
        PaperAnalysis.model_validate(json.loads(analysis_path.read_text(encoding="utf-8")))
        if analysis_path.exists()
        else None
    )
    fields = []
    if analysis:
        for label, value in (
            ("Research problem", analysis.research_problem),
            ("Main contributions", analysis.main_contributions),
            ("Method", analysis.method_summary),
            ("Experiments", analysis.experiment_summary),
            ("Results", analysis.main_results),
            ("Limitations", analysis.limitations),
        ):
            values = [value.value] if isinstance(value.value, str) else value.value
            rendered = html.escape("\n".join(values or [])) if values else "No evidence extracted"
            pages = ", ".join(str(item.page_start) for item in value.evidence)
            fields.append(
                f"<section class='card'><h2>{label}</h2><p>{rendered}</p>"
                f"<p class='muted'>Evidence pages: {html.escape(pages or 'none')}</p></section>"
            )
    body = (
        f"<h1>{html.escape(paper.title)}</h1>"
        f"<p class='muted'>Status: {paper.parse_status.value} / {paper.index_status}</p>"
        f"<p><a href='/api/v1/papers/{paper_id}/pdf'>Open PDF</a></p>" + "".join(fields)
    )
    return page(paper.title, body)


@router.get("/evaluation-legacy", response_class=HTMLResponse)
def evaluation_page() -> HTMLResponse:
    reports = [
        ("Release Candidate Audit", Path("docs/release-candidate-audit.md")),
        ("检索冒烟评测", Path("data/reports/retrieval-baseline-audit.md")),
        ("Evaluation v1", Path("docs/evaluation-report-v1.md")),
        ("Stability v1", Path("docs/stability-report-v1.md")),
        ("OCR v1", Path("docs/ocr-audit-v1.md")),
    ]
    cards = []
    for title, path in reports:
        content = path.read_text(encoding="utf-8") if path.exists() else "Report not generated yet."
        cards.append(
            f"<section class='card'><h2>{title}</h2>"
            f"<pre>{html.escape(content)}</pre></section>"
        )
    return page(
        "Evaluation",
        "<h1>基础评测中心 / Release Candidate Evidence</h1>" + "".join(cards),
    )


@router.get("/evaluation", response_class=HTMLResponse)
def evaluation_catalog_page() -> HTMLResponse:
    cards = []
    for report in REPORTS:
        exists = report.markdown_path.exists()
        updated = report.markdown_path.stat().st_mtime if exists else 0
        cards.append(
            f"<section class='card'><h2>{html.escape(report.title)}</h2>"
            f"<p><strong>{html.escape(report.category)}</strong> | "
            f"{'Available' if exists else 'Missing'} | updated {updated:.0f}</p>"
            f"<p>{html.escape(report.description)}</p>"
            f"<a href='/api/v1/ui/evaluation/{report.report_id}'>Open report</a></section>"
        )
    return page(
        "Evaluation",
        "<h1>基础评测中心 / Evaluation Report Catalog</h1>"
        "<p class='muted'>检索冒烟评测 is retained as legacy wording; "
        "current reports are loaded from the public-safe catalog.</p>"
        + "".join(cards),
    )


@router.get("/evaluation/{report_id}", response_class=HTMLResponse)
def evaluation_report_page(report_id: str) -> HTMLResponse:
    report = report_by_id(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="report not registered")
    if not report.markdown_path.exists():
        return page(
            report.title,
            f"<h1>{html.escape(report.title)}</h1><section class='card'>"
            f"<p>Missing report: {html.escape(str(report.markdown_path))}</p></section>",
        )
    markdown = report.markdown_path.read_text(encoding="utf-8")
    body = (
        f"<h1>{html.escape(report.title)}</h1><p class='muted'>{html.escape(report.category)}</p>"
        f"<article class='card markdown-body'>{render_markdown(markdown)}</article>"
    )
    if report.summary_json_path and report.summary_json_path.exists():
        raw = html.escape(report.summary_json_path.read_text(encoding="utf-8")[:50_000])
        body += f"<section class='card'><h2>Raw summary</h2><pre>{raw}</pre></section>"
    return page(report.title, body)


@router.get("/gold-review", response_class=HTMLResponse)
def gold_review_page() -> HTMLResponse:
    return page(
        "Gold Review",
        """
        <h1>Human Gold Review Workbench</h1>
        <div class='card'>
          <label>Reviewer <input id='reviewer'></label>
          <label>Status <select id='status-filter' onchange='load()'>
            <option value=''>all</option><option value='approved'>approved</option>
            <option value='pending'>pending</option><option value='invalid'>invalid</option>
          </select></label>
          <label>Question ID <input id='question-filter' placeholder='q001'></label>
          <button onclick='load()'>Load</button>
          <button onclick='previous()'>Previous</button>
          <button onclick='next()'>Next</button>
        </div>
        <p id='review-state' class='muted'>Loading</p>
        <section class='card' id='item'></section>
        <section class='card'><h2>Evidence blocks</h2><div id='evidence'></div></section>
        <section class='card'>
          <textarea id='notes' rows='4' style='width:95%' placeholder='Review notes'></textarea><br>
          <button onclick="act('approve')">Approve</button>
          <button onclick="act('modify_approve')">Approve after edits</button>
          <button onclick="act('unanswerable')">Mark unanswerable</button>
          <button onclick="act('invalid')">Invalid</button>
          <button onclick="act('defer')">Defer</button>
        </section>
        <script>
        let items=[], index=0;
        const state=document.getElementById('review-state');
        const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
        async function load(){
          state.textContent='Loading';
          try{
            const params=new URLSearchParams({limit:'100'});
            const status=document.getElementById('status-filter').value;
            if(status) params.set('status', status);
            const response=await fetch('/api/v1/evaluation/review?'+params.toString());
            if(!response.ok) throw new Error(`HTTP ${response.status}`);
            const data=await response.json(); items=data.items||[];
            const q=document.getElementById('question-filter').value.trim();
            if(q) items=items.filter(x=>String(x.question_id).includes(q));
            index=0;
            if(!items.length){
              document.getElementById('item').textContent='No review items match the current filter.';
              document.getElementById('evidence').textContent='';
              state.textContent='Empty';
              return;
            }
            await show();
          }catch(error){
            state.textContent='Failed: '+(error.message||error)+'. Retry with the Load button.';
            document.getElementById('item').textContent='加载 Gold Review 失败';
          }
        }
        async function show(){
          if(!items.length){state.textContent='Empty';return;}
          state.textContent=`Loading item ${index+1}/${items.length}`;
          try{
            const response=await fetch('/api/v1/evaluation/review/'+items[index].question_id);
            if(!response.ok) throw new Error(`HTTP ${response.status}`);
            const data=await response.json(); const item=data.item||{};
            const claims=(item.required_claims||[]).map(x=>`<li>${esc(x)}</li>`).join('');
            document.getElementById('item').innerHTML=
              `<h2>${esc(item.question_id)} · ${esc(item.category)} · ${esc(item.difficulty)}</h2>`+
              `<p><strong>Question:</strong> ${esc(item.question)}</p>`+
              `<p><strong>Answerable:</strong> ${esc(item.answerable)}</p>`+
              `<p><strong>Gold Answer:</strong> ${esc(item.gold_answer)}</p>`+
              `<p><strong>Gold Papers:</strong> ${esc((item.gold_paper_ids||[]).join(', '))}</p>`+
              `<p><strong>Gold Pages:</strong> ${esc((item.gold_pages||[]).join(', '))}</p>`+
              `<p><strong>Review Status:</strong> ${esc(item.review_status)} · ${esc(item.reviewer)} · ${esc(item.reviewed_at)}</p>`+
              `<p><strong>Review Notes:</strong> ${esc(item.review_notes)}</p>`+
              `<h3>Required Claims</h3><ul>${claims}</ul>`;
            const warningHtml=(data.warnings||[]).map(w=>`<p class='muted'>${esc(w.code)}: ${esc(w.paper_id)}</p>`).join('');
            const evidenceHtml=(data.evidence||[]).map(b=>
              `<section class='card'><p><strong>${esc(b.paper_id)}</strong> page ${esc(b.page_start)} block ${esc(b.block_id)}</p>`+
              `<p>${esc(b.section_path||'')}</p><pre>${esc(b.text)}</pre></section>`).join('');
            document.getElementById('evidence').innerHTML=evidenceHtml||warningHtml||'<p class="muted">No evidence blocks returned.</p>';
            state.textContent=`Loaded ${index+1}/${items.length}`;
          }catch(error){
            state.textContent='Failed: '+(error.message||error)+'. Retry with the Load button.';
          }
        }
        function previous(){index=Math.max(0,index-1);show()}
        function next(){index=Math.min(items.length-1,index+1);show()}
        async function act(action){
          const reviewer=document.getElementById('reviewer').value;
          if(!reviewer){alert('Reviewer is required');return}
          state.textContent='Saving';
          const response=await fetch('/api/v1/evaluation/review/'+items[index].question_id,{method:'POST',
            headers:{'Content-Type':'application/json'},body:JSON.stringify({action,reviewer,
            review_notes:document.getElementById('notes').value})});
          if(!response.ok){state.textContent=`Save failed: HTTP ${response.status}`; return;}
          state.textContent='Saved';
          await load();
        }
        load();
        </script>""",
    )

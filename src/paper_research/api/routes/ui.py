import html
import json
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from paper_research.analysis.types import PaperAnalysis
from paper_research.config import get_settings
from paper_research.db import get_db
from paper_research.repositories.paper import PaperRepository

router = APIRouter()
DbSession = Annotated[Session, Depends(get_db)]


def page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>"
        "body{font-family:system-ui;max-width:1100px;margin:40px auto;padding:0 24px;"
        "color:#172033}h1,h2{color:#173b67}.card{border:1px solid #dbe3ee;border-radius:12px;"
        "padding:18px;margin:14px 0;background:#fff}.muted{color:#64748b}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #dbe3ee;padding:8px}"
        "pre{white-space:pre-wrap;background:#f5f7fa;padding:16px;border-radius:10px}"
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
    rows = "".join(
        f"<tr><td><a href='/api/v1/ui/papers/{paper.id}'>{html.escape(paper.title)}</a></td>"
        f"<td>{paper.year or ''}</td><td>{paper.parse_status.value}</td>"
        f"<td>{paper.index_status}</td></tr>"
        for paper in PaperRepository(db).list(limit=100)
    )
    return page(
        "Paper Library",
        "<h1>Paper Library</h1><table><thead><tr><th>Title</th><th>Year</th>"
        f"<th>Parse</th><th>Index</th></tr></thead><tbody>{rows}</tbody></table>",
    )


@router.get("/search", response_class=HTMLResponse)
def search_page() -> HTMLResponse:
    return page(
        "Paper Search",
        """
        <h1>External Paper Search</h1>
        <div class='card'><input id='query' size='60' value='retrieval augmented generation'>
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
            `<p class="muted">${esc(x.source)} | ${esc(x.year)} | score `+
            `${esc(x.relevance_score)}</p></section>`).join('');
        }</script>""",
    )


@router.get("/research", response_class=HTMLResponse)
def research_page() -> HTMLResponse:
    return page(
        "Deep Research",
        """
        <h1>Deep Research</h1><div class='card'>
        <textarea id='query' rows='4' style='width:95%'>
        RAG methods, results, and limitations</textarea><br>
        <button onclick='runResearch()'>Run</button></div><pre id='report'>Waiting</pre>
        <script>
        async function runResearch(){document.getElementById('report').textContent='Running...';
          const r=await fetch('/api/v1/research/deep',{method:'POST',
           headers:{'Content-Type':'application/json'},body:JSON.stringify({
           query:document.getElementById('query').value,allow_external_search:false})});
          const d=await r.json();
          document.getElementById('report').textContent=d.report||JSON.stringify(d,null,2);
        }</script>""",
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


@router.get("/evaluation", response_class=HTMLResponse)
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


@router.get("/gold-review", response_class=HTMLResponse)
def gold_review_page() -> HTMLResponse:
    return page(
        "Gold Review",
        """
        <h1>Human Gold Review Workbench</h1>
        <div class='card'>
          <label>Reviewer <input id='reviewer'></label>
          <button onclick='previous()'>Previous</button>
          <button onclick='next()'>Next</button>
        </div>
        <section class='card'><pre id='item'>Loading...</pre></section>
        <section class='card'><h2>Evidence blocks</h2><pre id='evidence'></pre></section>
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
        async function load(){
          const response=await fetch('/api/v1/evaluation/review?limit=100');
          items=(await response.json()).items; await show();
        }
        async function show(){
          if(!items.length)return;
          const response=await fetch('/api/v1/evaluation/review/'+items[index].question_id);
          const data=await response.json();
          document.getElementById('item').textContent=JSON.stringify(data.item,null,2);
          document.getElementById('evidence').textContent=JSON.stringify(data.evidence,null,2);
        }
        function previous(){index=Math.max(0,index-1);show()}
        function next(){index=Math.min(items.length-1,index+1);show()}
        async function act(action){
          const reviewer=document.getElementById('reviewer').value;
          if(!reviewer){alert('Reviewer is required');return}
          await fetch('/api/v1/evaluation/review/'+items[index].question_id,{method:'POST',
            headers:{'Content-Type':'application/json'},body:JSON.stringify({action,reviewer,
            review_notes:document.getElementById('notes').value})});
          await load();
        }
        load();
        </script>""",
    )

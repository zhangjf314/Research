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


RESEARCH_MODE_UI = """
<h1>Research Execution</h1>
<p class='muted'>Choose an explicit execution mode. Both modes reuse the frozen Current Hybrid RAG backend; the difference is the research control path.</p>
<section class='card'>
  <h2>Choose research mode</h2>
  <div class='grid'>
    <section id='workflow-card' class='card mode-card' data-mode='workflow'>
      <h2>Deep Research Workflow</h2>
      <p><strong>Predefined research orchestration</strong></p>
      <p>Runs the existing fixed research workflow for evidence retrieval, evidence organization, report synthesis, and verification.</p>
      <p class='muted'>Fixed orchestration · Frozen RAG · Evidence synthesis · Verification</p>
      <button type='button' onclick='selectResearchMode("workflow")'>Use Workflow</button>
    </section>
    <section id='agent-card' class='card mode-card' data-mode='agent'>
      <h2>Research Agent</h2>
      <p><strong>State/observation-driven research execution</strong></p>
      <p>Uses Planner, current Evidence State, dynamic tool selection, observations, verification, checkpoints, and bounded replan capability.</p>
      <p class='muted'>Planner · Dynamic tools · Evidence State · Verification · Checkpoint · Bounded Replan</p>
      <button type='button' onclick='selectResearchMode("agent")'>Use Agent</button>
    </section>
  </div>
  <details>
    <summary>What is the difference between Workflow and Agent?</summary>
    <table>
      <thead><tr><th></th><th>Workflow</th><th>Agent</th></tr></thead>
      <tbody>
        <tr><td>Research flow</td><td>Predefined orchestration</td><td>Dynamic execution</td></tr>
        <tr><td>RAG</td><td>Frozen Current Hybrid</td><td>Frozen Current Hybrid</td></tr>
        <tr><td>Planner</td><td>Fixed orchestration</td><td>Planner-driven</td></tr>
        <tr><td>Tool Selection</td><td>Fixed workflow path</td><td>Dynamic tool/action selection</td></tr>
        <tr><td>Evidence State</td><td>Workflow-internal</td><td>Explicit state</td></tr>
        <tr><td>Verification</td><td>Available</td><td>Available</td></tr>
        <tr><td>Replan</td><td>Not Agent Replan</td><td>Bounded capability</td></tr>
      </tbody>
    </table>
  </details>
</section>
<div class='card'>
  <p><span id='mode-badge' class='muted'>[ WORKFLOW ]</span> Current mode: <strong id='mode-label'>Deep Research Workflow</strong></p>
  <p id='mode-description' class='muted'>Execution path: predefined orchestration → evidence synthesis → verification.</p>
  <p id='research-paper-scope' class='muted'>Paper scope: all indexed papers. <a href='/api/v1/ui/library'>Choose papers in the Library</a>.</p>
  <fieldset id='agent-options' hidden>
    <legend>ACTIVE论文与研究草稿</legend>
    <div id='active-papers'>正在核验ACTIVE来源…</div>
    <p id='source-error' role='alert'></p>
    <label><input id='generate-draft' type='checkbox' disabled onchange='toggleDraftOptions()'> 启用LLM研究草稿（默认关闭，可能产生费用）</label>
    <p id='draft-availability' class='muted'></p>
    <div id='draft-authorization' hidden>
      <label>调用令牌 <input id='draft-token' type='password' autocomplete='off' spellcheck='false'></label>
      <p class='muted'>仅在已获费用授权时启用。令牌只用于本次请求，不保存到浏览器存储。草稿为UNVERIFIED，不是可信报告。</p>
    </div>
  </fieldset>
  <textarea id='query' rows='4' style='width:95%' placeholder='Example: Compare the main technical routes, experimental findings, and limitations of retrieval-augmented generation methods.'></textarea><br>
  <button id='run-research' type='button' onclick='runResearch()'>Run Workflow</button>
  <button type='button' onclick='fillExampleQuery()'>Fill example</button>
  <button id='reset-research' type='button' onclick='resetResearchState()'>Reset</button>
  <p id='research-status' class='muted'></p>
</div>
<section class='card'>
  <h2 id='status-heading'>Deep Research Workflow status</h2>
  <table>
    <tbody>
      <tr><th>Mode</th><td id='status-mode'>Deep Research Workflow</td></tr>
      <tr><th>Task ID</th><td id='status-task-id'>-</td></tr>
      <tr><th>Status</th><td id='status-state'>Idle</td></tr>
      <tr><th>Elapsed time</th><td id='status-elapsed'>-</td></tr>
      <tr><th>Provider requests</th><td id='status-provider-requests'>-</td></tr>
      <tr><th>Tokens</th><td id='status-tokens'>-</td></tr>
      <tr><th>Estimated cost</th><td id='status-cost'>-</td></tr>
    </tbody>
  </table>
  <div id='workflow-telemetry'>
    <h3>Research Workflow</h3>
    <pre id='workflow-stages'>Waiting</pre>
  </div>
  <div id='agent-telemetry' hidden>
    <h3>Agent trace</h3>
    <table>
      <tbody>
        <tr><th>Planner</th><td id='agent-plan-version'>-</td></tr>
        <tr><th>Current step</th><td id='agent-step-count'>-</td></tr>
        <tr><th>Selected tool</th><td id='agent-selected-tool'>-</td></tr>
        <tr><th>Evidence count</th><td id='agent-evidence-count'>-</td></tr>
        <tr><th>Verification status</th><td id='agent-verification'>-</td></tr>
        <tr><th>Tool calls</th><td id='agent-tool-calls'>-</td></tr>
        <tr><th>Replan count</th><td id='agent-replan-count'>0</td></tr>
      </tbody>
    </table>
    <pre id='agent-trace'>Waiting</pre>
  </div>
</section>
<section class='card report-card'>
  <div class='report-toolbar'>
    <strong id='report-title'>Research Output</strong>
    <div id='report-actions' hidden>
      <button type='button' onclick='copyReport()'>Copy Markdown</button>
      <button type='button' onclick='toggleRaw()'>View Raw</button>
      <button id='save-markdown' type='button' onclick='saveMarkdown()'>保存Markdown</button>
    </div>
  </div>
  <article id='report' class='markdown-body'><p class='muted'>Waiting</p></article>
  <pre id='report-raw' class='report-raw' hidden></pre>
</section>
<section id='candidate-section' class='card' hidden>
  <h2>候选证据 · CANDIDATE</h2>
  <p class='muted'>具有可追溯来源，不代表已验证其语义或研究结论。</p>
  <div id='candidate-evidence'></div>
</section>
<p class='muted'>Research Agent and Workflow are two independent execution architectures. They share the base retrieval capability but use different research control strategies.</p>
<script>
let currentReportMarkdown = "";
let researchMode = new URLSearchParams(window.location.search).get('mode') === 'agent' ? 'agent' : 'workflow';
let taskExecutionMode = researchMode;
let activeTaskRunning = false;
let runStartedAt = null;
let activePapers = [];
let sourcesReady = false;
const modeAdapters = {
  workflow: {
    label: 'Deep Research Workflow',
    badge: '[ WORKFLOW ]',
    runLabel: 'Run Workflow',
    heading: 'Deep Research Workflow status',
    description: 'Execution path: predefined orchestration → evidence synthesis → verification.',
    async submit(query) {
      const response = await fetch('/api/v1/research/deep',{method:'POST',
        headers:{'Content-Type':'application/json'},body:JSON.stringify({query,paper_ids:selectedPaperIds(),allow_external_search:false})});
      return {response, data: await readJson(response)};
    },
    normalize(data) {
      const usage = data.model_usage || {};
      return {
        taskId: data.task_id || '',
        status: data.status || data.error_code || 'UNKNOWN',
        terminal: data.terminal !== false,
        providerRequests: data.provider_completed_request_count ?? data.request_attempt_count ?? 0,
        tokens: usage.total_tokens ?? 0,
        cost: usage.estimated_cost_usd ?? 0,
        report: typeof data.report === 'string' ? data.report : '',
        hasReportBody: typeof data.report === 'string' && data.report.trim().length > 0,
        succeeded: data.status === 'COMPLETED' && data.succeeded === true && data.report_available === true,
        paused: data.status === 'PAUSED',
        stopReason: data.stop_reason || '',
      };
    },
  },
  agent: {
    label: 'Research Agent',
    badge: '[ AGENT ]',
    runLabel: 'Run Agent',
    heading: 'Research Agent status',
    description: 'Execution path: Planner → Dynamic Tool Selection → Evidence State → Verification.',
    async submit(query) {
      const generateDraft = document.getElementById('generate-draft').checked;
      const headers = {'Content-Type':'application/json'};
      if (generateDraft) headers['X-Research-Draft-Token'] = document.getElementById('draft-token').value;
      const response = await fetch('/api/v1/research/agent',{method:'POST',
        headers,body:JSON.stringify({query,paper_ids:selectedPaperIds(),generate_draft:generateDraft})});
      return {response, data: await readJson(response)};
    },
    normalize(data) {
      const usage = data.token_usage || {};
      const tools = data.tool_history || [];
      const lastTool = [...tools].reverse().find(item => item.tool || item.tool_name || item.action || item.phase) || {};
      const report = typeof data.report_markdown === 'string'
        ? data.report_markdown
        : (typeof data.report === 'string' ? data.report : '');
      const selectedTool = lastTool.tool || lastTool.tool_name || lastTool.action || '';
      const selectedToolDisplay = selectedTool || (lastTool.phase ? `Decision event: ${lastTool.phase}` : '-');
      return {
        taskId: data.task_id || '',
        status: data.status || data.failure_code || 'UNKNOWN',
        terminal: data.terminal !== false,
        providerRequests: (data.provider_call_count ?? 0) + (data.research_draft?.request_count ?? 0),
        tokens: (usage.total_tokens ?? usage.total ?? 0) + (data.research_draft?.usage?.total_tokens ?? 0),
        cost: (data.estimated_cost ?? 0) + (data.research_draft?.usage?.estimated_cost_usd ?? 0),
        report,
        hasReportBody: report.trim().length > 0,
        succeeded: data.status === 'COMPLETED',
        paused: data.status === 'PAUSED',
        stopReason: data.stop_reason || data.failure_code || '',
        reportStatus: data.report_status || 'NOT_STARTED',
        reportFailureReason: data.report_failure_reason || '',
        reportProviderRequests: data.report_provider_requests ?? 0,
        reportUsage: data.report_usage || {},
        agent: {
          planVersion: data.plan_version ?? '-',
          stepCount: data.step_count ?? 0,
          selectedTool: selectedToolDisplay,
          evidenceCount: data.evidence_count ?? 0,
          verification: data.verification_state?.status || data.verification_state?.verification_status || '-',
          toolCalls: data.tool_call_count ?? tools.length,
          replanCount: tools.filter(item => String(item.phase || item.action || '').toUpperCase().includes('REPLAN')).length,
          trace: tools.map((item, idx) => {
            let phase = item.phase || item.action || item.tool || 'step';
            const tool = item.tool || item.tool_name || item.action || '';
            if (!tool) {
              phase = `Decision event: ${phase}`;
            }
            return `Step ${idx + 1}: ${phase}${tool && tool !== phase ? ' → ' + tool : ''}`;
          }).join('\\n') || 'No sanitized tool history returned.',
        },
      };
    },
  },
};
async function readJson(response){try{return await response.json();}catch(error){return {};}}
function httpMessage(response, data){
  if(response.status===429){
    const retry=response.headers.get('Retry-After') || data.retry_after_seconds || 'later';
    return `Rate limited. Please retry after ${retry} seconds. Request ${data.request_id || ''}`.trim();
  }
  return data.detail || data.error?.message || `HTTP ${response.status}`;
}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function setReportControls(hasReportBody, title) {
  document.getElementById('report-title').textContent = title;
  document.getElementById('report-actions').hidden = !hasReportBody;
}
function assertModeConsistency(mode, adapter) {
  const mismatches = [];
  if (document.getElementById('mode-label').textContent !== adapter.label) mismatches.push('mode label');
  if (document.getElementById('mode-badge').textContent !== adapter.badge) mismatches.push('mode badge');
  if (document.getElementById('run-research').textContent !== adapter.runLabel) mismatches.push('run button');
  if (document.getElementById('status-mode').textContent !== adapter.label) mismatches.push('status mode');
  if (mismatches.length) {
    throw new Error(`Research mode UI mismatch for ${mode}: ${mismatches.join(', ')}`);
  }
}
function selectResearchMode(mode) {
  if (activeTaskRunning) {
    document.getElementById('research-status').textContent = 'A task is running. Reset or wait for a terminal state before switching mode.';
    return;
  }
  researchMode = mode === 'agent' ? 'agent' : 'workflow';
  taskExecutionMode = researchMode;
  const adapter = modeAdapters[researchMode];
  document.getElementById('mode-label').textContent = adapter.label;
  document.getElementById('mode-badge').textContent = adapter.badge;
  document.getElementById('mode-description').textContent = adapter.description;
  document.getElementById('run-research').textContent = adapter.runLabel;
  document.getElementById('status-heading').textContent = adapter.heading;
  document.getElementById('status-mode').textContent = adapter.label;
  document.getElementById('workflow-card').style.outline = researchMode === 'workflow' ? '3px solid #155eaa' : '';
  document.getElementById('agent-card').style.outline = researchMode === 'agent' ? '3px solid #155eaa' : '';
  document.getElementById('workflow-telemetry').hidden = researchMode !== 'workflow';
  document.getElementById('agent-telemetry').hidden = researchMode !== 'agent';
  document.getElementById('agent-options').hidden = researchMode !== 'agent';
  if (window.history?.replaceState) {
    window.history.replaceState(null, '', `/api/v1/ui/research?mode=${researchMode}`);
  }
  assertModeConsistency(researchMode, adapter);
  clearTaskStatus();
  checkResearchCapabilities();
}
function clearTaskStatus() {
  document.getElementById('status-task-id').textContent = '-';
  document.getElementById('status-state').textContent = 'Idle';
  document.getElementById('status-elapsed').textContent = '-';
  document.getElementById('status-provider-requests').textContent = '-';
  document.getElementById('status-tokens').textContent = '-';
  document.getElementById('status-cost').textContent = '-';
  document.getElementById('workflow-stages').textContent = 'Waiting';
  document.getElementById('agent-plan-version').textContent = '-';
  document.getElementById('agent-step-count').textContent = '-';
  document.getElementById('agent-selected-tool').textContent = '-';
  document.getElementById('agent-evidence-count').textContent = '-';
  document.getElementById('agent-verification').textContent = '-';
  document.getElementById('agent-tool-calls').textContent = '-';
  document.getElementById('agent-replan-count').textContent = '0';
  document.getElementById('agent-trace').textContent = 'Waiting';
}
function updateTaskStatus(normalized, data) {
  const elapsed = runStartedAt ? ((Date.now() - runStartedAt) / 1000).toFixed(1) + 's' : '-';
  document.getElementById('status-mode').textContent = modeAdapters[taskExecutionMode].label;
  document.getElementById('status-task-id').textContent = normalized.taskId || '-';
  document.getElementById('status-state').textContent = normalized.status || '-';
  document.getElementById('status-elapsed').textContent = elapsed;
  document.getElementById('status-provider-requests').textContent = normalized.providerRequests ?? 0;
  document.getElementById('status-tokens').textContent = normalized.tokens ?? 0;
  document.getElementById('status-cost').textContent = normalized.cost ?? 0;
  if (taskExecutionMode === 'workflow') {
    const visitedStages = (data.node_history || []).map((item, idx) => `- ${idx + 1}. ${item}`).join('\\n');
    document.getElementById('workflow-stages').textContent = [
      'Visited stages are not success indicators.',
      `Terminal status: ${normalized.status || '-'}`,
      visitedStages || 'No workflow node history returned.',
    ].join('\\n');
  } else if (normalized.agent) {
    document.getElementById('agent-plan-version').textContent = normalized.agent.planVersion;
    document.getElementById('agent-step-count').textContent = normalized.agent.stepCount;
    document.getElementById('agent-selected-tool').textContent = normalized.agent.selectedTool;
    document.getElementById('agent-evidence-count').textContent = normalized.agent.evidenceCount;
    document.getElementById('agent-verification').textContent = normalized.agent.verification;
    document.getElementById('agent-tool-calls').textContent = normalized.agent.toolCalls;
    document.getElementById('agent-replan-count').textContent = normalized.agent.replanCount;
    document.getElementById('agent-trace').textContent = normalized.agent.trace;
  }
}
async function checkResearchCapabilities(){
  const status=document.getElementById('research-status');
  const button=document.getElementById('run-research');
  if (researchMode === 'agent') {
    button.disabled = activeTaskRunning || !sourcesReady;
    return;
  }
  try{
    const response=await fetch('/api/v1/capabilities');
    const data=await readJson(response);
    if (researchMode !== 'workflow') return;
    const capability=data.capabilities?.deep_research || data.capabilities?.research_synthesis;
    if(!response.ok || !capability || capability.status!=='available'){
      button.disabled=true;
      status.textContent=`Research runtime unavailable: ${capability?.detail || httpMessage(response,data)}`;
      return;
    }
    status.textContent=`Research runtime ready: ${capability.provider || 'provider'} / ${capability.model || 'model'}`;
    button.disabled = activeTaskRunning;
  }catch(error){
    if (researchMode !== 'workflow') return;
    button.disabled=true;
    status.textContent='Research capability check failed.';
  }
}
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
  if (activeTaskRunning) return;
  const report = document.getElementById('report');
  const raw = document.getElementById('report-raw');
  const status = document.getElementById('research-status');
  const button = document.getElementById('run-research');
  const query = document.getElementById('query').value.trim();
  taskExecutionMode = researchMode;
  const adapter = modeAdapters[taskExecutionMode];
  if (taskExecutionMode === 'agent' && (!sourcesReady || !selectedPaperIds().length)) {
    status.textContent = '请至少选择一篇已核验的ACTIVE论文。';
    return;
  }
  if (taskExecutionMode === 'agent' && document.getElementById('generate-draft').checked
      && !document.getElementById('draft-token').value.trim()) {
    status.textContent = '启用草稿需要有效调用令牌与费用授权。';
    return;
  }
  if (query.length < 3) {
    status.textContent = 'Please enter a research question with at least 3 characters.';
    return;
  }
  button.disabled = true;
  activeTaskRunning = true;
  runStartedAt = Date.now();
  status.textContent = `${adapter.label} running...`;
  currentReportMarkdown = '';
  document.getElementById('candidate-section').hidden = true;
  document.getElementById('candidate-evidence').replaceChildren();
  setReportControls(false, `${adapter.label} running`);
  report.innerHTML = "<p class='muted'>Generating report...</p>";
  raw.hidden = true;
  try {
    const {response, data} = await adapter.submit(query);
    if (!response.ok) {
      throw new Error(httpMessage(response, data));
    }
    const normalized = adapter.normalize(data);
    updateTaskStatus(normalized, data);
    raw.textContent = JSON.stringify({execution_mode: taskExecutionMode, ...data}, null, 2);
    if (taskExecutionMode === 'agent' && data.active_provenance
        && ['PARTIAL', 'COMPLETED', 'PAUSED'].includes(normalized.status)) {
      await renderCandidateResult(data, normalized);
      status.textContent = `Research Agent · ${normalized.status} · task ${normalized.taskId}`;
      return;
    }
    if (normalized.paused) {
      currentReportMarkdown = '';
      setReportControls(false, `${adapter.label} Paused`);
      report.textContent = `${adapter.label} paused\\n\\nTask: ${normalized.taskId || ''}\\nReason: ${normalized.stopReason || 'paused'}`;
      status.textContent = `${adapter.label} paused`;
      return;
    }
    if (!normalized.succeeded) {
      currentReportMarkdown = '';
      setReportControls(false, `${adapter.label} Failure Details`);
      report.textContent = [
        `${adapter.label} failed`,
        '',
        `Mode: ${adapter.label}`,
        `Status: ${normalized.status || 'UNKNOWN'}`,
        `Task: ${normalized.taskId || ''}`,
        `Reason: ${normalized.stopReason || ''}`,
        `Provider requests: ${normalized.providerRequests ?? 0}`,
        `Tokens: ${normalized.tokens ?? 0}`,
        `Estimated cost: ${normalized.cost ?? 0}`,
      ].join('\\n');
      status.textContent = `${adapter.label} failed`;
      return;
    }
    currentReportMarkdown = normalized.report || '';
    if (taskExecutionMode === 'workflow') {
      if (!currentReportMarkdown.trim()) {
        throw new Error('Research response contract error: completed task has no report.');
      }
      setReportControls(true, 'Research Report');
      const modeHeader = `Execution Mode\\n${adapter.label}\\n\\nExecution\\nWorkflow\\n\\nOrchestration\\nPredefined\\n\\nRAG\\nFrozen Current Hybrid\\n\\n`;
      report.innerHTML = await renderMarkdown(modeHeader + currentReportMarkdown);
      raw.textContent = currentReportMarkdown;
    } else {
      if (normalized.hasReportBody) {
        setReportControls(true, 'Research Report');
        const reportHtml = await renderMarkdown(currentReportMarkdown);
        report.innerHTML = [
          reportHtml,
          '<hr>',
          '<h2>Agent Execution</h2>',
          '<table><tbody>',
          `<tr><th>Status</th><td>${esc(normalized.status)}</td></tr>`,
          `<tr><th>Verification</th><td>${esc(normalized.agent?.verification || '-')}</td></tr>`,
          `<tr><th>Evidence count</th><td>${esc(normalized.agent?.evidenceCount ?? 0)}</td></tr>`,
          `<tr><th>Tool calls</th><td>${esc(normalized.agent?.toolCalls ?? 0)}</td></tr>`,
          `<tr><th>Replan</th><td>${esc(normalized.agent?.replanCount ?? 0)}</td></tr>`,
          `<tr><th>Report provider requests</th><td>${esc(normalized.reportProviderRequests ?? 0)}</td></tr>`,
          '</tbody></table>',
          '<h2>Sanitized Agent Trace</h2>',
          `<pre>${esc(normalized.agent?.trace || 'No sanitized trace returned.')}</pre>`,
        ].join('');
        raw.textContent = currentReportMarkdown;
      } else {
        currentReportMarkdown = '';
        setReportControls(false, 'Research Agent Execution Result');
        report.innerHTML = [
        `<h1>Research Agent Execution Result</h1>`,
        '<p>This Agent runtime completed evidence gathering and verification, but the current Agent API response does not include a final narrative research report.</p>',
        normalized.reportFailureReason ? `<p><strong>Final report status:</strong> ${esc(normalized.reportStatus)}. ${esc(normalized.reportFailureReason)}</p>` : `<p><strong>Final report status:</strong> ${esc(normalized.reportStatus)}</p>`,
        '<h2>Execution Summary</h2>',
        '<table><tbody>',
        `<tr><th>Mode</th><td>${esc(adapter.label)}</td></tr>`,
        `<tr><th>Control</th><td>State / observation-driven</td></tr>`,
        `<tr><th>RAG</th><td>Frozen Current Hybrid</td></tr>`,
        `<tr><th>Status</th><td>${esc(normalized.status)}</td></tr>`,
        `<tr><th>Verification</th><td>${esc(normalized.agent?.verification || '-')}</td></tr>`,
        `<tr><th>Evidence count</th><td>${esc(normalized.agent?.evidenceCount ?? 0)}</td></tr>`,
        `<tr><th>Tool calls</th><td>${esc(normalized.agent?.toolCalls ?? 0)}</td></tr>`,
        `<tr><th>Replan</th><td>${esc(normalized.agent?.replanCount ?? 0)}</td></tr>`,
        `<tr><th>Report provider requests</th><td>${esc(normalized.reportProviderRequests ?? 0)}</td></tr>`,
        '</tbody></table>',
        '<h2>Sanitized Agent Trace</h2>',
        `<pre>${esc(normalized.agent?.trace || 'No sanitized trace returned.')}</pre>`,
        ].join('');
      }
    }
    status.textContent = `${adapter.label} completed · task ${normalized.taskId || 'unknown'} · ${normalized.status || 'unknown'}`;
  } catch (error) {
    currentReportMarkdown = '';
    setReportControls(false, `${modeAdapters[taskExecutionMode].label} Failure Details`);
    report.textContent = error instanceof Error ? error.message : 'Research failed.';
    raw.textContent = '';
    status.textContent = `${modeAdapters[taskExecutionMode].label} failed`;
  } finally {
    activeTaskRunning = false;
    document.getElementById('draft-token').value = '';
    button.disabled = taskExecutionMode === 'agent' && !sourcesReady;
  }
}
async function copyReport() {
  const status = document.getElementById('research-status');
  if (!currentReportMarkdown) {
    status.textContent = 'No Markdown report to copy. Use View Raw for structured Agent output.';
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
function selectedPaperIds() {
  if (researchMode === 'agent') {
    return [...document.querySelectorAll('#active-papers input:checked')].map(input => input.value)
      .filter(id => activePapers.some(paper => paper.id === id));
  }
  try {
    const ids = JSON.parse(window.localStorage.getItem('paperResearchSelectedPaperIds') || '[]');
    return Array.isArray(ids) ? ids.filter(id => typeof id === 'string' && id) : [];
  } catch (error) {
    return [];
  }
}
function refreshResearchPaperScope() {
  const ids = selectedPaperIds();
  const target = document.getElementById('research-paper-scope');
  target.textContent = ids.length
    ? `Paper scope: ${ids.length} selected indexed paper${ids.length === 1 ? '' : 's'}. `
    : 'Paper scope: all indexed papers. ';
  const link = document.createElement('a');
  link.href = '/api/v1/ui/library';
  link.textContent = 'Choose papers in the Library';
  target.append(link, '.');
}
async function loadActivePapers() {
  const target = document.getElementById('active-papers');
  try {
    const response = await fetch('/api/v1/ui/research/sources');
    const data = await readJson(response);
    if (!response.ok) throw new Error(httpMessage(response, data));
    activePapers = data.papers;
    sourcesReady = data.status === 'ACTIVE' && activePapers.length > 0;
    target.innerHTML = activePapers.map(paper =>
      `<div><label><input type='checkbox' value='${esc(paper.id)}' checked> ${esc(paper.title)}</label> ` +
      `<a href='/api/v1/papers/${encodeURIComponent(paper.id)}/pdf' target='_blank' rel='noopener'>PDF</a> ` +
      `<small>${esc(paper.id)} · ACTIVE</small></div>`).join('');
    document.getElementById('generate-draft').disabled = !data.draft_enabled;
    document.getElementById('draft-availability').textContent = data.draft_enabled
      ? '服务端已配置草稿功能；仍需本次调用的明确授权和令牌。'
      : '服务端未启用真实LLM草稿或未配置调用令牌；仍可检索候选证据。';
    if (!sourcesReady) throw new Error('没有可用的ACTIVE论文，请由受信任操作者通过现有CLI准入PDF。');
  } catch (error) {
    sourcesReady = false;
    target.textContent = 'ACTIVE来源不可用';
    document.getElementById('source-error').textContent = error.message;
  }
  checkResearchCapabilities();
}
function toggleDraftOptions() {
  const enabled = document.getElementById('generate-draft').checked;
  document.getElementById('draft-authorization').hidden = !enabled;
  if (!enabled) document.getElementById('draft-token').value = '';
}
function paperTitle(id) {
  return activePapers.find(paper => paper.id === id)?.title || id || 'Unknown paper';
}
function paperLink(id, page) {
  return `/api/v1/papers/${encodeURIComponent(id)}/pdf#page=${encodeURIComponent(page || 1)}`;
}
async function renderCandidateResult(data, normalized) {
  const evidence = data.active_provenance.evidence || [];
  document.getElementById('candidate-section').hidden = false;
  document.getElementById('candidate-evidence').innerHTML = evidence.map(item =>
    `<details><summary>${esc(paperTitle(item.paper_id))} · p.${esc(item.page)}–${esc(item.page_end || item.page)} · ${esc(item.verification_status)}</summary>` +
    `<p><a href='${paperLink(item.paper_id, item.page)}' target='_blank' rel='noopener'>打开原PDF</a></p>` +
    `<p>paper: ${esc(item.paper_id)}<br>chunk: ${esc(item.chunk_id)}<br>block: ${esc((item.block_ids || []).join(', ') || item.block_id)}</p>` +
    `<pre style='white-space:pre-wrap'>${esc(item.text_or_reference)}</pre>` +
    (item.text_proof?.canonical_text_length > (item.text_proof?.display_text_length ?? (item.text_or_reference || '').length)
      ? '<p class="muted">展示文本已截断，请打开原PDF核对完整上下文。</p>' : '') + '</details>').join('')
    || '<p>本次没有候选证据。</p>';
  const draft = data.research_draft;
  const report = document.getElementById('report');
  if (draft?.status === 'UNVERIFIED' && typeof draft.markdown === 'string' && draft.markdown.trim()) {
    const gaps = Array.isArray(draft.evidence_gaps) ? draft.evidence_gaps : [];
    const sourceLines = (draft.sources || []).map(source =>
      `- [${source.evidence_id}] ${paperTitle(source.paper_id)} · p.${(source.pages || []).join('–')} · paper: ${source.paper_id} · chunk: ${source.chunk_id}`);
    currentReportMarkdown = '> UNVERIFIED · CANDIDATE · PARTIAL：未经语义验证，不是可信报告。\\n\\n' + draft.markdown;
    if (gaps.some(gap => !currentReportMarkdown.includes(gap))) {
      currentReportMarkdown += '\\n\\n## 证据缺口\\n\\n' + gaps.map(gap => '- ' + gap).join('\\n');
    }
    currentReportMarkdown += '\\n\\n## 引用来源\\n\\n' + sourceLines.join('\\n');
    report.innerHTML = await renderMarkdown(currentReportMarkdown);
    report.innerHTML += '<h2>完整evidence_gaps</h2><ul id="evidence-gaps">' +
      gaps.map(gap => `<li>${esc(gap)}</li>`).join('') + '</ul>' +
      `<p>模型：${esc(draft.model)} · 本次模型请求：${esc(draft.request_count)} · Token：${esc(draft.usage?.total_tokens)} · 费用估算(USD)：${esc(draft.usage?.estimated_cost_usd)}</p>`;
    setReportControls(true, 'UNVERIFIED研究草稿');
  } else {
    currentReportMarkdown = '';
    setReportControls(false, 'Research Agent · 候选证据');
    report.textContent = `${normalized.status} · ${evidence.length}条候选证据。` +
      (draft ? ` 草稿状态：${draft.status}。${draft.reason || ''}` : ' 本次未启用LLM草稿。') +
      ' 未获得语义验证，不提供可信报告资格。';
  }
}
function saveMarkdown() {
  if (!currentReportMarkdown) return;
  const url = URL.createObjectURL(new Blob([currentReportMarkdown], {type:'text/markdown;charset=utf-8'}));
  const link = document.createElement('a');
  link.href = url;
  const taskId = document.getElementById('status-task-id').textContent.replace(/[^a-zA-Z0-9_-]/g, '_');
  link.download = `research-${taskId}.md`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function fillExampleQuery() {
  document.getElementById('query').value =
    'Compare the main technical routes, experimental findings, and limitations of retrieval-augmented generation methods.';
}
function resetResearchState() {
  if (activeTaskRunning) return;
  activeTaskRunning = false;
  runStartedAt = null;
  taskExecutionMode = researchMode;
  currentReportMarkdown = '';
  document.getElementById('generate-draft').checked = false;
  document.getElementById('draft-token').value = '';
  toggleDraftOptions();
  document.getElementById('candidate-section').hidden = true;
  document.getElementById('candidate-evidence').replaceChildren();
  setReportControls(false, 'Research Output');
  document.getElementById('report').innerHTML = "<p class='muted'>Waiting</p>";
  document.getElementById('report-raw').textContent = '';
  document.getElementById('report-raw').hidden = true;
  document.getElementById('research-status').textContent = '';
  clearTaskStatus();
}
selectResearchMode(researchMode);
refreshResearchPaperScope();
loadActivePapers();
</script>
"""


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
        "gap:8px}#report-actions[hidden]{display:none}.report-toolbar button{padding:7px 11px;font-size:.9rem}.markdown-body{"
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
        "<a href='/api/v1/ui/research'>Research</a>"
        "<a href='/api/v1/ui/evaluation'>Evaluation</a>"
        "<a href='/api/v1/ui/gold-review'>Gold Review</a><a href='/docs'>API Docs</a>"
        f"</nav>{body}</body></html>"
    )


@router.post("/render-markdown", response_class=HTMLResponse)
def render_markdown_fragment(payload: MarkdownRenderRequest) -> HTMLResponse:
    return HTMLResponse(render_markdown(payload.markdown))


@router.get("", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    research_mode_cards = """
        <section class='card'>
          <h2>Research execution modes</h2>
          <p class='muted'>Workflow and Agent share the same frozen Current Hybrid RAG backend. Choose one execution mode explicitly.</p>
          <div class='grid'>
            <section class='card'>
              <h2>Deep Research Workflow</h2>
              <p><strong>Predefined research orchestration</strong></p>
              <p>Fixed orchestration for retrieval, evidence synthesis, and verification.</p>
              <a href='/api/v1/ui/research?mode=workflow'>Open Workflow</a>
            </section>
            <section class='card'>
              <h2>Research Agent</h2>
              <p><strong>State/observation-driven research execution</strong></p>
              <p>Planner, dynamic tools, Evidence State, verification, checkpoint, and bounded replan.</p>
              <a href='/api/v1/ui/research?mode=agent'>Open Agent</a>
            </section>
          </div>
        </section>
    """
    cards = "".join(
        f"<section class='card'><h2>{title}</h2><p>{description}</p>"
        f"<a href='{url}'>Open</a></section>"
        for title, description, url in (
            ("Paper Library", "Inspect parse, index, and analysis status.", "/api/v1/ui/library"),
            ("External Search", "Search arXiv and Semantic Scholar.", "/api/v1/ui/search"),
            (
                "Research",
                "Choose Deep Research Workflow or Research Agent.",
                "/api/v1/ui/research",
            ),
            ("Evaluation", "Read reproducible RC audit reports.", "/api/v1/ui/evaluation"),
        )
    )
    return page(
        "PaperResearch Agent",
        "<h1>PaperResearch Agent</h1>"
        "<p class='muted'>Paper RAG and evidence-oriented research assistant.</p>"
        f"{research_mode_cards}"
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
        <section class='card' id='direct-qa-card'>
          <h2>Direct QA</h2>
          <p id='selection-summary' class='muted'>Select one or more indexed papers below before asking a question.</p>
          <textarea id='direct-qa-question' rows='4' style='width:95%' placeholder='Ask a question about the selected papers.'></textarea><br>
          <button id='ask-direct-qa' type='button' onclick='askDirectQa()'>Ask selected papers</button>
          <p id='direct-qa-status' class='muted'></p>
          <section id='direct-qa-result' hidden>
            <h3>Answer</h3><article id='direct-qa-answer'></article>
            <h3>Sources and retrieved evidence</h3><div id='direct-qa-evidence'></div>
          </section>
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
        <table><thead><tr><th>Use</th><th>Title</th><th>Authors</th><th>Year</th><th>Source</th>
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
        async function readJson(response){try{return await response.json();}catch(error){return {};}}
        function httpMessage(response, data){
          if(response.status===429){
            const retry=response.headers.get('Retry-After') || data.retry_after_seconds || 'later';
            return `Rate limited. Please retry after ${retry} seconds. Request ${data.request_id || ''}`.trim();
          }
          return data.detail || data.error?.message || `HTTP ${response.status}`;
        }
        const selectionStorageKey='paperResearchSelectedPaperIds';
        let selectedPaperIds=new Set();
        try{selectedPaperIds=new Set(JSON.parse(localStorage.getItem(selectionStorageKey)||'[]'));}catch(error){selectedPaperIds=new Set();}
        function savePaperSelection(){
          localStorage.setItem(selectionStorageKey,JSON.stringify([...selectedPaperIds]));
          const target=document.getElementById('selection-summary');
          const count=selectedPaperIds.size;
          target.textContent=count ? `${count} indexed paper${count===1?'':'s'} selected. This scope is shared with Workflow and Agent.` : 'Select one or more indexed papers below before asking a question.';
        }
        function selectPaper(id, checked){
          if(checked) selectedPaperIds.add(id); else selectedPaperIds.delete(id);
          savePaperSelection();
        }
        function renderDirectQa(answer){
          const result=document.getElementById('direct-qa-result');
          result.hidden=false;
          document.getElementById('direct-qa-answer').textContent=answer.answer || answer.refusal_reason || 'No answer returned.';
          const citations=answer.citations||[];
          document.getElementById('direct-qa-evidence').innerHTML=citations.map(item=>{
            const provenance=(item.block_ids||[]).map(block=>`<code>${esc(block)}</code>`).join(' ');
            const section=item.section || 'Unlabeled section';
            return `<section class='card'><p><strong>${esc(item.paper_id)}</strong> — ${esc(section)}, pages ${esc(item.page_start)}–${esc(item.page_end)}</p>`+
              `<p>${esc(item.quote||'')}</p><p class='muted'>Canonical/block provenance: ${provenance||'not returned by backend'}</p>`+
              (item.pdf_url?`<a href='${esc(item.pdf_url)}'>Open source PDF</a>`:'')+'</section>';
          }).join('') || '<p class="muted">No cited evidence was returned by the backend.</p>';
        }
        async function askDirectQa(){
          const status=document.getElementById('direct-qa-status');
          const button=document.getElementById('ask-direct-qa');
          const question=document.getElementById('direct-qa-question').value.trim();
          const paperIds=[...selectedPaperIds];
          if(!paperIds.length){status.textContent='Select at least one indexed paper before asking Direct QA.';return;}
          if(!question){status.textContent='Enter a question before asking Direct QA.';return;}
          button.disabled=true; status.textContent='QA running...';
          try{
            const response=await fetch('/api/v1/qa',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question,paper_ids:paperIds})});
            const answer=await readJson(response);
            if(!response.ok) throw new Error(httpMessage(response,answer));
            renderDirectQa(answer); status.textContent=answer.refused?'Completed with an evidence limitation.':'Direct QA completed.';
          }catch(error){status.textContent='Direct QA failed: '+(error.message||error);}
          finally{button.disabled=false;}
        }
        async function loadLibrary(){
          const filter=document.getElementById('paper-filter').value;
          const params=new URLSearchParams({limit:'100'});
          const q=document.getElementById('paper-query').value.trim();
          if(q) params.set('q', q);
          if(filter==='not-indexed') params.set('not_indexed','true');
          if(filter==='missing-metadata') params.set('missing_metadata','true');
          if(filter==='upload'||filter==='external_search') params.set('source_type', filter);
          const tbody=document.getElementById('library-rows');
          tbody.innerHTML='<tr><td colspan="10">Loading...</td></tr>';
          const response=await fetch('/api/v1/papers?'+params.toString());
          const papers=await readJson(response);
          if(!response.ok){
            tbody.innerHTML=`<tr><td colspan="10">${esc(httpMessage(response, papers))}</td></tr>`;
            return;
          }
          const rows=(papers||[]).filter(p=>filter!=='ready'||p.index_status==='READY').map(p=>{
            const authors=(p.authors||[]).join('; ');
            const meta=(p.year?'':'Missing year')+(authors?'':' Missing authors');
            const payload=esc(JSON.stringify(p));
            const ready=p.index_status==='READY';
            const checked=ready&&selectedPaperIds.has(p.id)?' checked':'';
            const selector=`<input type='checkbox' aria-label='Select ${esc(p.title)}' onchange='selectPaper("${esc(p.id)}",this.checked)'${checked}${ready?'':' disabled'}>`;
            return `<tr><td>${selector}</td><td>${esc(p.title)}</td><td>${esc(authors||'—')}</td><td>${esc(p.year||'Missing year')}</td>`+
              `<td>${esc(p.source_type)}</td><td>${esc(p.parse_status)}</td><td>${esc(p.index_status)}</td>`+
              `<td>${esc(p.created_at)}</td><td>${esc(meta||'OK')}</td>`+
              `<td><a href='/api/v1/ui/papers/${esc(p.id)}'>Open</a> <a href='/api/v1/papers/${esc(p.id)}/pdf'>Open PDF</a> `+
              `<button type='button' data-paper='${payload}' onclick='editMetadata(this.dataset.paper)'>Edit Metadata</button> `+
              `<button type='button' onclick='enrichMetadata("${esc(p.id)}")'>Enrich Metadata</button> `+
              `<button type='button' onclick='indexPaper("${esc(p.id)}")'>Index/Reindex</button></td></tr>`;
          }).join('');
          document.getElementById('library-rows').innerHTML=rows||'<tr><td colspan="10">No papers match the current filter.</td></tr>';
          savePaperSelection();
          const missing=(papers||[]).filter(p=>!p.year || !(p.authors||[]).length).length;
          document.getElementById('missing-metadata-count').textContent=`Missing metadata: ${missing}`;
        }
        async function uploadPaper(){
          const file=document.getElementById('paper-file').files[0]; const status=document.getElementById('upload-status');
          if(!file){status.textContent='Choose a PDF first.';return}
          const form=new FormData(); form.append('file', file, file.name); status.textContent='Uploading...';
          try{const response=await fetch('/api/v1/papers/upload',{method:'POST',body:form}); const data=await readJson(response);
            if(!response.ok) throw new Error(httpMessage(response,data));
            status.textContent=`Uploaded ${file.name}; duplicate=${data.duplicate}; paper=${data.paper.id}; parse=${data.paper.parse_status}`;
            if(document.getElementById('auto-index').checked && !data.duplicate){await indexPaper(data.paper.id);}
            await loadLibrary();
          }catch(error){status.textContent='Upload failed: '+(error.message||error);}
        }
        async function indexPaper(id){
          const status=document.getElementById('upload-status'); status.textContent='Indexing paper...';
          try{const response=await fetch(`/api/v1/papers/${id}/index`,{method:'POST'}); const data=await readJson(response);
            if(!response.ok) throw new Error(httpMessage(response,data));
            status.textContent=`Index ready: ${data.chunk_count} chunks.`;
          }catch(error){status.textContent='Indexing failed: '+(error.message||error);}
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
          const data=await readJson(response); document.getElementById('edit-status').textContent=response.ok?'Saved':httpMessage(response,data);
          if(response.ok){hideMetadataEditor(); await loadLibrary();}
        }
        async function enrichMetadata(id){
          const status=document.getElementById('upload-status'); status.textContent='Enriching metadata...';
          try{
            const response=await fetch(`/api/v1/papers/${id}/enrich-metadata`,{method:'POST'});
            const data=await readJson(response);
            const changes=Object.entries(data.changes||{}).map(([k,v])=>`${k}: ${v.old ?? 'missing'} -> ${v.new ?? 'missing'}`).join('; ');
            status.textContent=response.ok ? `Metadata ${data.status}: ${changes||'no changes'}` : httpMessage(response,data);
            await loadLibrary();
          }catch(error){status.textContent='Metadata enrichment failed: '+(error.message||error);}
        }
        savePaperSelection(); loadLibrary();
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
        async function readJson(response){try{return await response.json();}catch(error){return {};}}
        function httpMessage(response, data){
          if(response.status===429){
            const retry=response.headers.get('Retry-After') || data.retry_after_seconds || 'later';
            return `Rate limited. Please retry after ${retry} seconds. Request ${data.request_id || ''}`.trim();
          }
          return data.detail || data.error?.message || `HTTP ${response.status}`;
        }
        async function searchPapers(){
          const q=document.getElementById('query').value.trim();
          const target=document.getElementById('results');
          if(!q){target.innerHTML='<section class="card">Enter a search query.</section>';return;}
          target.innerHTML='<section class="card">Searching...</section>';
          const r=await fetch('/api/v1/search/papers',{method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({query:q,limit:10,open_access_only:false})});
          const d=await readJson(r);
          if(!r.ok){target.innerHTML=`<section class="card">${esc(httpMessage(r,d))}</section>`;return;}
          const items=d.candidates||[];
          target.innerHTML=items.length ? items.map(x=>
            `<section class="card"><h2>${esc(x.title)}</h2><p>${esc(x.abstract)}</p>`+
            `<p class="muted">${esc((x.authors||[]).join('; '))}</p>`+
            `<p>${esc(x.year)} | ${esc(x.venue)} | ${esc(x.source)} | `+
            `${esc(x.doi||x.arxiv_id||'no identifier')}</p>`+
            `<p>Open Access: ${esc(x.open_access)} | PDF: ${x.pdf_url?'available':'No downloadable PDF'}</p>`+
            (x.pdf_url?`<button onclick='importPaper(this.dataset.candidate,false)' data-candidate='${esc(JSON.stringify(x))}'>Import PDF</button> `+
            `<button onclick='importPaper(this.dataset.candidate,true)' data-candidate='${esc(JSON.stringify(x))}'>Import and Index</button>`:
            `<button disabled>No downloadable PDF</button>`)+`</section>`).join('') :
            '<section class="card">No candidates found.</section>';
        }
        async function importPaper(serialized, autoIndex){
          const candidate=JSON.parse(serialized);
          const response=await fetch('/api/v1/search/import',{method:'POST',
            headers:{'Content-Type':'application/json'},body:JSON.stringify(candidate)});
          const paper=await readJson(response);
          if(!response.ok){alert(paper.detail||httpMessage(response,paper));return;}
          if(autoIndex){await fetch(`/api/v1/papers/${paper.id}/index`,{method:'POST'});}
          alert(`Imported ${paper.title}`);
        }</script>""",
    )


@router.get("/research/sources")
def research_sources(db: DbSession) -> dict:
    """Read-only paper scope from the same ACTIVE admission gate as the Agent."""
    from qdrant_client import QdrantClient

    from paper_research.indexing.registry import IndexIdentityError, IndexRegistry, runtime_index_identity
    from paper_research.retrieval.pipeline import retrieval_collection

    settings = get_settings()
    registry = IndexRegistry(settings.data_dir / "index_registry.json")
    try:
        collection = retrieval_collection(settings, registry)
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        try:
            with registry.query_admission(
                settings.active_collection, collection,
                runtime_index_identity(settings.provider_metadata, collection), client,
                supported=not settings.retrieval_v2_enabled,
            ) as (chunks, _):
                papers = []
                for paper_id in sorted({chunk.paper_id for chunk in chunks}):
                    paper = PaperRepository(db).get(uuid.UUID(paper_id))
                    if paper is None:
                        raise HTTPException(status_code=503, detail="ACTIVE_PAPER_METADATA_MISSING")
                    papers.append({"id": paper_id, "title": paper.title or paper_id})
        finally:
            client.close()
        token = settings.research_draft_token
        return {"status": "ACTIVE", "papers": papers,
                "draft_enabled": bool(settings.research_draft_enabled and settings.live_model_calls_enabled
                                      and settings.llm_provider != "template"
                                      and token and token.get_secret_value())}
    except IndexIdentityError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc


@router.get("/research", response_class=HTMLResponse)
def research_page() -> HTMLResponse:
    return page("Research", RESEARCH_MODE_UI)
    return page(
        "Research",
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
        async function readJson(response){try{return await response.json();}catch(error){return {};}}
        function httpMessage(response, data){
          if(response.status===429){
            const retry=response.headers.get('Retry-After') || data.retry_after_seconds || 'later';
            return `Rate limited. Please retry after ${retry} seconds. Request ${data.request_id || ''}`.trim();
          }
          return data.detail || data.error?.message || `HTTP ${response.status}`;
        }
        async function checkResearchCapabilities(){
          const status=document.getElementById('research-status');
          const button=document.getElementById('run-research');
          try{
            const response=await fetch('/api/v1/capabilities');
            const data=await readJson(response);
            const capability=data.capabilities?.deep_research || data.capabilities?.research_synthesis;
            if(!response.ok || !capability || capability.status!=='available'){
              button.disabled=true;
              status.textContent=`Deep Research unavailable: ${capability?.detail || httpMessage(response,data)}`;
              return;
            }
            status.textContent=`Deep Research ready: ${capability.provider || 'provider'} / ${capability.model || 'model'}`;
          }catch(error){
            button.disabled=true;
            status.textContent='Deep Research capability check failed.';
          }
        }
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
            status.textContent = 'Please enter a research question with at least 3 characters.';
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
            const data = await readJson(response);
            if (!response.ok) {
              throw new Error(httpMessage(response, data));
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
              ].join('\\n');
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
            'Compare the main technical routes, experimental findings, and limitations of retrieval-augmented generation methods.';
        }
        checkResearchCapabilities();
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

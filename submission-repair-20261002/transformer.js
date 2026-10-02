'use strict';

const $ = (selector) => document.querySelector(selector);
const storageKey = 'transformer-codesign-receipts-v1';
const readReceipts = () => {
  try { const data = JSON.parse(localStorage.getItem(storageKey) || '[]'); return Array.isArray(data) ? data : []; }
  catch { return []; }
};
const saveReceipts = (items) => { try { localStorage.setItem(storageKey, JSON.stringify(items)); return true; } catch { return false; } };
const showMessage = (element, message, error = false) => { element.hidden = false; element.textContent = message; element.classList.toggle('error', error); };
const timeLabel = (value) => new Date(value).toLocaleString('en-US');
let uploadsLocked = false;
// UI-only selection: no new field is sent to the existing submission API.
const versionSelect = $('#submission-version');
const vnextSelected = () => versionSelect?.value === 'vnext';
function updateSubmissionVersion() {
  const preview = vnextSelected();
  $('#submit-button').disabled = uploadsLocked || preview;
  // Keep all existing fields and required attributes intact for legacy uploads.
  $('#transformer-form').hidden = preview;
  if ($('#version-status')) $('#version-status').textContent = preview
    ? 'Phase Two uses its separate submission page (opens October 2 at 11:00 Beijing time). Follow the Submit Phase Two link above; this form remains Phase One only.'
    : 'Phase One selected. Existing ZIP requirements apply.';
}
versionSelect?.addEventListener('change', updateSubmissionVersion);
updateSubmissionVersion();

const tabs = [...document.querySelectorAll('[data-tab]')];
const tabIds = tabs.map(tab => tab.dataset.tab);
function activateTab(id, updateHash = false) {
  if (!tabIds.includes(id)) id = 'background';
  for (const tab of tabs) {
    const active = tab.dataset.tab === id;
    if (active) tab.setAttribute('aria-current', 'page');
    else tab.removeAttribute('aria-current');
    $('#' + tab.dataset.tab).hidden = !active;
  }
  if (updateHash) history.replaceState(null, '', '#' + id);
}
for (const tab of tabs) {
  tab.addEventListener('click', event => { event.preventDefault(); activateTab(tab.dataset.tab, true); });
}
window.addEventListener('hashchange', () => activateTab(location.hash.slice(1)));
if (location.hash === '#statement') location.replace('/statement/');
else activateTab(location.hash.slice(1));

function renderReceipt(data, target) {
  target.replaceChildren(); target.hidden = false;
  const title = document.createElement('h3'); title.textContent = `Submission ${data.id}`; target.append(title);
  const labels = {submitted: 'Received · local report recorded', queued: 'Received · local report recorded', running: 'Received · local report recorded', not_configured: 'Received · local report recorded', graded: `Verified · Score: ${Number(data.score).toFixed(2)}`, ineligible: 'Verified · Not eligible for a score', invalid: 'Invalid submission', error: 'Verification error'};
  const status = document.createElement('p'); status.textContent = labels[data.judge_status] || data.judge_status; target.append(status);
  const details = document.createElement('dl');
  for (const [label, value] of [['Display name', data.name], ['Workload', data.workload_version || 'challenge-workload-v0.6'], ['Submitted', timeLabel(data.created_at)], ['File SHA-256', data.file.sha256], ['Lookup key', data.receipt_key || 'Verified']]) {
    const dt = document.createElement('dt'), dd = document.createElement('dd'); dt.textContent = label; dd.textContent = value; details.append(dt, dd);
  }
  target.append(details);
  if (data.local_result) {
    const local = document.createElement('p');
    local.textContent = data.local_score == null ? 'Local report: no eligible experimental score' : `Local experimental score (self-reported): ${Number(data.local_score).toFixed(2)} · Area: ${Number(data.local_result.area_mm2).toFixed(3)} mm²`;
    target.append(local);
    const table = document.createElement('table');
    const head = document.createElement('thead'); head.innerHTML = '<tr><th>Local case</th><th>Cycles</th><th>Power</th><th>Correct</th></tr>'; table.append(head);
    const body = document.createElement('tbody');
    for (const [name, item] of Object.entries(data.local_result.cases || {})) {
      const row = document.createElement('tr');
      for (const value of [name, Number(item.cycles).toLocaleString('en-US'), `${Number(item.peak_power_w).toFixed(2)} W`, item.functional_passed ? 'Yes' : 'No']) {
        const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
      }
      body.append(row);
    }
    table.append(body); const wrap = document.createElement('div'); wrap.className = 'table-wrap'; wrap.append(table); target.append(wrap);
  }
  if (data.result) {
    const result = data.result;
    if (result.message) { const message = document.createElement('p'); message.textContent = result.message; target.append(message); }
    if (result.area_mm2 != null) {
      const summary = document.createElement('p'); summary.textContent = `Area: ${Number(result.area_mm2).toFixed(3)} mm² · Area limit: ${result.area_gate_passed ? 'passed' : 'failed'}`; target.append(summary);
    }
    if (result.cases) {
      const table = document.createElement('table');
      const head = document.createElement('thead'); head.innerHTML = '<tr><th>Case</th><th>Cycles</th><th>Correct</th><th>Power</th><th>Latency limit</th></tr>'; table.append(head);
      const body = document.createElement('tbody');
      for (const [name, item] of Object.entries(result.cases)) {
        const row = document.createElement('tr');
        for (const value of [name, Number(item.cycles).toLocaleString('en-US'), item.functional_passed ? 'Yes' : 'No', `${Number(item.peak_power_w).toFixed(2)} W ${item.power_gate_passed ? '✓' : '✗'}`, item.latency_gate_passed ? 'Passed' : 'Failed']) {
          const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
        }
        body.append(row);
      }
      table.append(body); const wrap = document.createElement('div'); wrap.className = 'table-wrap'; wrap.append(table); target.append(wrap);
    }
  }
  if (data.receipt_key) {
    const download = document.createElement('button'); download.type = 'button'; download.textContent = 'Download JSON receipt';
    download.onclick = () => {
      const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'}));
      const link = document.createElement('a'); link.href = url; link.download = `${data.id}-receipt.json`; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    };
    target.append(download);
  }
}

function followSubmission(id, key, target) {
  let attempts = 0;
  const poll = async () => {
    try {
      const response = await fetch(`/api/transformer/submissions/${encodeURIComponent(id)}`, {headers: {'X-Receipt-Key': key}});
      if (!response.ok) return;
      const data = {...await response.json(), receipt_key: key};
      renderReceipt(data, target);
      const receipts = readReceipts();
      const index = receipts.findIndex(item => item.id === id);
      if (index >= 0) { receipts[index] = data; saveReceipts(receipts); renderLocalReceipts(); }
      if (data.judge_status === 'submitted' && attempts++ < 120) setTimeout(poll, 15000);
      else refreshBoard();
    } catch { /* The receipt remains available for a later manual lookup. */ }
  };
  setTimeout(poll, 15000);
}

async function lookup(id, key) {
  showMessage($('#lookup-status'), 'Looking up the submission…');
  try {
    const response = await fetch(`/api/transformer/submissions/${encodeURIComponent(id)}`, {headers: {'X-Receipt-Key': key}});
    const data = await response.json(); if (!response.ok) throw Error(data.error || 'Lookup failed');
    renderReceipt({...data, receipt_key: key}, $('#lookup-result'));
    showMessage($('#lookup-status'), 'Submission retrieved from the server.');
  } catch (error) { $('#lookup-result').hidden = true; showMessage($('#lookup-status'), error.message, true); }
}

function renderLocalReceipts() {
  const list = $('#local-receipts'); list.replaceChildren();
  const receipts = readReceipts(); $('#no-receipts').hidden = receipts.length > 0;
  for (const item of receipts) {
    const li = document.createElement('li'), button = document.createElement('button');
    button.type = 'button'; button.textContent = `${item.id} · ${item.name} · ${timeLabel(item.created_at)}`;
    button.onclick = () => { const form = $('#lookup-form'); form.elements.id.value = item.id; form.elements.key.value = item.receipt_key; activateTab('submissions', true); lookup(item.id, item.receipt_key); };
    li.append(button); list.append(li);
  }
}

async function refreshBoard() {
  const body = $('#board-body'); body.replaceChildren();
  showMessage($('#board-status'), 'Loading leaderboard…');
  try {
    const response = await fetch('/api/transformer/leaderboard');
    const data = await response.json(); if (!response.ok) throw Error(data.error || 'Could not load leaderboard');
    if (!data.entries.length) { showMessage($('#board-status'), 'No eligible submissions yet.'); return; }
    for (const entry of data.entries) {
      const row = document.createElement('tr');
      for (const value of [entry.rank, entry.name, Number(entry.score).toFixed(2), entry.score_source === 'verified' ? 'Verified' : 'Local', timeLabel(entry.created_at)]) {
        const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
      }
      body.append(row);
    }
    showMessage($('#board-status'), `Showing the best score for ${data.entries.length} participants.`);
  } catch (error) { showMessage($('#board-status'), error.message, true); }
}

$('#transformer-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (vnextSelected()) return;
  if (uploadsLocked) return showMessage($('#submit-status'), 'Submissions are temporarily paused.', true);
  const form = event.currentTarget, file = form.elements.artifact.files[0];
  if (!form.reportValidity()) return;
  if (!file || !/\.zip$/i.test(file.name) || file.size === 0 || file.size > 25 * 1024 * 1024) return showMessage($('#submit-status'), 'Choose a nonempty ZIP up to 25 MiB.', true);
  const button = $('#submit-button'); button.disabled = true; showMessage($('#submit-status'), 'Uploading and saving…');
  try {
    const response = await fetch('/api/transformer/submissions', {method: 'POST', body: new FormData(form)});
    const data = await response.json(); if (!response.ok) throw Error(data.error || 'Submission failed');
    const saved = saveReceipts([data, ...readReceipts()].slice(0, 50));
    renderReceipt(data, $('#receipt')); renderLocalReceipts();
    refreshBoard();
    if (data.judge_status === 'submitted') followSubmission(data.id, data.receipt_key, $('#receipt'));
    showMessage($('#submit-status'), saved ? 'Submission saved. Download and keep the receipt.' : 'Submission saved. Browser storage is unavailable; download the receipt now.');
    form.reset(); $('#receipt').scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (error) { showMessage($('#submit-status'), error.message, true); }
  finally { updateSubmissionVersion(); }
});

$('#lookup-form').addEventListener('submit', (event) => {
  event.preventDefault(); const form = event.currentTarget;
  if (form.reportValidity()) lookup(form.elements.id.value.trim(), form.elements.key.value.trim());
});
$('#clear-receipts').onclick = () => { saveReceipts([]); renderLocalReceipts(); };
$('#refresh-board').onclick = refreshBoard;
async function refreshUploadState() {
  try {
    const response = await fetch('/api/health');
    if (!response.ok) return;
    const health = await response.json();
    if (health.uploads_locked) {
      uploadsLocked = true;
      $('#submit-button').disabled = true;
      showMessage($('#submit-status'), 'Submissions are temporarily paused. Existing receipts and the leaderboard remain available.');
    } else if (uploadsLocked) {
      uploadsLocked = false;
      updateSubmissionVersion();
      showMessage($('#submit-status'), '');
    }
  } catch { /* Server-side upload locking remains authoritative. */ }
}
renderLocalReceipts(); refreshBoard();
refreshUploadState();
setInterval(refreshUploadState, 10000);

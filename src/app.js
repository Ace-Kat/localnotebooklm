'use strict';

const API = 'http://127.0.0.1:47392';
const LARGE_CHUNK_THRESHOLD = 50000; // ~5GB worth of ~500-word chunks

// ── State ────────────────────────────────────────────────────────────────────
let state = {
  notebooks: [],
  activeId: null,
  documents: [],
  chatMessages: [],
  chatPending: false,
};

// ── API helpers ──────────────────────────────────────────────────────────────
async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  if (res.status === 204) return null;
  return res.json();
}

// ── Startup / setup flow ─────────────────────────────────────────────────────
const REQUIRED_MODELS = ['qwen2.5:7b-instruct-q4_K_M', 'nomic-embed-text'];

async function pollBackendReady(maxMs = 30000) {
  const t0 = Date.now();
  while (Date.now() - t0 < maxMs) {
    try {
      const res = await fetch(API + '/api/status');
      if (res.ok) return await res.json();
    } catch (_) {}
    await sleep(1000);
  }
  return null;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function runSetupFlow() {
  show('loading-screen');

  const status = await pollBackendReady(30000);
  if (!status) {
    show('setup-screen');
    renderSetupScreen({ ollama_running: false, models: {} });
    return;
  }

  // Ollama may still be starting up (just auto-launched on macOS). Poll until it
  // responds or we give up, so the user doesn't get stuck on the setup screen.
  let current = status;
  if (!current.ollama_running) {
    show('setup-screen');
    renderSetupScreen(current);
    current = await waitForOllama(current);
  }

  if (!current.ollama_running) {
    renderSetupScreen(current);
    return;
  }

  if (REQUIRED_MODELS.some(m => !current.models[m])) {
    show('setup-screen');
    renderSetupScreen(current);
    document.getElementById('setup-pull-btn').style.display = 'none';
    await autoPullAll(current.models);
    return;
  }

  await enterApp();
}

async function waitForOllama(initialStatus, maxAttempts = 15) {
  for (let i = 0; i < maxAttempts; i++) {
    await sleep(2000);
    try {
      const s = await fetch(API + '/api/status').then(r => r.json());
      renderSetupScreen(s);
      if (s.ollama_running) return s;
    } catch (_) {}
  }
  return initialStatus;
}

async function autoPullAll(modelStatus) {
  const errEl = document.getElementById('setup-error');
  errEl.style.display = 'none';
  try {
    for (const model of REQUIRED_MODELS) {
      if (modelStatus[model]) continue;
      await pullModelWithProgress(model);
    }
    await enterApp();
  } catch (err) {
    errEl.textContent = '⚠ Download failed: ' + err.message;
    errEl.style.display = '';
    document.getElementById('setup-pull-btn').style.display = '';
    document.getElementById('setup-pull-btn').disabled = false;
  }
}

const IS_MAC = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);

// Set platform-appropriate install hint once on load
(function () {
  const hint = document.getElementById('ollama-step-hint');
  if (hint && !IS_MAC) {
    hint.innerHTML = 'After installing, run <code style="display:inline;padding:1px 5px;">ollama serve</code> in a terminal.';
  }
})();

function renderSetupScreen(status) {
  const ollamaDesc = document.getElementById('ollama-step-desc');
  if (status.ollama_running) {
    ollamaDesc.textContent = '✓ Ollama is running.';
    ollamaDesc.style.color = '#15803d';
  } else if (IS_MAC) {
    ollamaDesc.textContent = 'Ollama is not running. Open Ollama from your Applications folder or menu bar, then click Check Again.';
    ollamaDesc.style.color = 'var(--danger)';
  } else {
    ollamaDesc.textContent = 'Ollama is not detected. Please install and start it.';
    ollamaDesc.style.color = 'var(--danger)';
  }

  const list = document.getElementById('model-status-list');
  list.innerHTML = '';
  const allReady = REQUIRED_MODELS.every(m => status.models[m]);

  for (const model of REQUIRED_MODELS) {
    const ready = !!status.models[model];
    const row = document.createElement('div');
    row.className = 'model-row';
    row.id = 'model-row-' + model.replace(/[^a-z0-9]/g, '-');
    row.innerHTML = `
      <span class="model-name">${model}</span>
      <span class="model-badge ${ready ? 'badge-ok' : 'badge-missing'}">${ready ? 'Ready' : 'Not pulled'}</span>
    `;
    list.appendChild(row);
  }

  const pullBtn = document.getElementById('setup-pull-btn');
  const doneBtn = document.getElementById('setup-done-btn');

  if (!status.ollama_running) {
    pullBtn.style.display = 'none';
    doneBtn.style.display = 'none';
  } else if (allReady) {
    pullBtn.style.display = 'none';
    doneBtn.style.display = '';
  } else {
    pullBtn.style.display = '';
    doneBtn.style.display = 'none';
  }
}

document.getElementById('setup-recheck-btn').addEventListener('click', async () => {
  try {
    const status = await fetch(API + '/api/status').then(r => r.json());
    renderSetupScreen(status);
    if (status.ollama_running && REQUIRED_MODELS.every(m => status.models[m])) {
      await enterApp();
    } else if (status.ollama_running && REQUIRED_MODELS.some(m => !status.models[m])) {
      document.getElementById('setup-pull-btn').style.display = 'none';
      await autoPullAll(status.models);
    }
  } catch (e) {
    renderSetupScreen({ ollama_running: false, models: {} });
  }
});

document.getElementById('setup-done-btn').addEventListener('click', enterApp);

document.getElementById('setup-pull-btn').addEventListener('click', async () => {
  const pullBtn = document.getElementById('setup-pull-btn');
  const errEl = document.getElementById('setup-error');
  pullBtn.disabled = true;
  errEl.style.display = 'none';

  let status;
  try {
    status = await fetch(API + '/api/status').then(r => r.json());
  } catch (e) {
    errEl.textContent = 'Cannot reach backend. Is the app running?';
    errEl.style.display = '';
    pullBtn.disabled = false;
    return;
  }

  if (!status.ollama_running) {
    errEl.textContent = 'Ollama is not running. Start it with `ollama serve` and try again.';
    errEl.style.display = '';
    pullBtn.disabled = false;
    return;
  }

  for (const model of REQUIRED_MODELS) {
    if (status.models[model]) continue;
    await pullModelWithProgress(model);
  }

  const finalStatus = await fetch(API + '/api/status').then(r => r.json());
  renderSetupScreen(finalStatus);
  if (REQUIRED_MODELS.every(m => finalStatus.models[m])) {
    await enterApp();
  } else {
    pullBtn.disabled = false;
  }
});

async function pullModelWithProgress(model) {
  const rowId = 'model-row-' + model.replace(/[^a-z0-9]/g, '-');
  const row = document.getElementById(rowId);

  row.innerHTML = `
    <span class="model-name">${model}</span>
    <span class="model-badge badge-pulling">Downloading…</span>
    <div class="pull-progress">
      <div class="pull-bar-track"><div class="pull-bar-fill" style="width:0%"></div></div>
      <span class="pull-status-text">Starting…</span>
    </div>
  `;

  const fill = row.querySelector('.pull-bar-fill');
  const statusText = row.querySelector('.pull-status-text');

  const res = await fetch(API + '/api/models/pull/' + encodeURIComponent(model), { method: 'POST' });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split('\n\n');
    buf = lines.pop();
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      try {
        const data = JSON.parse(line.slice(6));
        if (data.total && data.completed) {
          const pct = Math.round((data.completed / data.total) * 100);
          fill.style.width = pct + '%';
          statusText.textContent = `${data.status || 'downloading'} ${pct}%`;
        } else if (data.status) {
          statusText.textContent = data.status;
        }
        if (data.status === 'done') {
          fill.style.width = '100%';
          statusText.textContent = 'Complete';
          row.querySelector('.model-badge').textContent = 'Ready';
          row.querySelector('.model-badge').className = 'model-badge badge-ok';
        }
      } catch (_) {}
    }
  }
}

// ── Enter app ────────────────────────────────────────────────────────────────
async function enterApp() {
  show('main-app');
  await loadNotebooks();
}

// ── Training ──────────────────────────────────────────────────────────────────
let trainingInProgress = false;

async function checkTrainingStatus() {
  if (!state.activeId) return;
  try {
    const s = await api('GET', `/api/notebooks/${state.activeId}/training-status`);
    renderTrainSection(s.trained);
  } catch (_) {
    renderTrainSection(false);
  }
}

function renderTrainSection(trained) {
  const section = document.getElementById('train-section');
  const banner = document.getElementById('train-banner');
  const label = document.getElementById('train-status-label');
  const sub = document.getElementById('train-status-sub');
  const btn = document.getElementById('train-btn');
  const progressSection = document.getElementById('train-progress-section');

  if (state.documents.length === 0) {
    section.classList.add('hidden');
    return;
  }

  section.classList.remove('hidden');
  progressSection.classList.add('hidden');
  banner.classList.remove('hidden');

  if (trained) {
    banner.classList.add('trained');
    label.textContent = 'Model trained on these documents';
    sub.textContent = 'Chat uses your fine-tuned model + retrieval';
    btn.textContent = 'Retrain';
  } else {
    banner.classList.remove('trained');
    label.textContent = 'Train model on these documents';
    sub.textContent = 'Fine-tune the AI on your content — not just retrieval';
    btn.textContent = 'Train Model';
  }
  btn.disabled = trainingInProgress;
}

document.getElementById('train-btn').addEventListener('click', () => {
  if (trainingInProgress || !state.activeId) return;
  document.getElementById('train-confirm-modal').classList.remove('hidden');
  setTimeout(() => document.getElementById('train-confirm-ok').focus(), 50);
});

document.getElementById('train-confirm-cancel').addEventListener('click', () => {
  document.getElementById('train-confirm-modal').classList.add('hidden');
});

document.getElementById('train-confirm-ok').addEventListener('click', async () => {
  document.getElementById('train-confirm-modal').classList.add('hidden');

  trainingInProgress = true;
  const banner = document.getElementById('train-banner');
  const progressSection = document.getElementById('train-progress-section');
  const fill = document.getElementById('train-progress-fill');
  const progressText = document.getElementById('train-progress-text');
  const btn = document.getElementById('train-btn');

  btn.disabled = true;
  banner.classList.add('hidden');
  progressSection.classList.remove('hidden');
  fill.style.width = '0%';
  fill.style.background = 'var(--accent)';
  progressText.style.color = '';
  progressText.textContent = 'Starting…';

  try {
    const res = await fetch(API + '/api/notebooks/' + state.activeId + '/train', { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n\n');
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const data = JSON.parse(line.slice(6));
          if (data.type === 'progress') {
            fill.style.width = (data.percent || 0) + '%';
            progressText.textContent = data.message || 'Training…';
          } else if (data.type === 'done') {
            fill.style.width = '100%';
            progressText.textContent = data.message || 'Training complete!';
            trainingInProgress = false;
            await sleep(1800);
            await checkTrainingStatus();
            updateModelModeBar(true);
          } else if (data.type === 'error') {
            throw new Error(data.message);
          }
        } catch (parseErr) {
          if (parseErr.message !== 'Unexpected end of JSON input') throw parseErr;
        }
      }
    }
  } catch (err) {
    progressText.textContent = '⚠ ' + err.message;
    progressText.style.color = 'var(--danger)';
    fill.style.background = 'var(--danger)';
    trainingInProgress = false;
    btn.disabled = false;
    banner.classList.remove('hidden');
  }
});

function updateModelModeBar(trained) {
  const bar = document.getElementById('model-mode-bar');
  const text = document.getElementById('model-mode-text');
  if (trained) {
    bar.classList.remove('hidden');
    text.textContent = '🧠 Using fine-tuned model + retrieval';
  } else {
    bar.classList.add('hidden');
  }
}

// ── Notebooks ────────────────────────────────────────────────────────────────
async function loadNotebooks() {
  state.notebooks = await api('GET', '/api/notebooks');
  renderSidebar();
}

function renderSidebar() {
  const list = document.getElementById('notebook-list');
  list.innerHTML = '';
  if (state.notebooks.length === 0) {
    list.innerHTML = '<p style="padding:12px 10px;font-size:12px;color:var(--text-3);">No notebooks yet. Create one above.</p>';
    return;
  }
  for (const nb of state.notebooks) {
    const btn = document.createElement('button');
    btn.className = 'nb-item' + (nb.id === state.activeId ? ' active' : '');
    btn.innerHTML = `
      <span class="nb-item-name">${esc(nb.name)}</span>
      <span class="nb-item-date">${formatDate(nb.last_opened_at)}</span>
    `;
    btn.addEventListener('click', () => openNotebook(nb.id));
    list.appendChild(btn);
  }
}

async function openNotebook(id) {
  state.activeId = id;
  renderSidebar();
  document.getElementById('empty-state').classList.add('hidden');
  document.getElementById('notebook-view').classList.remove('hidden');

  const nb = state.notebooks.find(n => n.id === id);
  document.getElementById('notebook-title-input').value = nb ? nb.name : '';

  await Promise.all([loadDocuments(), loadChatHistory(), checkTrainingStatus()]);
  switchTab('documents');
}

// ── New notebook ──────────────────────────────────────────────────────────────
document.getElementById('new-notebook-btn').addEventListener('click', () => {
  document.getElementById('new-notebook-name').value = '';
  document.getElementById('new-notebook-modal').classList.remove('hidden');
  document.getElementById('new-notebook-name').focus();
});

document.getElementById('new-notebook-cancel').addEventListener('click', () =>
  document.getElementById('new-notebook-modal').classList.add('hidden')
);

document.getElementById('new-notebook-confirm').addEventListener('click', async () => {
  const name = document.getElementById('new-notebook-name').value.trim();
  if (!name) return;
  document.getElementById('new-notebook-modal').classList.add('hidden');
  const nb = await api('POST', '/api/notebooks', { name });
  state.notebooks.unshift(nb);
  renderSidebar();
  await openNotebook(nb.id);
});

document.getElementById('new-notebook-name').addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('new-notebook-confirm').click();
  if (e.key === 'Escape') document.getElementById('new-notebook-cancel').click();
});

// ── Rename notebook ───────────────────────────────────────────────────────────
document.getElementById('notebook-title-input').addEventListener('change', async e => {
  const name = e.target.value.trim();
  if (!name || !state.activeId) return;
  const nb = await api('PUT', '/api/notebooks/' + state.activeId, { name });
  const idx = state.notebooks.findIndex(n => n.id === state.activeId);
  if (idx !== -1) state.notebooks[idx] = nb;
  renderSidebar();
});

// ── Delete notebook ───────────────────────────────────────────────────────────
document.getElementById('delete-notebook-btn').addEventListener('click', () => {
  const nb = state.notebooks.find(n => n.id === state.activeId);
  if (!nb) return;
  document.getElementById('delete-nb-name').textContent = nb.name;
  document.getElementById('delete-notebook-modal').classList.remove('hidden');
});

document.getElementById('delete-nb-cancel').addEventListener('click', () =>
  document.getElementById('delete-notebook-modal').classList.add('hidden')
);

document.getElementById('delete-nb-confirm').addEventListener('click', async () => {
  document.getElementById('delete-notebook-modal').classList.add('hidden');
  await api('DELETE', '/api/notebooks/' + state.activeId);
  state.notebooks = state.notebooks.filter(n => n.id !== state.activeId);
  state.activeId = null;
  renderSidebar();
  document.getElementById('notebook-view').classList.add('hidden');
  document.getElementById('empty-state').classList.remove('hidden');
});

// ── Tabs ──────────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.tab === tab);
  });
  document.getElementById('documents-panel').classList.toggle('hidden', tab !== 'documents');
  document.getElementById('chat-panel').classList.toggle('hidden', tab !== 'chat');
  if (tab === 'chat') {
    scrollChatToBottom();
    document.getElementById('chat-input').focus();
  }
}

// ── Documents ─────────────────────────────────────────────────────────────────
async function loadDocuments() {
  state.documents = await api('GET', '/api/notebooks/' + state.activeId + '/documents');
  renderDocuments();
}

function renderDocuments() {
  const list = document.getElementById('doc-list');
  list.innerHTML = '';
  if (state.documents.length === 0) {
    list.innerHTML = '<p style="font-size:13px;color:var(--text-3);text-align:center;padding:16px;">No documents yet.</p>';
  }
  for (const doc of state.documents) {
    const row = document.createElement('div');
    row.className = 'doc-row';
    row.innerHTML = `
      <span class="doc-icon">${docIcon(doc.filename)}</span>
      <div class="doc-info">
        <div class="doc-name">${esc(doc.filename)}</div>
        <div class="doc-meta">${doc.chunk_count.toLocaleString()} chunks · ${formatDate(doc.upload_time)}</div>
      </div>
      <button class="doc-delete-btn" title="Remove document" data-filename="${esc(doc.filename)}">✕</button>
    `;
    row.querySelector('.doc-delete-btn').addEventListener('click', () => removeDocument(doc.filename));
    list.appendChild(row);
  }

  // Large notebook notice
  const totalChunks = state.documents.reduce((s, d) => s + d.chunk_count, 0);
  document.getElementById('large-notebook-notice').classList.toggle('hidden', totalChunks < LARGE_CHUNK_THRESHOLD);

  // Re-render train section now that doc count is known (skip if training is running
  // to avoid hiding the progress bar mid-train)
  if (!trainingInProgress) checkTrainingStatus();
}

async function removeDocument(filename) {
  await api('DELETE', '/api/notebooks/' + state.activeId + '/documents/' + encodeURIComponent(filename));
  await loadDocuments();
}

function docIcon(filename) {
  const ext = filename.split('.').pop().toLowerCase();
  return ext === 'pdf' ? '📕' : ext === 'md' ? '📝' : '📄';
}

// ── Upload ────────────────────────────────────────────────────────────────────
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');

uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  handleFiles(Array.from(e.dataTransfer.files));
});
fileInput.addEventListener('change', () => {
  handleFiles(Array.from(fileInput.files));
  fileInput.value = '';
});

function handleFiles(files) {
  for (const file of files) {
    uploadFile(file);
  }
}

async function uploadFile(file) {
  const queueEl = document.getElementById('upload-queue');
  const itemId = 'upload-' + Date.now() + '-' + Math.random().toString(36).slice(2);

  const item = document.createElement('div');
  item.className = 'upload-item';
  item.id = itemId;
  item.innerHTML = `
    <div class="upload-item-header">
      <span class="upload-item-name">${esc(file.name)}</span>
      <span class="upload-item-status">Preparing…</span>
    </div>
    <div class="progress-track"><div class="progress-fill" style="width:0%"></div></div>
  `;
  queueEl.appendChild(item);

  const statusEl = item.querySelector('.upload-item-status');
  const fill = item.querySelector('.progress-fill');

  try {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(
      API + '/api/notebooks/' + state.activeId + '/upload',
      { method: 'POST', body: formData }
    );

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n\n');
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const data = JSON.parse(line.slice(6));
          if (data.type === 'progress') {
            const pct = data.total > 0 ? Math.round((data.current / data.total) * 100) : 0;
            fill.style.width = pct + '%';
            statusEl.textContent = `Embedding chunk ${data.current} / ${data.total}`;
          } else if (data.type === 'done') {
            fill.style.width = '100%';
            statusEl.textContent = `Done — ${data.chunks.toLocaleString()} chunks`;
          } else if (data.type === 'error') {
            throw new Error(data.message);
          }
        } catch (parseErr) {
          if (parseErr.message !== 'Unexpected end of JSON input') throw parseErr;
        }
      }
    }

    await sleep(1500);
    item.remove();
    await loadDocuments();
  } catch (err) {
    statusEl.textContent = '⚠ ' + err.message;
    statusEl.style.color = 'var(--danger)';
    fill.style.background = 'var(--danger)';
  }
}

// ── Chat ──────────────────────────────────────────────────────────────────────
async function loadChatHistory() {
  state.chatMessages = await api('GET', '/api/notebooks/' + state.activeId + '/chat');
  renderChat();
}

function renderChat() {
  const el = document.getElementById('chat-messages');
  el.innerHTML = '';

  if (state.chatMessages.length === 0) {
    el.innerHTML = `
      <div class="chat-empty">
        <div class="chat-empty-icon">💬</div>
        <p style="font-size:14px;">Ask a question about your documents</p>
      </div>
    `;
    return;
  }

  for (const msg of state.chatMessages) {
    el.appendChild(renderMessage(msg));
  }
  scrollChatToBottom();
}

function renderMessage(msg) {
  const div = document.createElement('div');
  div.className = 'msg msg-' + msg.role;

  let citationsHtml = '';
  if (msg.citations && msg.citations.length > 0) {
    const tags = msg.citations.map(c =>
      `<span class="source-tag">${esc(c.filename)} · chunk ${c.chunk_index}</span>`
    ).join('');
    citationsHtml = `<div class="msg-sources"><strong>Sources:</strong> ${tags}</div>`;
  }

  div.innerHTML = `
    <div class="msg-bubble">
      ${renderMarkdown(msg.message)}
      ${citationsHtml}
    </div>
  `;
  return div;
}

function renderMarkdown(text) {
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');
}

const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');

chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
});
sendBtn.addEventListener('click', sendMessage);

async function sendMessage() {
  if (state.chatPending) return;
  const question = chatInput.value.trim();
  if (!question || !state.activeId) return;

  chatInput.value = '';
  chatInput.style.height = 'auto';

  const userMsg = { role: 'user', message: question, citations: [] };
  state.chatMessages.push(userMsg);
  const el = document.getElementById('chat-messages');

  // Remove empty state if present
  const emptyEl = el.querySelector('.chat-empty');
  if (emptyEl) emptyEl.remove();

  el.appendChild(renderMessage(userMsg));

  const thinkingDiv = document.createElement('div');
  thinkingDiv.className = 'msg msg-assistant thinking-msg';
  thinkingDiv.innerHTML = '<div class="msg-bubble">Thinking…</div>';
  el.appendChild(thinkingDiv);
  scrollChatToBottom();

  state.chatPending = true;
  sendBtn.disabled = true;

  try {
    const response = await api('POST', '/api/notebooks/' + state.activeId + '/chat', { question });
    thinkingDiv.remove();
    state.chatMessages.push(response);
    el.appendChild(renderMessage(response));
    scrollChatToBottom();
    if (response.used_adapter) updateModelModeBar(true);
  } catch (err) {
    thinkingDiv.remove();
    const errMsg = { role: 'assistant', message: '⚠ Error: ' + err.message, citations: [] };
    state.chatMessages.push(errMsg);
    el.appendChild(renderMessage(errMsg));
    scrollChatToBottom();
  } finally {
    state.chatPending = false;
    sendBtn.disabled = false;
    chatInput.focus();
  }
}

function scrollChatToBottom() {
  const el = document.getElementById('chat-messages');
  requestAnimationFrame(() => { el.scrollTop = el.scrollHeight; });
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function show(screenId) {
  ['loading-screen', 'setup-screen', 'main-app'].forEach(id => {
    document.getElementById(id).classList.toggle('hidden', id !== screenId);
  });
  // loading-screen is display:flex via CSS; force the inline style so it overrides
  // any previous inline display:none set on it by a prior show() call.
  const loading = document.getElementById('loading-screen');
  loading.style.display = screenId === 'loading-screen' ? 'flex' : 'none';
}

function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatDate(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    const now = new Date();
    const diffMs = now - d;
    const diffMins = Math.floor(diffMs / 60000);
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHrs = Math.floor(diffMins / 60);
    if (diffHrs < 24) return `${diffHrs}h ago`;
    const diffDays = Math.floor(diffHrs / 24);
    if (diffDays < 7) return `${diffDays}d ago`;
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch (_) { return ''; }
}

// ── Boot ──────────────────────────────────────────────────────────────────────
runSetupFlow().catch(console.error);

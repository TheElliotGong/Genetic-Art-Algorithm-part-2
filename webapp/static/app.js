/* Front-end for the Voronoi genetic art web interface.
 *
 * Three jobs: pick a target image, collect hyperparameters, and follow a run.
 * Progress arrives over server-sent events, with polling as a fallback for
 * proxies that buffer streamed responses.
 */

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const POLL_INTERVAL_MS = 700;

// Widest the comparison viewer is allowed to get, matching the CSS cap.
const VIEWER_MAX_WIDTH = 620;

const PRESETS = {
  quick: {
    generations: 200, population_size: 40, num_points: 120,
    max_dimension: 260, render_scale: 2, preview_every: 5,
  },
  balanced: {},           // filled from the server defaults at startup
  detailed: {
    generations: 2500, population_size: 120, num_points: 400,
    max_dimension: 640, render_scale: 4, preview_every: 25,
  },
};

const state = {
  imageId: null,
  targetUrl: null,
  jobId: null,
  source: null,
  previewVersion: -1,
  targetLoaded: false,
  eventSource: null,
  pollTimer: null,
  tickTimer: null,
  lastSnapshot: null,
};

/* ------------------------------------------------------------------ utils */

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || !isFinite(seconds)) return '—';
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours) return `${hours}h ${String(minutes).padStart(2, '0')}m`;
  if (minutes) return `${minutes}m ${String(secs).padStart(2, '0')}s`;
  return `${secs}s`;
}

function formatNumber(value) {
  if (value === null || value === undefined) return '—';
  return Math.round(value).toLocaleString();
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  let payload = null;
  try {
    payload = await response.json();
  } catch (error) {
    payload = null;
  }
  if (!response.ok) {
    const detail = payload && payload.detail;
    throw new Error(typeof detail === 'string' ? detail : `Request failed (${response.status})`);
  }
  return payload;
}

function showError(element, message) {
  element.textContent = message || '';
  element.hidden = !message;
}

/* --------------------------------------------------------- form plumbing */

/** Keep a range input, its mirrored number box and the track fill in sync. */
function syncControl(input) {
  const mirror = document.querySelector(`[data-mirror="${input.dataset.param}"]`);
  if (mirror && mirror !== document.activeElement) mirror.value = input.value;
  if (input.type === 'range') {
    const min = Number(input.min);
    const max = Number(input.max);
    const fill = max > min ? ((Number(input.value) - min) / (max - min)) * 100 : 0;
    input.style.setProperty('--fill', `${fill}%`);
  }
}

function setParam(name, value) {
  const input = document.querySelector(`[data-param="${name}"]`);
  if (!input) return;
  if (input.type === 'checkbox') input.checked = Boolean(value);
  else input.value = value;
  syncControl(input);
}

function collectParams() {
  const params = {};
  $$('[data-param]').forEach((input) => {
    const name = input.dataset.param;
    if (input.type === 'checkbox') params[name] = input.checked;
    else if (input.type === 'range' || input.type === 'number') params[name] = Number(input.value);
    else params[name] = input.value;
  });
  return params;
}

function applyPreset(name) {
  const preset = { ...PRESETS.balanced, ...(PRESETS[name] || {}) };
  Object.entries(preset).forEach(([key, value]) => setParam(key, value));
  $$('.chip[data-preset]').forEach((chip) => chip.classList.toggle('is-active', chip.dataset.preset === name));
  updateRunSummary();
}

/** Reflect mode/outline choices in which control groups are visible. */
function updateConditionalGroups() {
  $('#tiling-group').hidden = $('[data-param="mode"]').value !== 'tiled';
  $('#outline-options').style.display = $('[data-param="outline"]').checked ? '' : 'none';
}

function updateRunSummary() {
  const params = collectParams();
  const tiles = params.mode === 'tiled' ? params.tile_rows * params.tile_cols : 1;
  const perTarget = params.generations + (params.genome_duplication ? 1 : 0);
  const total = perTarget * tiles;
  const evaluations = total * params.population_size;
  $('#run-summary').textContent =
    `${total.toLocaleString()} generations` +
    (tiles > 1 ? ` across ${tiles} tiles` : '') +
    ` · ${evaluations.toLocaleString()} renders`;
}

/* --------------------------------------------------------- target picker */

function setTarget(descriptor) {
  state.imageId = descriptor.image_id;
  state.targetUrl = descriptor.url;
  state.source = descriptor.label;
  $('#target-thumb').src = descriptor.url;
  $('#target-label').textContent = descriptor.label;
  $('#target-dims').textContent = `${descriptor.width} × ${descriptor.height} px`;
  $('#target-card').hidden = false;
  $('#run-btn').disabled = false;
  showError($('#source-error'), null);
}

function clearTarget() {
  state.imageId = null;
  state.targetUrl = null;
  $('#target-card').hidden = true;
  $('#run-btn').disabled = true;
}

async function uploadFile(file) {
  showError($('#source-error'), null);
  const body = new FormData();
  body.append('file', file);
  body.append('label', file.name);
  try {
    setTarget(await requestJson('/api/uploads', { method: 'POST', body }));
  } catch (error) {
    showError($('#source-error'), error.message);
  }
}

async function loadFromUrl() {
  const url = $('#url-input').value.trim();
  if (!url) return;
  showError($('#source-error'), null);
  $('#url-load').disabled = true;
  $('#url-load').textContent = 'Fetching…';
  try {
    setTarget(await requestJson('/api/uploads/url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    }));
  } catch (error) {
    showError($('#source-error'), error.message);
  } finally {
    $('#url-load').disabled = false;
    $('#url-load').textContent = 'Fetch';
  }
}

async function loadSamples() {
  try {
    const { samples } = await requestJson('/api/samples');
    const grid = $('#sample-grid');
    grid.innerHTML = '';
    if (!samples.length) {
      grid.innerHTML = '<p class="hint">No sample images found in <code>img/</code>.</p>';
      return;
    }
    samples.forEach(({ name }) => {
      const button = document.createElement('button');
      button.className = 'sample';
      button.title = name;
      button.innerHTML = `<img alt="${name}" src="/api/samples/${encodeURIComponent(name)}">`;
      button.addEventListener('click', async () => {
        try {
          setTarget(await requestJson('/api/uploads/sample', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
          }));
        } catch (error) {
          showError($('#source-error'), error.message);
        }
      });
      grid.appendChild(button);
    });
  } catch (error) {
    /* Samples are a convenience; a failure here should not block the app. */
  }
}

/* ------------------------------------------------------------ run control */

async function startRun() {
  if (!state.imageId) return;
  showError($('#run-error'), null);
  $('#run-btn').disabled = true;
  try {
    const job = await requestJson('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_id: state.imageId, ...collectParams() }),
    });
    attachToJob(job.id);
    render(job);
    refreshHistory();
  } catch (error) {
    showError($('#run-error'), error.message);
  } finally {
    $('#run-btn').disabled = !state.imageId;
  }
}

async function cancelRun() {
  if (!state.jobId) return;
  $('#cancel-btn').disabled = true;
  try {
    await requestJson(`/api/jobs/${state.jobId}/cancel`, { method: 'POST' });
  } catch (error) {
    showError($('#run-error'), error.message);
  }
}

function detach() {
  if (state.eventSource) { state.eventSource.close(); state.eventSource = null; }
  if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
  if (state.tickTimer) { clearInterval(state.tickTimer); state.tickTimer = null; }
}

function attachToJob(jobId) {
  detach();
  state.jobId = jobId;
  state.previewVersion = -1;
  state.targetLoaded = false;
  $('#idle-state').hidden = true;
  $('#run-state').hidden = false;
  $('#result-bar').hidden = true;
  $('#viewer-empty').hidden = false;
  $('#viewer-preview').removeAttribute('src');
  // The scaled target only exists once the worker picks the job up, so the
  // image is wired in from the first snapshot that reports it rather than here.
  $('#viewer-target').removeAttribute('src');

  const source = new EventSource(`/api/jobs/${jobId}/events`);
  state.eventSource = source;
  source.onmessage = (event) => {
    const snapshot = JSON.parse(event.data);
    render(snapshot);
    if (['done', 'error', 'cancelled'].includes(snapshot.status)) {
      detach();
      refreshHistory();
    }
  };
  source.onerror = () => {
    // A closed stream after a terminal status is expected; otherwise fall back
    // to polling so a buffering proxy cannot leave the UI frozen.
    source.close();
    state.eventSource = null;
    if (!state.lastSnapshot || !['done', 'error', 'cancelled'].includes(state.lastSnapshot.status)) {
      startPolling(jobId);
    }
  };

  // The server only pushes on generation boundaries, which can be slow for big
  // populations; this keeps the elapsed clock moving in between.
  state.tickTimer = setInterval(tickElapsed, 1000);
}

function startPolling(jobId) {
  if (state.pollTimer) return;
  state.pollTimer = setInterval(async () => {
    try {
      const snapshot = await requestJson(`/api/jobs/${jobId}`);
      render(snapshot);
      if (['done', 'error', 'cancelled'].includes(snapshot.status)) {
        detach();
        refreshHistory();
      }
    } catch (error) {
      detach();
    }
  }, POLL_INTERVAL_MS);
}

function tickElapsed() {
  const snapshot = state.lastSnapshot;
  if (!snapshot || snapshot.status !== 'running') return;
  snapshot.elapsed_seconds += 1;
  $('#stat-elapsed').textContent = formatDuration(snapshot.elapsed_seconds);
  if (snapshot.eta_seconds !== null && snapshot.eta_seconds !== undefined) {
    snapshot.eta_seconds = Math.max(0, snapshot.eta_seconds - 1);
    $('#stat-eta').textContent = formatDuration(snapshot.eta_seconds);
  }
}

/* ----------------------------------------------------------- rendering UI */

function render(snapshot) {
  state.lastSnapshot = snapshot;

  const pill = $('#status-pill');
  pill.textContent = snapshot.status;
  pill.dataset.status = snapshot.status;

  let stage = snapshot.stage || '';
  if (snapshot.status === 'queued' && snapshot.queue_position) {
    stage = `Queued — position ${snapshot.queue_position}`;
  }
  $('#stage-text').textContent = stage;

  const percent = (snapshot.progress || 0) * 100;
  $('#progress-fill').style.width = `${percent.toFixed(2)}%`;
  $('#progress-percent').textContent = `${percent.toFixed(1)}%`;
  $('#progress-gens').textContent =
    `generation ${snapshot.generation.toLocaleString()} / ${snapshot.total_generations.toLocaleString()}`;
  const track = $('#progress-bar-root');
  track.classList.toggle('is-active', snapshot.status === 'running');
  track.setAttribute('aria-valuenow', percent.toFixed(0));

  renderTileTicks(snapshot);

  $('#stat-elapsed').textContent = formatDuration(snapshot.elapsed_seconds);
  $('#stat-eta').textContent = snapshot.status === 'running' ? formatDuration(snapshot.eta_seconds) : '—';
  $('#stat-similarity').textContent =
    snapshot.similarity === null || snapshot.similarity === undefined
      ? '—'
      : `${(snapshot.similarity * 100).toFixed(2)}%`;
  $('#stat-points').textContent = formatNumber(snapshot.num_points);
  $('#stat-best').textContent = formatNumber(snapshot.best_fitness);
  $('#stat-tile-wrap').hidden = snapshot.tile_count <= 1;
  $('#stat-tile').textContent = `${snapshot.tile_index + 1} / ${snapshot.tile_count}`;

  $('#cancel-btn').disabled = !['queued', 'running'].includes(snapshot.status);
  $('#cancel-btn').textContent = snapshot.status === 'running' ? 'Stop' : 'Stopped';

  if (snapshot.has_target && !state.targetLoaded) {
    state.targetLoaded = true;
    $('#viewer-target').src = `/api/jobs/${snapshot.id}/target.png`;
  }

  if (snapshot.has_preview && snapshot.preview_version !== state.previewVersion) {
    state.previewVersion = snapshot.preview_version;
    $('#viewer-preview').src = `/api/jobs/${snapshot.id}/preview.png?v=${snapshot.preview_version}`;
    $('#viewer-empty').hidden = true;
  }

  renderChart(snapshot.history || []);

  const finished = snapshot.status === 'done' && snapshot.has_result;
  $('#result-bar').hidden = !finished;
  if (finished) {
    $('#download-btn').href = `/api/jobs/${snapshot.id}/result.png?download=true`;
    $('#result-note').textContent =
      `Rendered at ${snapshot.params.render_scale}× in ${formatDuration(snapshot.elapsed_seconds)}.`;
  }

  showError($('#run-error'), snapshot.status === 'error' ? snapshot.error : null);
}

/** Size the comparison frame to the image's aspect ratio.
 *
 * Both images are absolutely positioned so that the split wipe cannot shift
 * anything, which means the frame carries the geometry. Its height is bounded
 * here rather than in CSS, because `aspect-ratio` plus `max-height` clamps the
 * height without shrinking the width - which would stretch the painting.
 */
function fitViewer() {
  const preview = $('#viewer-preview');
  const target = $('#viewer-target');
  const image = preview.naturalWidth ? preview : target;
  if (!image.naturalWidth || !image.naturalHeight) return;

  const aspect = image.naturalWidth / image.naturalHeight;
  const maxHeight = Math.max(240, window.innerHeight * 0.62);
  const inner = $('#viewer-inner');
  inner.style.aspectRatio = `${image.naturalWidth} / ${image.naturalHeight}`;
  inner.style.maxWidth = `${Math.min(VIEWER_MAX_WIDTH, maxHeight * aspect)}px`;
}

/** Draw a divider on the progress bar at each tile boundary. */
function renderTileTicks(snapshot) {
  const container = $('#tile-ticks');
  const wanted = snapshot.tile_count > 1 ? snapshot.tile_count - 1 : 0;
  if (container.childElementCount === wanted) return;
  container.innerHTML = '';
  for (let i = 1; i <= wanted; i += 1) {
    const tick = document.createElement('i');
    tick.style.left = `${(i / snapshot.tile_count) * 100}%`;
    container.appendChild(tick);
  }
}

function renderChart(history) {
  const line = $('#chart-line');
  const area = $('#chart-area');
  if (history.length < 2) {
    line.setAttribute('points', '');
    area.setAttribute('points', '');
    $('#chart-min').textContent = '—';
    $('#chart-max').textContent = '—';
    return;
  }

  const width = 600;
  const height = 140;
  const padding = 6;
  const values = history.map((point) => point.similarity);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const span = Math.max(1e-9, maxValue - minValue);
  const lastGeneration = history[history.length - 1].generation || 1;

  const points = history.map((point) => {
    const x = (point.generation / lastGeneration) * (width - padding * 2) + padding;
    const y = height - padding - ((point.similarity - minValue) / span) * (height - padding * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  line.setAttribute('points', points.join(' '));
  area.setAttribute('points', `${padding},${height} ${points.join(' ')} ${width - padding},${height}`);
  $('#chart-min').textContent = `${(minValue * 100).toFixed(2)}%`;
  $('#chart-max').textContent = `${(maxValue * 100).toFixed(2)}% best`;
}

/* -------------------------------------------------------------- history */

async function refreshHistory() {
  try {
    const { jobs } = await requestJson('/api/jobs');
    const list = $('#history-list');
    list.innerHTML = '';
    if (!jobs.length) {
      list.innerHTML = '<li class="history-empty">Nothing yet.</li>';
      return;
    }
    jobs.slice(0, 12).forEach((job) => {
      const item = document.createElement('li');
      item.className = 'history-item' + (job.id === state.jobId ? ' is-active' : '');
      const similarity = job.similarity === null || job.similarity === undefined
        ? '—'
        : `${(job.similarity * 100).toFixed(1)}%`;
      item.innerHTML =
        `<span class="pill" data-status="${job.status}">${job.status}</span>` +
        `<span>${job.params.mode} · ${job.params.num_points} pts · ${job.generation.toLocaleString()} gens</span>` +
        `<span class="meta">${similarity} · ${formatDuration(job.elapsed_seconds)}</span>`;
      item.addEventListener('click', () => {
        attachToJob(job.id);
        render(job);
        if (['done', 'error', 'cancelled'].includes(job.status)) detach();
      });
      list.appendChild(item);
    });
  } catch (error) {
    /* Non-critical. */
  }
}

/* ----------------------------------------------------------------- setup */

async function loadDefaults() {
  try {
    const { fields } = await requestJson('/api/schema');
    Object.entries(fields).forEach(([name, meta]) => {
      PRESETS.balanced[name] = meta.default;
      const input = document.querySelector(`[data-param="${name}"]`);
      if (!input) return;
      // Server bounds win over the markup's, so the two can never drift apart.
      if (meta.ge !== undefined && input.type !== 'checkbox') input.min = meta.ge;
      if (meta.le !== undefined && input.type !== 'checkbox') input.max = meta.le;
      const mirror = document.querySelector(`[data-mirror="${name}"]`);
      if (mirror) {
        if (meta.ge !== undefined) mirror.min = meta.ge;
        if (meta.le !== undefined) mirror.max = meta.le;
      }
      if (meta.description) input.title = meta.description;
      setParam(name, meta.default);
    });
  } catch (error) {
    showError($('#run-error'), `Could not load defaults: ${error.message}`);
  }
  updateConditionalGroups();
  updateRunSummary();
}

function wireEvents() {
  $$('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      $$('.tab').forEach((other) => other.classList.toggle('is-active', other === tab));
      $$('.source-pane').forEach((pane) => {
        pane.classList.toggle('is-active', pane.dataset.pane === tab.dataset.source);
      });
    });
  });

  $('#file-input').addEventListener('change', (event) => {
    if (event.target.files[0]) uploadFile(event.target.files[0]);
  });

  const dropzone = $('#dropzone');
  ['dragenter', 'dragover'].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.add('is-hot');
    });
  });
  ['dragleave', 'drop'].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove('is-hot');
    });
  });
  dropzone.addEventListener('drop', (event) => {
    const file = event.dataTransfer.files[0];
    if (file) uploadFile(file);
  });

  $('#url-load').addEventListener('click', loadFromUrl);
  $('#url-input').addEventListener('keydown', (event) => {
    if (event.key === 'Enter') { event.preventDefault(); loadFromUrl(); }
  });
  $('#target-clear').addEventListener('click', clearTarget);

  $$('[data-param]').forEach((input) => {
    input.addEventListener('input', () => {
      syncControl(input);
      updateConditionalGroups();
      updateRunSummary();
      $$('.chip[data-preset]').forEach((chip) => chip.classList.remove('is-active'));
    });
  });

  $$('[data-mirror]').forEach((mirror) => {
    mirror.addEventListener('change', () => {
      const input = document.querySelector(`[data-param="${mirror.dataset.mirror}"]`);
      if (!input) return;
      const value = Math.min(Number(input.max), Math.max(Number(input.min), Number(mirror.value)));
      mirror.value = value;
      input.value = value;
      syncControl(input);
      updateRunSummary();
    });
  });

  $$('.chip[data-preset]').forEach((chip) => {
    chip.addEventListener('click', () => applyPreset(chip.dataset.preset));
  });

  $('#run-btn').addEventListener('click', startRun);
  $('#cancel-btn').addEventListener('click', cancelRun);

  $('#split-range').addEventListener('input', (event) => {
    const value = Number(event.target.value);
    $('#viewer-preview').style.clipPath = `inset(0 ${100 - value}% 0 0)`;
  });

  $('#viewer-target').addEventListener('load', fitViewer);
  $('#viewer-preview').addEventListener('load', fitViewer);
  window.addEventListener('resize', fitViewer);
}

wireEvents();
loadDefaults();
loadSamples();
refreshHistory();

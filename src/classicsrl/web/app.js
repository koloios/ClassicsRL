import { pongScores, pongRenderer } from './pong.js';

const $ = (id) => document.getElementById(id);
const number = (value, digits = 0) => Number.isFinite(value)
  ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : '—';
const compact = (value) => Number.isFinite(value)
  ? Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(value) : '—';
const text = (id, value) => { $(id).textContent = value; };
function element(tag, content, className) {
  const node = document.createElement(tag);
  if (content !== undefined) node.textContent = content;
  if (className) node.className = className;
  return node;
}
const state = { runs: [], id: null, data: null, checkpoint: null, evaluation: null,
  episode: 0, decisions: [], scores: [], appearance: 'minimal', recording: null,
  frame: -1, generation: 0, busy: false };
const video = $('replay');
const paintPong = pongRenderer(video, $('pong-court'));
const metricLabels = {
  'eval/mean_return': 'Evaluation return', 'train/episode_return': 'Episode return',
  'train/td_loss': 'TD loss', 'train/q_mean': 'Mean Q-value', 'train/epsilon': 'Exploration ε',
  'train/steps_per_second': 'Training throughput', 'train/grad_norm': 'Gradient norm',
  'train/episode_length': 'Episode length',
  'train/features/conv3_active_fraction': 'Conv3 active fraction',
  'train/features/conv3_zero_fraction': 'Conv3 zero-output fraction',
  'train/q/observation_std': 'Q-value variation across observations',
  'train/q/greedy_action_count': 'Distinct greedy actions',
};

async function request(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Could not read experiment data (${response.status}).`);
  return response.json();
}
function options(select, items, selected) {
  const previous = [...select.options].map((option) => [option.value, option.textContent]);
  if (JSON.stringify(previous) === JSON.stringify(items)) {
    if (selected !== undefined) select.value = selected;
    return;
  }
  select.replaceChildren(...items.map(([value, label]) => {
    const option = element('option', label);
    option.value = value;
    return option;
  }));
  if (items.some(([value]) => String(value) === String(selected))) select.value = selected;
}
function setView(view) {
  document.body.dataset.view = view;
  $('configuration').hidden = view !== 'configuration';
  document.querySelectorAll('.view-tabs button').forEach((button) => {
    button.setAttribute('aria-selected', button.dataset.view === view);
  });
  if (view === 'overview') drawChart();
}

function renderRuns() {
  text('run-count', state.runs.length);
  const query = $('run-search').value.trim().toLowerCase();
  const runs = state.runs.filter((run) => `${run.id} ${run.env_id}`.toLowerCase().includes(query));
  $('run-list').replaceChildren(...runs.map((run) => {
    const button = element('button', undefined, `run-button ${run.id === state.id ? 'selected' : ''}`);
    button.setAttribute('aria-current', run.id === state.id ? 'true' : 'false');
    button.setAttribute('aria-pressed', run.id === state.id ? 'true' : 'false');
    const heading = element('span', undefined, 'run-row');
    heading.append(element('span', String(run.env_id).replace('ALE/', '').replace('-v5', ''), 'run-game'),
      element('span', compact(run.step), 'run-step'));
    const name = element('span', run.id, 'run-name');
    name.title = run.id;
    button.append(heading, name);
    button.addEventListener('click', () => selectRun(run.id));
    return button;
  }));
  if (!runs.length) $('run-list').append(element('p', 'No matching experiments.', 'sidebar-empty'));
}

async function selectRun(id) {
  state.id = id;
  state.data = null;
  state.checkpoint = null;
  state.evaluation = null;
  state.recording = null;
  state.decisions = [];
  state.scores = [];
  video.pause();
  video.removeAttribute('src');
  const url = new URL(location.href);
  url.searchParams.set('run', id);
  history.replaceState(null, '', url);
  renderRuns();
  await refreshDetail();
}
async function refreshDetail() {
  const id = state.id;
  const generation = ++state.generation;
  try {
    const data = await request(`/api/runs/${encodeURIComponent(id)}`);
    if (generation !== state.generation || id !== state.id) return;
    state.data = data;
    renderExperiment();
    $('connection-error').hidden = true;
  } catch (error) { showError(error); }
}
function showError(error) {
  text('connection-error', `${error.message} Check that the local dashboard server is running.`);
  $('connection-error').hidden = false;
}
async function refresh() {
  if (state.busy) return;
  state.busy = true;
  try {
    const result = await request('/api/runs');
    state.runs = result.runs;
    $('welcome').hidden = Boolean(state.runs.length);
    $('experiment').hidden = !state.runs.length;
    if (!state.runs.length) { renderRuns(); return; }
    const wanted = state.id ?? new URL(location.href).searchParams.get('run');
    const id = state.runs.some((run) => run.id === wanted) ? wanted : state.runs[0].id;
    if (state.id !== id) await selectRun(id);
    else { renderRuns(); await refreshDetail(); }
  } catch (error) { showError(error); }
  finally { state.busy = false; }
}

function renderExperiment() {
  const { summary: run, metrics, config, warnings } = state.data;
  text('breadcrumb', run.id);
  text('game-title', String(run.env_id).replace('ALE/', '').replace('-v5', ''));
  text('run-subtitle', run.id);
  text('run-status', { complete: 'Finished', active: 'Training', paused: 'Paused', incomplete: 'No recent updates' }[run.status]);
  $('run-status').className = `badge ${run.status}`;
  text('device-badge', (run.device ?? 'Unknown device').toUpperCase());
  text('seed-badge', `Seed ${run.seed ?? '—'}`);
  text('stat-steps', number(run.step));
  text('stat-budget', `${number(run.target_steps)} target steps`);
  $('step-progress').style.width = `${Math.min(100, Math.max(0, run.step / run.target_steps * 100 || 0))}%`;
  text('stat-return', number(run.best_evaluation, 2));
  text('stat-return-note', run.best_evaluation === null ? 'No validation yet' : 'Greedy validation · full protocol');
  text('stat-updates', number(run.updates));
  text('stat-training-time', Number.isFinite(run.seconds) ? `${number(run.seconds / 60, 1)} min training time` : 'Training time unavailable');
  const speed = (metrics['train/steps_per_second'] ?? []).filter((row) => !state.data.segments.length
    || row.segment === state.data.segments.at(-1).segment_id).at(-1);
  text('stat-speed', number(speed?.value, 1));
  text('last-updated', `Read ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`);
  text('data-warning', warnings.join(' '));
  $('data-warning').hidden = !warnings.length;
  const metric = $('metric-select').value;
  const available = Object.keys(metricLabels).filter((key) => key === 'eval/mean_return'
    ? state.data.evaluations.some((entry) => entry.compatible_for_learning
      && entry.evaluation_role === 'validation' && Number.isFinite(entry.agent_steps))
    : metrics[key]?.length);
  options($('metric-select'), available.map((key) => [key, metricLabels[key]]), metric);
  drawChart();
  renderEvaluations();
  renderConfiguration(config);
  renderInspector();
}

function svgNode(tag, attributes = {}, content) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
  if (content !== undefined) node.textContent = content;
  return node;
}
function drawChart() {
  if (!state.data) return;
  const key = $('metric-select').value;
  const evaluation = key === 'eval/mean_return';
  const points = evaluation ? state.data.evaluations.filter((entry) => entry.compatible_for_learning
    && entry.evaluation_role === 'validation' && Number.isFinite(entry.agent_steps)).map((entry) => ({ step: entry.agent_steps,
      value: entry.mean_return, segment: `${entry.policy}:${entry.checkpoint.replace(/step-\d+/, 'step')}`,
      policy: entry.policy })) : state.data.metrics[key] ?? [];
  text('chart-description', evaluation ? 'Mean return · repeated full-game validation episodes'
    : 'Raw observations · each continuation shown separately');
  text('chart-legend', evaluation ? 'Validation only · greedy (blue) · random (gray)' : 'Training observations · no smoothing');
  $('chart').replaceChildren();
  if (!points.length) { $('chart').append(element('p', 'Metrics will appear as training progresses.', 'chart-empty')); return; }
  const width = 660, height = 260, left = 56, right = 18, top = 18, bottom = 36;
  const maxX = Math.max(1, ...points.map((point) => point.step));
  let minY = Math.min(...points.map((point) => point.value));
  let maxY = Math.max(...points.map((point) => point.value));
  const pad = maxY === minY ? Math.max(1, Math.abs(maxY) * .05) : (maxY - minY) * .12;
  minY -= pad; maxY += pad;
  const x = (value) => left + value / maxX * (width - left - right);
  const y = (value) => top + (maxY - value) / (maxY - minY) * (height - top - bottom);
  const svg = svgNode('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img',
    'aria-label': `${metricLabels[key] ?? key} over agent steps` });
  const tickLabel = (value) => value !== 0 && Math.abs(value) < .01 ? value.toExponential(1) : number(value, 2);
  for (let index = 0; index < 5; index++) {
    const value = minY + (maxY - minY) * index / 4;
    svg.append(svgNode('line', { x1: left, x2: width - right, y1: y(value), y2: y(value), stroke: '#edf0f5' }),
      svgNode('text', { x: left - 10, y: y(value) + 4, 'text-anchor': 'end', fill: '#7a8595', 'font-size': 11 }, tickLabel(value)),
      svgNode('text', { x: x(maxX * index / 4), y: height - 10, 'text-anchor': 'middle', fill: '#7a8595', 'font-size': 11 }, compact(maxX * index / 4)));
  }
  const groups = Map.groupBy ? Map.groupBy(points, (point) => point.segment)
    : points.reduce((map, point) => map.set(point.segment, [...(map.get(point.segment) ?? []), point]), new Map());
  for (const group of groups.values()) {
    const color = group[0].policy === 'random' ? '#9ca7b8' : '#3868d9';
    svg.append(svgNode('polyline', { points: group.map((point) => `${x(point.step)},${y(point.value)}`).join(' '),
      fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linejoin': 'round' }));
    for (const point of group) {
      const dot = svgNode('circle', { cx: x(point.step), cy: y(point.value), r: evaluation ? 4 : 2, fill: color });
      dot.append(svgNode('title', {}, `${number(point.step)} steps · ${number(point.value, 4)}`));
      svg.append(dot);
    }
  }
  $('chart').append(svg);
}

function renderEvaluations() {
  const entries = [...state.data.evaluations].reverse();
  text('evaluation-count', entries.length);
  $('evaluation-rows').replaceChildren(...entries.map((entry) => {
    const row = element('tr');
    row.append(element('td', compact(entry.agent_steps ?? state.data.checkpoints.find((cp) => cp.name === entry.checkpoint)?.step)),
      element('td', entry.policy), element('td', number(entry.mean_return, 2), 'score'), element('td', entry.episodes.length),
      element('td', Number.isFinite(entry.win_rate) ? `${number(entry.win_rate * 100, 0)}% · ${entry.full_game_wins}/${entry.episodes.length}` : '—'));
    const role = { validation: 'Validation', heldout: 'Final test', demonstration: 'Demo', recording: 'Recording' }[entry.evaluation_role] ?? entry.evaluation_role;
    row.children[1].append(element('span', role, 'preview-label'));
    if (entry.truncations) row.children[4].append(element('span', `${entry.truncations} time-limited`, 'preview-label'));
    if (!entry.compatible_for_learning) row.children[1].append(element('span', 'Preview / other protocol', 'preview-label'));
    const cell = element('td');
    const button = element('button', entry.has_video ? 'Inspect ↗' : 'Record ↗', 'table-button');
    button.addEventListener('click', () => {
      state.checkpoint = entry.checkpoint; state.evaluation = entry.id; state.episode = 0;
      renderInspector();
      if (entry.has_video) setView('inspector'); else recordDialog();
    });
    cell.append(button); row.append(cell);
    return row;
  }));
  if (!entries.length) {
    const cell = element('td', 'No evaluations yet. Training logs remain available above.', 'no-evaluations');
    cell.colSpan = 6;
    const row = element('tr'); row.append(cell); $('evaluation-rows').append(row);
  }
}

function renderConfiguration(config) {
  $('config-table').replaceChildren(...Object.entries(config).map(([key, value]) => {
    const row = element('div', undefined, 'config-row');
    row.append(element('span', key), element('code', String(value)));
    return row;
  }));
  const meta = state.data.metadata;
  const latest = [...state.data.segments].reverse().find((segment) => segment.event === 'start');
  const source = latest?.runtime ?? meta;
  const facts = { 'Run ID': meta.run_id, 'Git revision': source.revision ?? meta.revision,
    'Working tree': (source.dirty ?? meta.dirty) ? 'Uncommitted changes recorded' : 'Clean',
    'Python': source.python ?? meta.python, 'Network': meta.network, 'Loss': meta.loss,
    'Resume behavior': meta.resume_mode, 'Created': new Date(meta.created_at).toLocaleString(),
    ...Object.fromEntries(Object.entries(source.dependencies ?? meta.dependencies ?? {}).map(([key, value]) => [key, value])) };
  $('provenance').replaceChildren(...Object.entries(facts).flatMap(([key, value]) => [element('dt', key), element('dd', value ?? 'Unavailable')]));
}

function selectedEvaluation() { return state.data?.evaluations.find((entry) => entry.id === state.evaluation); }
function renderAppearance() {
  const supported = Boolean(state.recording) && state.data?.config.env_id === 'ALE/Pong-v5'
    && video.videoWidth === 160 && video.videoHeight === 210;
  const minimal = supported && state.appearance === 'minimal';
  $('appearance-control').hidden = !supported;
  $('pong-view').hidden = !minimal;
  video.hidden = !state.recording || minimal;
  $('game-screen').classList.toggle('minimal-pong', minimal);
  text('playback-description', minimal ? 'Minimal appearance · recorded decisions' : 'Original frames · recorded decisions');
  document.querySelectorAll('[data-appearance]').forEach((button) =>
    button.setAttribute('aria-pressed', button.dataset.appearance === state.appearance));
  if (minimal) paintPong();
}
function renderInspector() {
  const checkpoints = [...state.data.checkpoints].reverse();
  if (!checkpoints.some((cp) => cp.name === state.checkpoint)) state.checkpoint = checkpoints[0]?.name;
  options($('checkpoint-select'), checkpoints.map((cp) => [cp.name, `${compact(cp.step)} steps · ${cp.name}`]), state.checkpoint);
  const evaluations = state.data.evaluations.filter((entry) => entry.checkpoint === state.checkpoint);
  if (!evaluations.some((entry) => entry.id === state.evaluation)) {
    state.evaluation = evaluations.findLast((entry) => entry.has_video && entry.policy === 'greedy')?.id
      ?? evaluations.findLast((entry) => entry.has_video)?.id ?? evaluations.at(-1)?.id;
    state.episode = 0;
  }
  options($('evaluation-select'), evaluations.map((entry) => [entry.id,
    `${entry.folder ?? entry.policy} · ${entry.episodes.length} episode(s)${entry.has_video ? ' · video' : ''}`]), state.evaluation);
  $('evaluation-control').hidden = !evaluations.length;
  const evaluation = selectedEvaluation();
  const episodes = evaluation?.episodes ?? [];
  options($('episode-select'), episodes.map((episode, index) => [String(index),
    `${index + 1} · seed ${episode.seed} · return ${number(episode.return)}`]), String(state.episode));
  $('episode-control').hidden = !episodes.length;
  text('policy-caption', `${evaluation?.policy ?? 'greedy'} · ${state.checkpoint ?? 'No checkpoints yet'}`);
  $('record-another').disabled = $('record-preview').disabled = !state.checkpoint;
  const episode = episodes[state.episode];
  const recording = episode?.video_url ?? null;
  $('no-recording').hidden = Boolean(recording);
  $('playback-controls').hidden = !recording;
  $('preview-note').hidden = !recording || evaluation.compatible_for_learning;
  text('preview-note', `Different evaluation protocol · episode cap ${number(evaluation?.episode_cap)} steps. Excluded from learning statistics.`);
  if (recording === state.recording) { renderAppearance(); return; }
  state.recording = recording;
  state.frame = -1; state.decisions = []; state.scores = [];
  text('agent-score', '—'); text('opponent-score', '—');
  renderAppearance();
  video.pause();
  $('q-values').replaceChildren(element('p', 'Select a recorded episode to inspect action values.', 'muted'));
  $('decision-detail').hidden = true;
  text('chosen-action', 'Awaiting a recording');
  if (!recording) { video.removeAttribute('src'); video.load(); return; }
  text('recording-return', `Return ${number(episode.return)}`);
  video.src = recording;
  video.playbackRate = Number($('playback-speed').value);
  request(episode.decisions_url).then((result) => {
    if (state.recording !== recording) return;
    state.decisions = result.decisions.filter((decision) => Array.isArray(decision.action_values));
    state.scores = pongScores(state.decisions);
    $('decision-slider').max = state.decisions.length;
    updateDecision(true);
  }).catch(showError);
}

function clock(seconds) { return `${Math.floor((seconds || 0) / 60)}:${String(Math.floor((seconds || 0) % 60)).padStart(2, '0')}`; }
function updateDecision(force = false, mediaTime = video.currentTime) {
  const fps = selectedEvaluation()?.video_fps ?? 15;
  const index = Math.min(state.decisions.length, Math.floor(mediaTime * fps + .0001));
  text('playback-time', `${clock(video.currentTime)} / ${clock(video.duration)}`);
  $('play-pause').textContent = video.paused ? '▶' : 'Ⅱ';
  $('play-pause').setAttribute('aria-label', video.paused ? 'Play recording' : 'Pause recording');
  if (index === state.frame && !force) return;
  state.frame = index;
  const score = state.scores[index];
  text('agent-score', score ? number(score.agent) : '—');
  text('opponent-score', score ? number(score.opponent) : '—');
  $('decision-slider').value = index;
  const decision = state.decisions[Math.min(index, state.decisions.length - 1)];
  if (!decision) return;
  const final = index === state.decisions.length;
  text('decision-index', final ? `Final frame · ${number(index)} decisions` : `Decision ${number(index + 1)} / ${number(state.decisions.length)}`);
  text('chosen-action', final ? 'Last action' : decision.action_name);
  const values = decision.action_values;
  const names = selectedEvaluation()?.action_names ?? [];
  const min = Math.min(0, ...values), max = Math.max(0, ...values), span = max - min || 1;
  $('q-values').replaceChildren(...values.map((value, action) => {
    const row = element('div', undefined, `q-row ${action === decision.action ? 'selected' : ''}`);
    const bar = element('div', undefined, 'q-bar');
    const fill = element('span');
    fill.style.left = `${(Math.min(0, value) - min) / span * 100}%`;
    fill.style.width = `${Math.abs(value) / span * 100}%`;
    const zero = element('i', undefined, 'q-zero');
    zero.style.left = `${-min / span * 100}%`;
    bar.append(fill, zero);
    row.append(element('span', names[action] ?? `Action ${action}`, 'q-label'), bar, element('span', number(value, 3), 'q-number'));
    return row;
  }));
  $('decision-detail').hidden = false;
  text('decision-reward', number(decision.reward, 2));
  text('decision-state', decision.terminated ? 'Terminated' : decision.truncated ? 'Time limit' : 'In progress');
}
function seekFrame(index) {
  video.pause();
  const fps = selectedEvaluation()?.video_fps ?? 15;
  // Seek inside the frame: exact boundaries can round to the preceding MP4 timestamp.
  video.currentTime = (Math.max(0, Math.min(state.decisions.length, index)) + .5) / fps;
  updateDecision(true);
}
function recordDialog() {
  const checkpoint = state.data?.checkpoints.find((cp) => cp.name === state.checkpoint);
  if (!checkpoint) return;
  const quote = (word) => /^[a-zA-Z0-9_./:-]+$/.test(word) ? word : `'${word.replaceAll("'", "'\\''")}'`;
  text('record-command', checkpoint.record_command.map(quote).join(' '));
  text('copy-status', '');
  $('record-dialog').showModal();
}

$('run-search').addEventListener('input', renderRuns);
$('refresh').addEventListener('click', refresh);
$('metric-select').addEventListener('change', drawChart);
document.querySelectorAll('.view-tabs button').forEach((button) => button.addEventListener('click', () => setView(button.dataset.view)));
$('checkpoint-select').addEventListener('change', () => {
  state.checkpoint = $('checkpoint-select').value; state.evaluation = null; renderInspector();
});
$('evaluation-select').addEventListener('change', () => {
  state.evaluation = $('evaluation-select').value; state.episode = 0; renderInspector();
});
$('episode-select').addEventListener('change', () => { state.episode = Number($('episode-select').value); renderInspector(); });
$('play-pause').addEventListener('click', () => {
  if (video.paused) video.play().catch(showError); else video.pause();
});
$('previous-frame').addEventListener('click', () => seekFrame(state.frame - 1));
$('next-frame').addEventListener('click', () => seekFrame(state.frame + 1));
$('decision-slider').addEventListener('input', () => seekFrame(Number($('decision-slider').value)));
$('playback-speed').addEventListener('change', () => { video.playbackRate = Number($('playback-speed').value); });
document.querySelectorAll('[data-appearance]').forEach((button) => button.addEventListener('click', () => {
  state.appearance = button.dataset.appearance;
  renderAppearance();
}));
document.addEventListener('keydown', (event) => {
  if (!state.recording || !event.target.closest('.policy-panel')
    || ['INPUT', 'SELECT'].includes(event.target.tagName)) return;
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    event.preventDefault(); seekFrame(state.frame + (event.key === 'ArrowRight' ? 1 : -1));
  }
});
video.addEventListener('loadeddata', () => { renderAppearance(); updateDecision(true); });
['timeupdate', 'loadedmetadata', 'seeked', 'pause', 'ended'].forEach((event) => video.addEventListener(event, () => {
  updateDecision();
  if (event === 'seeked') renderAppearance();
}));
if (video.requestVideoFrameCallback) {
  const frame = (_now, metadata) => {
    if (!$('pong-view').hidden) paintPong();
    updateDecision(false, metadata.mediaTime);
    video.requestVideoFrameCallback(frame);
  };
  video.requestVideoFrameCallback(frame);
} else video.addEventListener('play', () => {
  function tick() {
    updateDecision();
    if (!$('pong-view').hidden) paintPong();
    if (!video.paused) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
});
['record-preview', 'record-another'].forEach((id) => $(id).addEventListener('click', recordDialog));
$('close-dialog').addEventListener('click', () => $('record-dialog').close());
$('copy-command').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($('record-command').textContent); text('copy-status', 'Copied.'); }
  catch { text('copy-status', 'Select and copy the command above.'); }
});
setInterval(() => { if ($('auto-refresh').checked && !document.hidden && !$('record-dialog').open) refresh(); }, 5000);
await refresh();

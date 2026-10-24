'use strict';
/* Дашборд «Контроль ТБ»: пять уровней по ролям из кейса.
   1 Оператор (живая лента) · 2 Мастер (фильтр по своему участку) · 3 Журнал · 4 Аналитика · 5 Система.
   Без сборки: обычный JS + ECharts из web/vendor. Все данные — из API журнала. */

const STATUS_RU = { new: 'Новое', confirmed: 'Подтверждено', false_alarm: 'Ложное', sent_to_master: 'Передано мастеру', resolved: 'Закрыто' };
const SEV_RU = { critical: 'Критично', warning: 'Предупреждение' };
const CAT_RU = { environment: 'Среда', behavior: 'Поведение', zone: 'Зона', interaction: 'Взаимодействие', attribute: 'Экипировка' };
const ACTIONS = [
  ['confirmed', 'Подтвердить'],
  ['sent_to_master', 'Передать мастеру'],
  ['false_alarm', 'Ложное срабатывание'],
  ['resolved', 'Закрыть'],
];
const ICON = {
  critical: '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 1.2 15.2 14H.8L8 1.2Zm-.8 4.8v4.2h1.6V6H7.2Zm0 5.4v1.6h1.6v-1.6H7.2Z"/></svg>',
  warning: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.6" fill="none" stroke="currentColor" stroke-width="1.6"/><path fill="currentColor" d="M7.2 4h1.6v5.2H7.2zm0 6.2h1.6v1.6H7.2z"/></svg>',
};

let TZ = 'Europe/Moscow';
const state = {
  feed: [], selected: null, opScope: 'new', opArea: '', banner: null,
  jOffset: 0, jTotal: 0, jLimit: 50,
  days: 30, anArea: '', sound: true, charts: {}, tables: {},
};

/* ---------- утилиты ---------- */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k === 'svg') n.innerHTML = v;              // только для своих иконок
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat()) {
    if (c == null || c === false) continue;
    n.append(c instanceof Node ? c : document.createTextNode(String(c)));  // текст — всегда как текст
  }
  return n;
}
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

async function api(path, params = {}, opts = {}) {
  const url = new URL(path, location.origin);
  for (const [k, v] of Object.entries(params)) {
    if (v == null || v === '') continue;
    if (Array.isArray(v)) v.forEach((x) => url.searchParams.append(k, x));
    else url.searchParams.set(k, v);
  }
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json();
}

const fmtTime = (iso, date = true) => (iso ? new Intl.DateTimeFormat('ru-RU', {
  timeZone: TZ, ...(date ? { day: '2-digit', month: '2-digit' } : {}), hour: '2-digit', minute: '2-digit', second: '2-digit',
}).format(new Date(iso)) : '—');
const fmtDay = (iso) => new Intl.DateTimeFormat('ru-RU', { timeZone: TZ, day: '2-digit', month: '2-digit' }).format(new Date(iso));
const fmtMs = (ms) => (ms == null ? '—' : ms < 1000 ? `${Math.round(ms)} мс` : `${(ms / 1000).toFixed(1).replace('.', ',')} с`);
const fmtMsFine = (ms) => (ms == null ? '—' : ms < 100 ? `${ms.toFixed(1).replace('.', ',')} мс` : fmtMs(ms));
const fmtPct = (x) => (x == null ? '—' : `${Math.round(x * 100)}%`);
const fmtInt = (n) => new Intl.NumberFormat('ru-RU').format(n ?? 0);

function sevBadge(sev) {
  const b = el('span', { class: `sev ${sev}`, svg: ICON[sev] || '' });
  b.append(document.createTextNode(SEV_RU[sev] || sev));   // уровень: цвет + иконка + слово
  return b;
}
const statusChip = (st) => el('span', { class: `chip ${st === 'new' ? 'new' : ''}` }, STATUS_RU[st] || st);

function storeGet(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } }
function storeSet(k, v) { try { localStorage.setItem(k, v); } catch { /* приватный режим */ } }

/* ---------- вкладки ---------- */
const LOADERS = { operator: loadFeed, journal: loadJournal, analytics: loadAnalytics, system: loadSystem };
function showTab(name) {
  if (!LOADERS[name]) name = 'operator';
  $$('.tab').forEach((s) => { s.hidden = s.id !== `tab-${name}`; });
  $$('.tabs a').forEach((a) => a.classList.toggle('on', a.dataset.tab === name));
  state.tab = name;
  clearInterval(state.sysTimer);
  if (name === 'system') state.sysTimer = setInterval(() => { if (!document.hidden) loadSystemLive().catch(showError); }, 5000);
  LOADERS[name]().catch(showError);
}
function showError(err) { console.error(err); }

/* ---------- справочники для фильтров ---------- */
async function loadLookups() {
  const cams = await api('/api/cameras');
  const areas = [...new Set(cams.map((c) => c.area))].sort();
  const fill = (sel, items) => items.forEach(([v, t]) => sel.append(el('option', { value: v }, t)));
  fill($('#op-area'), areas.map((a) => [a, a]));
  fill($('#an-area'), areas.map((a) => [a, a]));
  fill($('#j-filters [name=area]'), areas.map((a) => [a, a]));
  fill($('#j-filters [name=camera_id]'), cams.map((c) => [c.id, `${c.id} — ${c.name}`]));
  fill($('#j-filters [name=status]'), Object.entries(STATUS_RU));
  const sum = await api('/api/stats/summary');
  fill($('#j-filters [name=pattern_id]'), sum.by_pattern.map((p) => [p.pattern_id, p.name]).sort((a, b) => a[1].localeCompare(b[1], 'ru')));
  state.opArea = storeGet('opArea', '');
  if (areas.includes(state.opArea)) $('#op-area').value = state.opArea; else state.opArea = '';
}

/* ---------- 1–2. Оператор и мастер ---------- */
function feedOrder(a, b) {
  return (b.status === 'new') - (a.status === 'new')
    || (b.severity === 'critical') - (a.severity === 'critical')
    || new Date(b.started_at) - new Date(a.started_at);
}

async function loadFeed() {
  const p = { limit: 100, area: state.opArea || null };
  if (state.opScope === 'new') p.status = 'new';
  else p.date_from = new Date(Date.now() - 24 * 3600e3).toISOString();
  const page = await api('/api/events', p);
  state.feed = page.items;
  renderFeed();
}

function renderFeed(freshId) {
  const list = $('#feed');
  const items = state.feed.slice().sort(feedOrder);
  if (!items.length) {
    list.replaceChildren(el('li', { class: 'empty' }, state.opScope === 'new' ? 'Нет событий, требующих реакции' : 'За сутки событий нет'));
  } else {
    list.replaceChildren(...items.map((e) => el('li', {
      class: [e.id === state.selected && 'sel', e.id === freshId && 'fresh'].filter(Boolean).join(' '),
      tabindex: 0, 'data-id': e.id,
      onclick: () => selectEvent(e.id),
      onkeydown: (ev) => { if (ev.key === 'Enter') selectEvent(e.id); },
    },
    el('div', { class: 'row' }, sevBadge(e.severity), statusChip(e.status)),
    el('div', { class: 'title' }, `${e.extra?.demo ? '[Демо] ' : ''}${e.pattern_name}`),
    el('div', { class: 'meta' }, `${fmtTime(e.started_at)} · ${e.camera_id} · ${e.area}`))));
  }
  const n = state.feed.filter((e) => e.status === 'new').length;
  $('#op-count').textContent = state.opScope === 'new' ? `Требуют реакции: ${n}` : `За сутки: ${state.feed.length}`;
}

async function selectEvent(id) {
  state.selected = id;
  $$('#feed li').forEach((li) => li.classList.toggle('sel', Number(li.dataset.id) === id));
  const e = await api(`/api/events/${id}`);
  if (state.selected === id) renderDetail($('#op-detail'), e, onStatusChanged);
}

function onStatusChanged(e) {
  const i = state.feed.findIndex((x) => x.id === e.id);
  if (i >= 0) {
    if (state.opScope === 'new' && e.status !== 'new') state.feed.splice(i, 1);
    else state.feed[i] = e;
  }
  if (state.banner === e.id && e.status !== 'new') hideBanner();
  renderFeed();
}

function fact(label, value) { return el('div', {}, el('dt', {}, label), el('dd', {}, value ?? '—')); }

function renderDetail(box, e, onChange) {
  const sameEvent = box.dataset.eventId === String(e.id);
  const previousWho = sameEvent ? $('.review-who', box)?.value : null;
  const previousComment = sameEvent ? $('.review-comment', box)?.value : null;
  const oldVideo = sameEvent ? $('video', box) : null;
  box.dataset.eventId = String(e.id);
  box.className = 'detail';
  const clipInfo = e.extra?.clip || {};
  let clip;
  if (e.clip_path) {
    const player = oldVideo?.getAttribute('src') === e.clip_path ? oldVideo : el('video', {
      src: e.clip_path, controls: true, playsinline: true, preload: 'metadata',
      'aria-label': 'Видео события', style: 'width:100%;max-height:420px;background:#111',
    });
    clip = el('section', { class: 'event-clip' }, el('h3', {}, 'Видео события'), player,
      clipInfo.partial ? el('p', { class: 'muted' }, 'Сохранён доступный фрагмент: начало или конец записи ограничены.') : null,
      el('a', { href: e.clip_path, download: true, class: 'btn ghost' }, 'Скачать клип'));
  } else {
    const messages = { recording: 'Записывается видео после события…', encoding: 'Подготавливается видеоклип…',
      failed: 'Не удалось подготовить клип. Кадр и событие сохранены.' };
    clip = el('p', { class: 'clip-status muted', role: 'status' }, messages[clipInfo.state] || 'Видеоклип не приложен');
  }
  const shot = el('div', { class: e.snapshot_path ? 'shot' : 'shot none' },
    e.snapshot_path ? el('img', { src: e.snapshot_path, alt: `Кадр: ${e.pattern_name}, ${e.camera_id}` })
      : 'Кадр-доказательство не приложен');
  const who = el('input', { type: 'text', class: 'review-who', value: previousWho ?? storeGet('who', 'оператор-1'), 'aria-label': 'Кто реагирует', style: 'flex:0 1 160px' });
  const comment = el('input', { type: 'text', class: 'review-comment', value: previousComment ?? '', placeholder: 'Комментарий (необязательно)', 'aria-label': 'Комментарий' });
  const buttons = ACTIONS.filter(([st]) => st !== e.status).map(([st, label], i) => el('button', {
    class: i === 0 ? 'btn' : 'btn ghost', type: 'button', disabled: box.dataset.submitting === '1',
    onclick: async (ev) => {
      if (box.dataset.submitting === '1') return;
      box.dataset.submitting = '1';
      $$('.actions button', box).forEach((b) => { b.disabled = true; });
      $('.review-error', box).textContent = '';
      storeSet('who', who.value || 'оператор');
      try {
        const upd = await api(`/api/events/${e.id}/status`, {}, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status: st, user: who.value || 'оператор', comment: comment.value || null }),
        });
        box.dataset.submitting = '0';
        if (box.dataset.eventId === String(e.id)) {
          const field = $('.review-comment', box);
          if (field) field.value = '';
          renderDetail(box, upd, onChange);
          $('.review-error', box).textContent = 'Решение сохранено на сервере.';
        }
        onChange && onChange(upd);
      } catch (err) {
        box.dataset.submitting = '0';
        $$('.actions button', box).forEach((b) => { b.disabled = false; });
        $('.review-error', box).textContent = 'Не удалось сохранить решение. Проверьте связь и актуальный статус события.';
        showError(err);
      }
    },
  }, label));
  const cats = (e.categories || []).map((c) => CAT_RU[c] || c).join(' + ') || '—';
  box.replaceChildren(...[
    el('div', { class: 'row', style: 'display:flex;gap:8px;align-items:center' }, sevBadge(e.severity), statusChip(e.status)),
    el('h2', {}, e.pattern_name),
    shot,
    clip,
    e.extra?.followup ? el('p', { class: 'event-followup' },
      `Уточнение: ${e.extra.followup.condition || 'Обнаружено продолжение события'}`) : null,
    el('dl', { class: 'facts' },
      fact('Камера', e.camera_id), fact('Участок', e.area), fact('Зона', e.zone || '—'),
      fact('Трек', e.track_id ?? '—'), fact('Уверенность', fmtPct(e.confidence)), fact('Категории', cats),
      fact('Начало', fmtTime(e.started_at)), fact('Тревога', fmtTime(e.confirmed_at)),
      fact('Время до тревоги', fmtMs(e.time_to_alert_ms)), fact('Задержка конвейера', fmtMs(e.pipeline_latency_ms)),
      fact('Окончание', fmtTime(e.ended_at)),
      fact('Кто отреагировал', e.reviewed_by ? `${e.reviewed_by}, ${fmtTime(e.reviewed_at)}` : '—')),
    el('div', { class: 'actions' }, who, comment, ...buttons),
    el('p', { class: 'review-error muted', role: 'status', 'aria-live': 'polite' }),
    (e.history && e.history.length)
      ? el('ol', { class: 'history' }, ...e.history.map((h) => el('li', {},
        `${fmtTime(h.at)} — ${h.user}: ${STATUS_RU[h.from_status] || h.from_status} → ${STATUS_RU[h.to_status] || h.to_status}${h.comment ? ` («${h.comment}»)` : ''}`)))
      : el('p', { class: 'muted' }, 'Реакций пока не было'),
  ].filter((item) => item != null));
}

function showBanner(e) {
  state.banner = e.id;
  const b = $('#critical-banner');
  b.replaceChildren(sevBadge('critical'),
    el('span', {}, `${e.pattern_name} — ${e.area}, ${e.camera_id}, ${fmtTime(e.started_at, false)}`),
    el('button', { class: 'btn', type: 'button', onclick: () => { selectEvent(e.id); } }, 'Открыть'));
  b.hidden = false;
}
function hideBanner() { state.banner = null; $('#critical-banner').hidden = true; }

/* живая лента */
function setLive(on) {
  $('#live').classList.toggle('off', !on);
  $('#live-text').textContent = on ? 'Лента: онлайн' : 'Лента: нет связи';
}
function connectWS() {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/events`);
  ws.onopen = () => {
    setLive(true);
    // Recover events/media missed while disconnected.
    loadFeed().catch(showError);
    refreshOpenDetails();
    if (state.tab === 'journal') loadJournal().catch(showError);
  };
  ws.onclose = () => { setLive(false); setTimeout(connectWS, 2000); };
  ws.onmessage = (m) => { try { onLive(JSON.parse(m.data)); } catch (err) { showError(err); } };
}
async function refreshDetail(box, id, onChange) {
  const version = (Number(box.dataset.refreshVersion) || 0) + 1;
  box.dataset.refreshVersion = String(version);
  try {
    const e = await api(`/api/events/${id}`);
    if (box.dataset.eventId === String(id) && box.dataset.refreshVersion === String(version)) {
      renderDetail(box, e, onChange);
    }
  } catch (err) { showError(err); }
}
function refreshOpenDetails(eventId = null) {
  const op = $('#op-detail');
  if (op.dataset.eventId && state.tab === 'operator' && (!eventId || Number(op.dataset.eventId) === eventId)) {
    refreshDetail(op, Number(op.dataset.eventId), onStatusChanged);
  }
  const drawer = $('#drawer-body');
  if (!$('#drawer').hidden && drawer.dataset.eventId && (!eventId || Number(drawer.dataset.eventId) === eventId)) {
    refreshDetail(drawer, Number(drawer.dataset.eventId), () => loadJournal().catch(showError));
  }
}
function onLive(msg) {
  const e = msg.event;
  if (!e) return;
  if (state.tab === 'journal') loadJournal().catch(showError);
  if (msg.type === 'event_created') {
    if (state.opArea && e.area !== state.opArea) return;
    if (!state.feed.some((x) => x.id === e.id)) state.feed.unshift(e);
    renderFeed(e.id);
    if (e.severity === 'critical') { showBanner(e); beep(); }
    return;
  }
  const i = state.feed.findIndex((x) => x.id === e.id);
  if (i >= 0) {
    if (state.opScope === 'new' && e.status !== 'new') state.feed.splice(i, 1); else state.feed[i] = e;
    renderFeed();
  }
  if (state.banner === e.id && e.status !== 'new') hideBanner();
  refreshOpenDetails(e.id);
}

let actx;
function beep() {
  if (!state.sound) return;
  try {
    actx = actx || new (window.AudioContext || window.webkitAudioContext)();
    const t = actx.currentTime;
    [[0, 880], [0.25, 660]].forEach(([d, f]) => {
      const o = actx.createOscillator(); const g = actx.createGain();
      o.frequency.value = f; o.connect(g); g.connect(actx.destination);
      g.gain.setValueAtTime(0.0001, t + d);
      g.gain.exponentialRampToValueAtTime(0.2, t + d + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t + d + 0.2);
      o.start(t + d); o.stop(t + d + 0.22);
    });
  } catch { /* браузер не дал звук без действия пользователя */ }
}

/* ---------- 3. Журнал ---------- */
function journalParams() {
  const p = {};
  for (const [k, v] of new FormData($('#j-filters'))) {
    if (!v) continue;
    p[k] = (k === 'date_from' || k === 'date_to') ? new Date(v).toISOString() : v;
  }
  return p;
}

async function loadJournal() {
  const p = journalParams();
  const page = await api('/api/events', { ...p, limit: state.jLimit, offset: state.jOffset });
  state.jTotal = page.total;
  $('#j-table tbody').replaceChildren(...(page.items.length ? page.items.map((e) => el('tr', { onclick: () => openDrawer(e.id) },
    el('td', {}, fmtTime(e.started_at)), el('td', {}, e.pattern_name), el('td', {}, sevBadge(e.severity)),
    el('td', {}, e.area), el('td', {}, e.camera_id), el('td', {}, statusChip(e.status)),
    el('td', {}, e.reviewed_by || '—'), el('td', { class: 'num' }, fmtMs(e.time_to_alert_ms)))) : [
    el('tr', {}, el('td', { colspan: 8, class: 'muted' }, 'По этим фильтрам событий нет'))]));
  const from = page.total ? state.jOffset + 1 : 0;
  $('#j-page').textContent = `${fmtInt(from)}–${fmtInt(Math.min(state.jOffset + state.jLimit, page.total))} из ${fmtInt(page.total)}`;
  $('#j-prev').disabled = state.jOffset === 0;
  $('#j-next').disabled = state.jOffset + state.jLimit >= page.total;
  const q = new URLSearchParams({ ...p, tz: TZ });
  $('#j-export').href = `/api/events/export.csv?${q}`;
}

async function openDrawer(id) {
  const e = await api(`/api/events/${id}`);
  $('#drawer').hidden = false;
  renderDetail($('#drawer-body'), e, () => loadJournal().catch(showError));
}

/* ---------- 4. Аналитика ---------- */
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function theme() {
  return {
    surface: css('--surface'), text: css('--text'), text2: css('--text-2'), muted: css('--muted'),
    grid: css('--grid'), axis: css('--axis'), border: css('--border'),
    s1: css('--series-1'), s2: css('--series-2'), seq: css('--seq').split(',').map((s) => s.trim()),
  };
}
function base(t) {
  return {
    animationDuration: 300,
    textStyle: { fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: t.text2 },
    tooltip: {
      backgroundColor: t.surface, borderColor: t.border, borderWidth: 1, textStyle: { color: t.text, fontSize: 13 },
      extraCssText: 'border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,.14);',
    },
  };
}
const axisStyle = (t) => ({
  axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false },
  axisLabel: { color: t.muted }, splitLine: { lineStyle: { color: t.grid, width: 1 } },
});
const tipRow = (color, value, label) => `<div style="display:flex;align-items:center;gap:8px"><span style="width:12px;height:2px;background:${color};display:inline-block"></span><b>${esc(value)}</b><span style="opacity:.75">${esc(label)}</span></div>`;

function chartFor(key) {
  const div = $(`[data-chart="${key}"] .chart`);
  let c = state.charts[key];
  if (!c || c.getDom() !== div) {
    c = echarts.init(div, null, { renderer: 'svg' });
    state.charts[key] = c;
    // график всегда подстраивается под фактический размер карточки (вкладки, таблица, окно)
    new ResizeObserver(() => { if (div.clientWidth) c.resize(); }).observe(div);
  }
  return c;
}
function setTable(key, cols, rows) {
  state.tables[key] = { cols, rows };
  const box = $(`[data-chart="${key}"] .tablebox`);
  if (!box.hidden) renderTable(key);
}
function renderTable(key) {
  const { cols, rows } = state.tables[key] || { cols: [], rows: [] };
  $(`[data-chart="${key}"] .tablebox`).replaceChildren(el('table', { class: 'table' },
    el('thead', {}, el('tr', {}, ...cols.map((c, i) => el('th', { class: i ? 'num' : '' }, c)))),
    el('tbody', {}, ...rows.map((r) => el('tr', {}, ...r.map((v, i) => el('td', { class: i ? 'num' : '' }, typeof v === 'number' ? fmtInt(v) : v)))))));
}

async function loadAnalytics() {
  const sec = $('#tab-analytics');
  sec.classList.add('loading');                                   // держим прошлую картинку, без мигания
  const p = { date_from: new Date(Date.now() - state.days * 864e5).toISOString(), area: state.anArea || null, tz: TZ };
  const [sum, ts, hm, q] = await Promise.all([
    api('/api/stats/summary', p),
    api('/api/stats/timeseries', { ...p, bucket: 'day' }),
    api('/api/stats/heatmap', { ...p, rows: state.anArea ? 'camera' : 'area' }),
    api('/api/stats/quality', p),
  ]);
  renderTiles(sum, q);
  const t = theme();
  chartDaily(t, ts); chartPatterns(t, sum); chartShifts(t, sum); chartHeatmap(t, hm);
  sec.classList.remove('loading');
}

function tile(label, value, sub, hero) {
  return el('div', { class: `tile${hero ? ' hero' : ''}` }, el('div', { class: 'label' }, label), el('div', { class: 'value' }, value), sub ? el('div', { class: 'sub' }, sub) : null);
}
function renderTiles(sum, q) {
  const o = q.overall;
  $('#an-tiles').replaceChildren(
    tile('Нарушений за период', fmtInt(sum.total), `за ${state.days} дней${state.anArea ? ` · ${state.anArea}` : ''}`, true),
    tile('Критичных', fmtInt(sum.by_severity.critical || 0), `ждут реакции: ${fmtInt(sum.open_critical)}`),
    tile('Точность тревог (precision)', fmtPct(o.precision), `по ${fmtInt(o.reviewed)} отметкам операторов`),
    tile('Время до тревоги, медиана', fmtMs(o.time_to_alert_ms_p50), `p95: ${fmtMs(o.time_to_alert_ms_p95)}`),
    tile('Ночная смена', fmtPct(sum.total ? sum.by_shift.night / sum.total : null), `${fmtInt(sum.by_shift.night)} из ${fmtInt(sum.total)} нарушений`),
  );
}

function chartDaily(t, ts) {
  const labels = ts.map((d) => fmtDay(d.t));
  const values = ts.map((d) => d.total);
  chartFor('daily').setOption({
    ...base(t),
    grid: { left: 8, right: 40, top: 16, bottom: 8, containLabel: true },
    tooltip: { ...base(t).tooltip, trigger: 'axis', axisPointer: { type: 'line', lineStyle: { color: t.axis, width: 1 } },
      formatter: (ps) => `${tipRow(t.s1, fmtInt(ps[0].value), 'нарушений')}<div style="opacity:.75;margin-top:2px">${esc(ps[0].axisValue)}</div>` },
    xAxis: { type: 'category', data: labels, boundaryGap: false, ...axisStyle(t), splitLine: { show: false } },
    yAxis: { type: 'value', minInterval: 1, ...axisStyle(t), axisLine: { show: false } },
    series: [{
      type: 'line', data: values, showSymbol: false, symbolSize: 8, lineStyle: { width: 2, color: t.s1, cap: 'round', join: 'round' },
      itemStyle: { color: t.s1, borderColor: t.surface, borderWidth: 2 }, areaStyle: { color: t.s1, opacity: 0.1 },
      endLabel: { show: true, color: t.text2, formatter: (p) => fmtInt(p.value) },
    }],
  }, true);
  setTable('daily', ['День', 'Нарушений'], ts.map((d, i) => [labels[i], d.total]));
}

function chartPatterns(t, sum) {
  const top = sum.by_pattern.slice(0, 10);
  chartFor('patterns').setOption({
    ...base(t),
    grid: { left: 8, right: 48, top: 8, bottom: 8, containLabel: true },
    tooltip: { ...base(t).tooltip, trigger: 'item', formatter: (p) => `${tipRow(t.s1, fmtInt(p.value), 'нарушений')}<div style="margin-top:2px">${esc(p.name)}</div>` },
    xAxis: { type: 'value', ...axisStyle(t), axisLabel: { show: false }, splitLine: { show: false }, axisLine: { show: false } },
    yAxis: { type: 'category', inverse: true, data: top.map((p) => p.name), ...axisStyle(t),
      axisLabel: { color: t.text2, width: 230, overflow: 'truncate' } },
    series: [{ type: 'bar', data: top.map((p) => p.count), barMaxWidth: 20, itemStyle: { color: t.s1, borderRadius: [0, 4, 4, 0] },
      label: { show: true, position: 'right', color: t.text2, formatter: (p) => fmtInt(p.value) },
      emphasis: { itemStyle: { opacity: 0.85 } } }],
  }, true);
  setTable('patterns', ['Нарушение', 'Количество'], sum.by_pattern.map((p) => [p.name, p.count]));
}

function chartShifts(t, sum) {
  const areas = [...new Set(sum.by_area_shift.map((r) => r.area))];
  const get = (a, s) => (sum.by_area_shift.find((r) => r.area === a && r.shift === s) || { count: 0 }).count;
  const totals = areas.map((a) => get(a, 'day') + get(a, 'night'));
  const order = areas.map((a, i) => [a, totals[i]]).sort((x, y) => y[1] - x[1]).map((x) => x[0]);
  const series = [['day', 'Дневная 08–20', t.s1], ['night', 'Ночная 20–08', t.s2]].map(([k, name, color]) => ({
    type: 'bar', name, data: order.map((a) => get(a, k)), barMaxWidth: 16, barGap: '12%',
    itemStyle: { color, borderRadius: [0, 4, 4, 0] }, label: { show: true, position: 'right', color: t.text2, fontSize: 12 },
  }));
  chartFor('shifts').setOption({
    ...base(t),
    legend: { top: 0, left: 0, icon: 'rect', itemWidth: 12, itemHeight: 12, textStyle: { color: t.text2 } },
    grid: { left: 8, right: 40, top: 32, bottom: 8, containLabel: true },
    tooltip: { ...base(t).tooltip, trigger: 'axis', axisPointer: { type: 'shadow', shadowStyle: { color: t.grid, opacity: 0.4 } },
      formatter: (ps) => `<div style="margin-bottom:4px">${esc(ps[0].axisValue)}</div>${ps.map((p) => tipRow(p.color, fmtInt(p.value), p.seriesName)).join('')}` },
    xAxis: { type: 'value', ...axisStyle(t), axisLabel: { show: false }, splitLine: { show: false }, axisLine: { show: false } },
    yAxis: { type: 'category', inverse: true, data: order, ...axisStyle(t), axisLabel: { color: t.text2, width: 160, overflow: 'truncate' } },
    series,
  }, true);
  setTable('shifts', ['Участок', 'Дневная', 'Ночная'], order.map((a) => [a, get(a, 'day'), get(a, 'night')]));
}

function chartHeatmap(t, hm) {
  const hours = hm.hours.map((h) => String(h).padStart(2, '0'));
  const data = [];
  hm.values.forEach((row, y) => row.forEach((v, x) => data.push([x, y, v])));
  const max = Math.max(1, ...data.map((d) => d[2]));
  chartFor('heatmap').setOption({
    ...base(t),
    grid: { left: 8, right: 16, top: 8, bottom: 48, containLabel: true },
    tooltip: { ...base(t).tooltip, trigger: 'item', formatter: (p) => `${tipRow(t.s1, fmtInt(p.value[2]), 'нарушений')}<div style="margin-top:2px">${esc(hm.rows[p.value[1]])}, ${hours[p.value[0]]}:00–${hours[p.value[0]]}:59</div>` },
    xAxis: { type: 'category', data: hours, ...axisStyle(t), splitLine: { show: false }, axisLine: { show: false } },
    yAxis: { type: 'category', data: hm.rows, inverse: true, ...axisStyle(t), splitLine: { show: false }, axisLine: { show: false },
      axisLabel: { color: t.text2, width: 170, overflow: 'truncate' } },
    visualMap: { min: 0, max, calculable: false, orient: 'horizontal', left: 'center', bottom: 0, itemWidth: 12, itemHeight: 180,
      inRange: { color: t.seq }, textStyle: { color: t.muted }, text: [`${fmtInt(max)}`, '0'] },
    series: [{ type: 'heatmap', data, itemStyle: { borderColor: t.surface, borderWidth: 2, borderRadius: 3 },
      emphasis: { itemStyle: { borderColor: t.text, borderWidth: 2 } } }],
  }, true);
  setTable('heatmap', ['Участок / камера', ...hours], hm.rows.map((r, i) => [r, ...hm.values[i]]));
}

/* ---------- 5. Система: узлы LinQ, загрузка TPU, FPS, запас по камерам ---------- */
const fmtNum = (x, d = 1) => (x == null ? '—' : x.toFixed(d).replace('.', ','));
function utilLevel(u, cap) {
  if (u > 0.9) return ['over', 'перегрузка'];
  if (u >= Math.min(cap, 0.9) - 0.1 && u >= 0.7) return ['high', 'высокая'];
  return ['', 'норма'];
}
function nodeState(n) {
  if (n.source === 'bench') return el('span', { class: 'node-state off' }, el('span', { class: 'dot' }), `замер ${fmtTime(n.last_seen)}`);
  return n.online
    ? el('span', { class: 'node-state' }, el('span', { class: 'dot' }), 'в сети')
    : el('span', { class: 'node-state off' }, el('span', { class: 'dot' }), `нет связи с ${fmtTime(n.last_seen)}`);
}
function capText(c) {
  if (!c) return 'Нет данных для расчёта: узел ещё не присылал замеры.';
  const base = `Цена кадра ${fmtNum(c.tpu_ms_per_frame, 1)} мс TPU → при ${c.target_fps} FPS на камеру и загрузке до ${Math.round(c.util_cap * 100)}% узел держит до ${c.max_streams} камер, сейчас ${c.current_streams}. `;
  if (c.headroom > 0) return [base, el('b', {}, `Можно добавить ещё ${c.headroom}.`)];
  if (c.headroom === 0) return [base, el('b', {}, 'Узел заполнен.')];
  return [base, el('b', {}, `Перегрузка: уберите ${-c.headroom} камер(ы) или снизьте FPS.`)];
}

function renderNode(n, cap) {
  const m = n.latest;
  if (!m) return el('div', { class: 'node' }, el('div', { class: 'node-head' }, el('h3', {}, n.name), nodeState(n)), el('p', { class: 'muted' }, 'Метрик пока нет'));
  const [lvl, word] = utilLevel(m.tpu_util, cap);
  const fpsAvg = Object.values(m.per_stream || {});
  const avg = fpsAvg.length ? fpsAvg.reduce((a, b) => a + b, 0) / fpsAvg.length : null;
  const models = Object.entries(m.models || {});
  const meter = el('div', { class: `meter ${lvl}`, role: 'meter', 'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-valuenow': Math.round(m.tpu_util * 100), 'aria-label': 'Загрузка TPU' },
    el('span', { style: `width:${Math.min(100, m.tpu_util * 100).toFixed(1)}%` }),
    el('i', { class: 'cap', style: `left:calc(${cap * 100}% - 1px)`, title: `предел ${Math.round(cap * 100)}%` }));
  return el('div', { class: 'node' },
    el('div', { class: 'node-head' },
      el('h3', {}, n.name, n.source === 'demo' ? el('span', { class: 'tag' }, 'демо') : null, n.source === 'bench' ? el('span', { class: 'tag' }, 'замер') : null),
      nodeState(n)),
    el('div', { class: 'meter-row' },
      el('div', { class: 'meter-label' }, el('span', {}, 'Загрузка TPU ', el('span', { class: 'muted' }, `· ${word}`)), el('span', { class: 'big' }, fmtPct(m.tpu_util))),
      meter),
    el('dl', { class: 'kv' },
      el('div', {}, el('dt', {}, 'Потоков'), el('dd', {}, m.streams)),
      el('div', {}, el('dt', {}, 'FPS на поток, мин / ср.'), el('dd', {}, `${fmtNum(m.fps_stream_min)} / ${fmtNum(avg)}`)),
      el('div', {}, el('dt', {}, 'Кадров/с на узле'), el('dd', {}, fmtNum(m.fps_total))),
      el('div', {}, el('dt', {}, 'CPU · отброшено кадров'), el('dd', {}, `${fmtPct(m.cpu_util)} · ${m.dropped}`))),
    models.length ? el('table', { class: 'table' },
      el('thead', {}, el('tr', {}, el('th', {}, 'Модель'), el('th', { class: 'num' }, 'Вызовов/кадр'), el('th', { class: 'num' }, 'p50'), el('th', { class: 'num' }, 'p95'))),
      el('tbody', {}, ...models.map(([name, st]) => el('tr', {},
        el('td', {}, name), el('td', { class: 'num' }, m.frames ? fmtNum(st.calls / m.frames, 2) : '—'),
        el('td', { class: 'num' }, fmtMsFine(st.p50_ms ?? st.mean_ms)), el('td', { class: 'num' }, fmtMsFine(st.p95_ms)))))) : null,
    el('div', { class: 'capline' }, capText(n.capacity)));
}

function seriesChart(key, t, data, nodes, field, { pct = false, ref = null, refLabel = '' } = {}) {
  const colors = [t.s1, t.s2];
  const ids = Object.keys(data);
  const name = (id) => (nodes.find((n) => n.id === id) || { name: id }).name;
  const timeFmt = new Intl.DateTimeFormat('ru-RU', { timeZone: TZ, hour: '2-digit', minute: '2-digit' });
  const series = ids.slice(0, 2).map((id, i) => ({   // больше двух узлов — отдельные графики/таблица
    type: 'line', name: name(id), showSymbol: false, symbolSize: 8, sampling: 'lttb',
    lineStyle: { width: 2, color: colors[i] }, itemStyle: { color: colors[i], borderColor: t.surface, borderWidth: 2 },
    data: data[id].map((p) => [p.t, pct ? (p[field] ?? 0) * 100 : p[field]]),
    markLine: i === 0 && ref != null ? { symbol: 'none', silent: true, lineStyle: { color: t.muted, width: 1, type: 'solid' },
      label: { formatter: refLabel, color: t.muted, position: 'insideEndTop' }, data: [{ yAxis: ref }] } : undefined,
  }));
  chartFor(key).setOption({
    ...base(t),
    legend: ids.length > 1 ? { top: 0, left: 0, icon: 'roundRect', itemWidth: 14, itemHeight: 3, textStyle: { color: t.text2 } } : undefined,
    grid: { left: 8, right: 16, top: ids.length > 1 ? 30 : 12, bottom: 8, containLabel: true },
    tooltip: { ...base(t).tooltip, trigger: 'axis', axisPointer: { type: 'line', lineStyle: { color: t.axis, width: 1 } },
      formatter: (ps) => `${ps.map((p) => tipRow(p.color, pct ? `${fmtNum(p.value[1], 0)}%` : fmtNum(p.value[1]), p.seriesName)).join('')}<div style="opacity:.75;margin-top:2px">${timeFmt.format(new Date(ps[0].value[0]))}</div>` },
    xAxis: { type: 'time', ...axisStyle(t), splitLine: { show: false }, axisLabel: { color: t.muted, formatter: (v) => timeFmt.format(new Date(v)) } },
    yAxis: { type: 'value', ...axisStyle(t), axisLine: { show: false }, min: 0, max: pct ? 100 : undefined,
      axisLabel: { color: t.muted, formatter: pct ? '{value}%' : '{value}' } },
    series,
  }, true);
  const rows = [];
  ids.forEach((id) => data[id].forEach((p) => rows.push([`${timeFmt.format(new Date(p.t))} · ${name(id)}`, pct ? `${fmtNum((p[field] ?? 0) * 100, 0)}%` : fmtNum(p[field])])));
  setTable(key, ['Время · узел', pct ? 'Загрузка TPU' : 'FPS'], rows.slice(-200));
}

async function loadSystemLive() {
  const fps = Number($('#cap-fps').value); const cap = Number($('#cap-util').value);
  const [nodes, data] = await Promise.all([
    api('/api/metrics/nodes', { target_fps: fps, util_cap: cap }),
    api('/api/metrics/series', { minutes: 60 }),
  ]);
  $('#demo-note').hidden = !nodes.some((n) => n.source === 'demo');
  $('#nodes').replaceChildren(...(nodes.length ? nodes.map((n) => renderNode(n, cap))
    : [el('div', { class: 'notice' }, 'Узлы ещё не присылали метрики. Подключите client/metrics_client.py в конвейере или загрузите замер из Jupyter: python -m scripts.import_metrics bench_result.json')]));
  const t = theme();
  seriesChart('tpu', t, data, nodes, 'tpu_util', { pct: true, ref: cap * 100, refLabel: `предел ${Math.round(cap * 100)}%` });
  seriesChart('fps', t, data, nodes, 'fps_stream_min', { ref: fps, refLabel: `цель ${fps} FPS` });
  $('#sys-updated').textContent = `обновлено ${fmtTime(new Date().toISOString(), false)} · каждые 5 с`;
}

async function loadSystem() {
  const p = { date_from: new Date(Date.now() - 30 * 864e5).toISOString() };
  const [h, q] = await Promise.all([api('/health'), api('/api/stats/quality', p), loadSystemLive()]);
  const o = q.overall;
  $('#sys-tiles').replaceChildren(
    tile('Сервис журнала', h.status === 'ok' ? 'Работает' : 'Сбой', `БД: ${h.db} · часовой пояс: ${h.plant_tz}`),
    tile('Задержка конвейера, медиана', fmtMs(o.pipeline_latency_ms_p50), `p95: ${fmtMs(o.pipeline_latency_ms_p95)}`),
    tile('Время до тревоги, медиана', fmtMs(o.time_to_alert_ms_p50), `p95: ${fmtMs(o.time_to_alert_ms_p95)} · включает подтверждение во времени`),
    tile('Точность тревог за 30 дней', fmtPct(o.precision), `ложных: ${fmtInt(o.false_positive)} из ${fmtInt(o.reviewed)} проверенных`),
  );
  const rows = Object.entries(q.by_pattern).sort((a, b) => b[1].events - a[1].events);
  $('#sys-table tbody').replaceChildren(...rows.map(([, r]) => el('tr', {},
    el('td', {}, r.name), el('td', { class: 'num' }, fmtInt(r.events)), el('td', { class: 'num' }, fmtInt(r.reviewed)),
    el('td', { class: 'num' }, fmtPct(r.precision)), el('td', { class: 'num' }, fmtMs(r.time_to_alert_ms_p50)),
    el('td', { class: 'num' }, fmtMs(r.time_to_alert_ms_p95)), el('td', { class: 'num' }, fmtMs(r.pipeline_latency_ms_p50)),
    el('td', { class: 'num' }, fmtMs(r.pipeline_latency_ms_p95)))));
}

/* ---------- запуск ---------- */
function bind() {
  window.addEventListener('hashchange', () => showTab(location.hash.slice(1)));
  $('#op-area').addEventListener('change', (e) => { state.opArea = e.target.value; storeSet('opArea', state.opArea); loadFeed().catch(showError); });
  $$('[data-op-scope]').forEach((b) => b.addEventListener('click', () => {
    state.opScope = b.dataset.opScope;
    $$('[data-op-scope]').forEach((x) => x.classList.toggle('on', x === b));
    loadFeed().catch(showError);
  }));
  $('#sound').addEventListener('click', (e) => {
    state.sound = !state.sound;
    e.target.textContent = state.sound ? 'Звук: вкл' : 'Звук: выкл';
    e.target.setAttribute('aria-pressed', String(state.sound));
  });
  $('#simulate').addEventListener('click', () => api('/api/dev/simulate', {}, { method: 'POST' }).catch(showError));
  $('#j-filters').addEventListener('submit', (e) => { e.preventDefault(); state.jOffset = 0; loadJournal().catch(showError); });
  $('#j-filters').addEventListener('reset', () => setTimeout(() => { state.jOffset = 0; loadJournal().catch(showError); }));
  $('#j-prev').addEventListener('click', () => { state.jOffset = Math.max(0, state.jOffset - state.jLimit); loadJournal().catch(showError); });
  $('#j-next').addEventListener('click', () => { state.jOffset += state.jLimit; loadJournal().catch(showError); });
  $('#drawer-close').addEventListener('click', () => { $('#drawer').hidden = true; });
  $$('[data-days]').forEach((b) => b.addEventListener('click', () => {
    state.days = Number(b.dataset.days);
    $$('[data-days]').forEach((x) => x.classList.toggle('on', x === b));
    loadAnalytics().catch(showError);
  }));
  ['#cap-fps', '#cap-util'].forEach((id) => $(id).addEventListener('change', () => loadSystemLive().catch(showError)));
  $('#an-area').addEventListener('change', (e) => { state.anArea = e.target.value; loadAnalytics().catch(showError); });
  $$('[data-table-toggle]').forEach((b) => b.addEventListener('click', () => {
    const card = b.closest('[data-chart]');
    const box = $('.tablebox', card); const chart = $('.chart', card);
    const toTable = box.hidden;
    box.hidden = !toTable; chart.hidden = toTable;
    b.textContent = toTable ? 'График' : 'Таблица';
    if (toTable) renderTable(card.dataset.chart); else state.charts[card.dataset.chart]?.resize();
  }));
  window.addEventListener('resize', () => Object.values(state.charts).forEach((c) => c.resize()));
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (state.tab === 'analytics') loadAnalytics().catch(showError);
    if (state.tab === 'system') loadSystemLive().catch(showError);
  });
}

async function init() {
  try {
    const h = await api('/health');
    TZ = h.plant_tz || TZ;
    $('#simulate').hidden = !h.demo_mode;
  } catch (err) { showError(err); }
  bind();
  await loadLookups().catch(showError);
  connectWS();
  showTab(location.hash.slice(1) || 'operator');
}
init();

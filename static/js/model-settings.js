/* Настройки модели: расширенные параметры GGUF (llama.cpp) с квантизацией.
 * Тот же UI, что и для Gradio: KV-cache, flash-attn, RoPE, штрафы, Mirostat. */
(function () {
  'use strict';

  const FIELDS = [
    // ── Память и контекст ──
    { key: 'ctx', label: 'Контекст, токенов', type: 'number',
      min: 512, max: 262144, step: 512,
      hint: 'Чем больше — тем больше KV-кэш в VRAM' },
    { key: 'batch', label: 'n_batch', type: 'number',
      min: 64, max: 4096, step: 64,
      hint: 'Размер батча промпта. Больше — быстрее, но больше VRAM' },
    { key: 'gpu_layers', label: 'Слоёв на GPU', type: 'number',
      min: -1, max: 999,
      hint: '-1 = все слои. Не влезает — уменьшите' },

    // ── Квантизация KV-кэша ──
    { key: 'kv_cache_type', label: 'Тип KV-кэша', type: 'select',
      options: ['f16', 'q8_0', 'q4_0', 'q4_1'],
      hint: 'q8_0 почти не влияет на качество, экономит VRAM' },

    // ── Flash / offload ──
    { key: 'flash_attn', label: 'Flash Attention', type: 'checkbox',
      hint: 'Экономит VRAM на длинном контексте' },
    { key: 'offload_kqv', label: 'Offload KQV на GPU', type: 'checkbox',
      hint: 'Переносит K/Q/V на GPU' },

    // ── Память ОС ──
    { key: 'use_mmap', label: 'mmap (файл в память)', type: 'checkbox',
      hint: 'Медленнее, но экономит RAM' },
    { key: 'use_mlock', label: 'mlock (не выгружать)', type: 'checkbox',
      hint: 'Запрещает ОС выгружать веса в swap' },

    // ── CPU ──
    { key: 'threads', label: 'Потоков CPU', type: 'number',
      min: 1, max: 128,
      hint: 'При полной GPU-загрузке почти не важно' },

    // ── GPU ──
    { key: 'main_gpu', label: 'Основная GPU', type: 'select',
      options: ['0', '1', '2', '3'],
      hint: 'Номер карты (как в nvidia-smi)' },
    { key: 'tensor_split', label: 'Доли по GPU', type: 'text',
      placeholder: 'например 12,6 (пусто = авто)',
      hint: 'Пропорции слоёв по картам' },

    // ── RoPE ──
    { key: 'rope_freq_base', label: 'RoPE freq base', type: 'number',
      min: 0, max: 1000000, step: 1,
      hint: '0 = из модели. Увеличение — для длинного контекста' },
    { key: 'rope_freq_scale', label: 'RoPE freq scale', type: 'number',
      min: 0, max: 100, step: 0.01,
      hint: '0 = авто' },

    // ── Сэмплинг по умолчанию (применяется как дефолт к chat) ──
    { key: 'temperature', label: 'temperature', type: 'number',
      min: 0, max: 2, step: 0.05 },
    { key: 'top_p', label: 'top_p', type: 'number',
      min: 0, max: 1, step: 0.01 },
    { key: 'top_k', label: 'top_k', type: 'number',
      min: 1, max: 200, step: 1 },
    { key: 'min_p', label: 'min_p', type: 'number',
      min: 0, max: 1, step: 0.01 },
    { key: 'typical_p', label: 'typical_p', type: 'number',
      min: 0, max: 1, step: 0.01 },
    { key: 'repeat_penalty', label: 'repeat_penalty', type: 'number',
      min: 1, max: 2, step: 0.01 },
    { key: 'frequency_penalty', label: 'frequency_penalty', type: 'number',
      min: 0, max: 2, step: 0.05 },
    { key: 'presence_penalty', label: 'presence_penalty', type: 'number',
      min: 0, max: 2, step: 0.05 },

    // ── Mirostat ──
    { key: 'mirostat_mode', label: 'Mirostat mode', type: 'select',
      options: ['0', '1', '2'],
      hint: '0 = выкл. 1 = mirostat, 2 = mirostat-v2' },
    { key: 'mirostat_tau', label: 'Mirostat tau', type: 'number',
      min: 0.1, max: 10, step: 0.1 },
    { key: 'mirostat_eta', label: 'Mirostat eta', type: 'number',
      min: 0.01, max: 1, step: 0.01 },

    { key: 'parallel', label: 'Параллельных слотов', type: 'number',
      min: 1, max: 8,
      hint: '1 — для одного пользователя. Больше = больше VRAM' },
  ];

  const CSS = `
  #msOverlay{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;z-index:9000;
    align-items:center;justify-content:center;padding:16px;box-sizing:border-box}
  #msOverlay.open{display:flex}
  #msModal{width:min(920px,100%);max-height:90vh;display:flex;flex-direction:column;
    background:linear-gradient(180deg,#17181c,#0d0e11);color:#eceef1;
    border:1px solid rgba(255,255,255,.09);border-radius:14px;
    box-shadow:0 20px 60px rgba(0,0,0,.6);font:14px/1.45 system-ui,sans-serif}
  #msModal header{display:flex;align-items:center;justify-content:space-between;
    padding:14px 18px;border-bottom:1px solid rgba(255,255,255,.08);
    background:linear-gradient(90deg,rgba(255,255,255,.03),rgba(255,255,255,0))}
  #msModal header h2{font-size:16px;margin:0;font-weight:600;
    background:linear-gradient(135deg,#5b626f,#a7afbc);
    -webkit-background-clip:text;background-clip:text;color:transparent}
  #msModal .ms-x{background:none;border:0;color:#a6abb5;font-size:22px;
    cursor:pointer;padding:0 4px}
  #msModal .ms-x:hover{color:#eceef1}
  #msBody{display:grid;grid-template-columns:260px 1fr;gap:0;min-height:0;
    flex:1;overflow:hidden}
  @media (max-width:720px){#msBody{grid-template-columns:1fr}}
  #msList{overflow:auto;border-right:1px solid rgba(255,255,255,.08);
    padding:8px;background:rgba(0,0,0,.15)}
  #msList .ms-item{display:block;width:100%;text-align:left;background:none;
    border:0;color:#eceef1;padding:9px 10px;border-radius:8px;cursor:pointer}
  #msList .ms-item:hover{background:rgba(255,255,255,.05)}
  #msList .ms-item.sel{background:linear-gradient(135deg,
    rgba(120,140,255,.18),rgba(80,90,180,.10))}
  #msList .ms-name{font-size:13px;word-break:break-all}
  #msList .ms-meta{font-size:11px;opacity:.65;margin-top:2px}
  #msList .ms-dot{display:inline-block;width:7px;height:7px;border-radius:50%;
    background:#4cc38a;margin-right:6px;box-shadow:0 0 8px rgba(76,195,138,.5)}
  #msList .ms-q{display:inline-block;font-family:ui-monospace,monospace;
    font-size:10.5px;color:#cfd5df;background:linear-gradient(135deg,#3a3f4a,#1f2228);
    padding:1px 6px;border-radius:5px;margin-left:4px;
    border:1px solid rgba(255,255,255,.1)}
  #msForm{overflow:auto;padding:14px 18px;min-height:0}
  #msForm .ms-row{display:grid;grid-template-columns:220px 1fr;gap:6px 14px;
    align-items:center;margin-bottom:10px}
  @media (max-width:520px){#msForm .ms-row{grid-template-columns:1fr}}
  #msForm label{font-weight:500;color:#c4cbd6;font-size:13px}
  #msForm .ms-hint{grid-column:2;font-size:11.5px;color:#787e89;margin-top:-4px}
  #msForm input[type=number],#msForm input[type=text],#msForm select{
    background:#0f1013;color:#eceef1;border:1px solid rgba(255,255,255,.15);
    border-radius:8px;padding:6px 8px;width:100%;box-sizing:border-box;font:inherit}
  #msForm input:focus,#msForm select:focus{
    outline:none;border-color:#b6bec9;
    box-shadow:0 0 0 3px rgba(182,190,201,.12)}
  #msForm .ms-empty{opacity:.7;padding:20px 0}
  #msFoot{display:flex;flex-wrap:wrap;gap:8px;align-items:center;
    padding:12px 18px;border-top:1px solid rgba(255,255,255,.08);
    background:rgba(0,0,0,.2)}
  #msFoot .ms-status{flex:1;min-width:180px;font-size:13px;opacity:.9}
  #msFoot .ms-status.err{color:#ff7b7b}
  #msFoot button{border:1px solid rgba(255,255,255,.15);background:none;
    color:#eceef1;border-radius:8px;padding:7px 12px;cursor:pointer;font:inherit;
    transition:background 150ms ease,border-color 150ms ease}
  #msFoot button:hover:not(:disabled){background:rgba(255,255,255,.06);
    border-color:rgba(255,255,255,.28)}
  #msFoot button.primary{background:linear-gradient(135deg,#3d4250,#22252c);
    border-color:rgba(255,255,255,.15);color:#fff;
    box-shadow:0 4px 15px rgba(0,0,0,.35)}
  #msFoot button.primary:hover:not(:disabled){
    background:linear-gradient(135deg,#4b5162,#2b2e35)}
  #msFoot button:disabled{opacity:.4;cursor:default}`;

  let state = null, selected = null, busy = false;
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function injectUi() {
    if ($('#msOverlay')) return;
    const style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);
    const wrap = document.createElement('div');
    wrap.id = 'msOverlay';
    wrap.innerHTML = `
      <div id="msModal" role="dialog" aria-modal="true" aria-labelledby="msTitle">
        <header>
          <h2 id="msTitle">Настройки модели · параметры квантизации</h2>
          <button class="ms-x" id="msClose" type="button" aria-label="Закрыть">×</button>
        </header>
        <div id="msBody">
          <div id="msList"></div>
          <div id="msForm"></div>
        </div>
        <div id="msFoot">
          <div class="ms-status" id="msStatus"></div>
          <button type="button" id="msReset">К умолчаниям</button>
          <button type="button" id="msUnload" disabled>Выгрузить из памяти</button>
          <button type="button" id="msSave" disabled>Сохранить</button>
          <button type="button" id="msLoad" class="primary" disabled>Сохранить и загрузить</button>
        </div>
      </div>`;
    document.body.appendChild(wrap);
    $('#msClose').addEventListener('click', close);
    wrap.addEventListener('click', (e) => { if (e.target === wrap && !busy) close(); });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && isOpen() && !busy) close();
    });
    $('#msReset').addEventListener('click', resetToDefaults);
    $('#msSave').addEventListener('click', () => save(false));
    $('#msLoad').addEventListener('click', () => save(true));
    $('#msUnload').addEventListener('click', unloadModel);
  }

  const isOpen = () => $('#msOverlay') && $('#msOverlay').classList.contains('open');
  function setStatus(text, isErr) {
    const s = $('#msStatus');
    s.textContent = text || '';
    s.classList.toggle('err', !!isErr);
  }
  function setBusy(v) {
    busy = v;
    ['#msSave', '#msLoad', '#msReset'].forEach((id) => {
      $(id).disabled = v || !editable();
    });
    $('#msUnload').disabled = v || !(state && state.loaded);
  }
  const editable = () => !!(state && state.editable);

  async function open() {
    injectUi();
    $('#msOverlay').classList.add('open');
    await reload();
  }
  function close() { $('#msOverlay').classList.remove('open'); }

  async function reload() {
    setStatus('Загрузка списка моделей…');
    try {
      const r = await fetch('/ai/model-settings',
        { credentials: 'same-origin', cache: 'no-store' });
      if (!r.ok) throw new Error(r.status === 401 ? 'Нужно войти' : `HTTP ${r.status}`);
      state = await r.json();
    } catch (e) {
      setStatus('Не удалось получить список: ' + e.message, true);
      return;
    }
    if (!selected || !state.models.some((m) => m.id === selected)) {
      selected = (state.loaded && state.models.find((m) => m.id === state.loaded)?.id)
              || (state.models[0] && state.models[0].id) || null;
    }
    renderList();
    renderForm();
    setStatus(state.editable ? '' : (state.note || ''));
  }

  function renderList() {
    const list = $('#msList');
    if (!state.models.length) {
      list.innerHTML = '<div class="ms-empty">Моделей не найдено' +
        (state.models_dir ? ` в ${esc(state.models_dir)}` : '') + '</div>';
      return;
    }
    list.innerHTML = state.models.map((m) => {
      const q = m.quant || '';
      return `<button class="ms-item ${m.id === selected ? 'sel' : ''}"
              type="button" data-id="${esc(m.id)}">
        <div class="ms-name" title="${esc(m.id)}">
          ${m.loaded ? '<span class="ms-dot" title="сейчас в памяти"></span>' : ''}
          ${esc(m.id.split('/').pop().replace(/\.gguf$/i, ''))}
          ${q ? `<span class="ms-q">${esc(q)}</span>` : ''}
        </div>
        <div class="ms-meta">${m.size_gb != null ? m.size_gb + ' ГБ' : ''}${m.saved ? ' · свои параметры' : ''}</div>
      </button>`;
    }).join('');
    list.querySelectorAll('.ms-item').forEach((b) => b.addEventListener('click', () => {
      if (busy) return;
      selected = b.dataset.id;
      renderList();
      renderForm();
      setStatus('');
    }));
  }

  const currentModel = () => state && state.models.find((m) => m.id === selected);

  function renderForm() {
    const form = $('#msForm');
    const m = currentModel();
    if (!m) {
      form.innerHTML = '<div class="ms-empty">Выберите модель слева.</div>';
      setBusy(false);
      return;
    }
    const p = m.params || state.defaults || {};
    const rows = FIELDS.map((f) => {
      const v = p[f.key];
      let input;
      if (f.type === 'checkbox') {
        input = `<input type="checkbox" id="f_${f.key}" ${v ? 'checked' : ''}>`;
      } else if (f.type === 'select') {
        const opts = f.options.map((o) =>
          `<option value="${o}" ${String(v) === o ? 'selected' : ''}>${o}</option>`
        ).join('');
        input = `<select id="f_${f.key}">${opts}</select>`;
      } else if (f.type === 'number') {
        input = `<input type="number" id="f_${f.key}" value="${esc(v)}"
                min="${f.min}" max="${f.max}" step="${f.step || 1}">`;
      } else {
        input = `<input type="text" id="f_${f.key}" value="${esc(v)}"
                placeholder="${esc(f.placeholder || '')}">`;
      }
      return `<div class="ms-row">
        <label for="f_${f.key}">${esc(f.label)}</label>${input}
        ${f.hint ? `<div class="ms-hint">${esc(f.hint)}</div>` : ''}
      </div>`;
    }).join('');

    const q = m.quant ? `<span class="ms-q">${esc(m.quant)}</span>` : '';
    const info = m.loaded
      ? '<div class="ms-hint" style="grid-column:1/-1;margin-bottom:10px;color:#4cc38a">Сейчас загружена в llama-server.</div>'
      : '';
    form.innerHTML = `<div style="margin-bottom:12px">
      <b>${esc(m.id)}</b>${q}${m.size_gb != null ? ` · ${m.size_gb} ГБ` : ''}
      </div>${info}${rows}`;
    setBusy(false);
  }

  function readForm() {
    const out = {};
    for (const f of FIELDS) {
      const el = $('#f_' + f.key);
      if (!el) continue;
      if (f.type === 'checkbox') out[f.key] = !!el.checked;
      else if (f.type === 'number') out[f.key] = parseFloat(el.value);
      else if (f.type === 'select') out[f.key] = el.value;
      else out[f.key] = (el.value || '').trim();
    }
    return out;
  }

  function resetToDefaults() {
    if (!state || !state.defaults) return;
    const d = state.defaults;
    for (const f of FIELDS) {
      const el = $('#f_' + f.key);
      if (!el) continue;
      if (f.type === 'checkbox') el.checked = !!d[f.key];
      else el.value = d[f.key];
    }
    setStatus('Значения по умолчанию подставлены. Нажмите «Сохранить».');
  }

  async function save(loadNow) {
    const m = currentModel();
    if (!m || busy) return;
    const params = readForm();
    setBusy(true);
    setStatus(loadNow ? 'Сохраняю и загружаю модель — может занять минуты…'
                      : 'Сохраняю…');
    try {
      const r1 = await fetch('/ai/model-settings', {
        method: 'PUT', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: m.id, params }),
      });
      const d1 = await r1.json().catch(() => ({}));
      if (!r1.ok) throw new Error(d1.detail || `HTTP ${r1.status}`);
      if (!loadNow) {
        setStatus('Параметры сохранены. Применятся при следующей загрузке.');
        await reload();
        return;
      }
      const r2 = await fetch('/ai/models/select', {
        method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: m.id, params }),
      });
      const d2 = await r2.json().catch(() => ({}));
      if (r2.status === 409) {
        setStatus('Идёт ответ. Модель загрузится автоматически после него.');
      } else if (!r2.ok) {
        throw new Error(d2.error || d2.message || `HTTP ${r2.status}`);
      } else {
        try { localStorage.setItem('chatModel', m.id); } catch (_) {}
        setStatus(d2.changed === false
          ? 'Модель уже загружена с этими параметрами.'
          : 'Модель загружена. Обновите страницу (F5).');
      }
      await reload();
    } catch (e) {
      setStatus('Ошибка: ' + e.message, true);
    } finally {
      setBusy(false);
    }
  }

  async function unloadModel() {
    if (busy || !state || !state.loaded) return;
    setBusy(true);
    setStatus('Выгружаю модель…');
    let msg = 'Модель выгружена, память освобождена.';
    let err = false;
    try {
      const r = await fetch('/ai/model-settings/unload',
        { method: 'POST', credentials: 'same-origin' });
      const d = await r.json().catch(() => ({}));
      if (r.status === 409) {
        msg = 'Идёт ответ. Выгрузка недоступна до его окончания.';
        err = true;
      } else if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
    } catch (e) {
      msg = 'Ошибка: ' + e.message;
      err = true;
    }
    setBusy(false);
    await reload();
    setStatus(msg, err);
  }

  function bindButton() {
    const btn = $('#modelSettingsBtn');
    if (btn) btn.addEventListener('click', open);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bindButton);
  } else {
    bindButton();
  }
})();
# HTML for /train (kept separate from server.py for readability).

TRAIN_PAGE_HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>YOLOX 学習</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root { font-family: system-ui, sans-serif; }
    body { max-width: 58rem; margin: 1.5rem auto; padding: 0 1rem; line-height: 1.45; }
    nav { margin-bottom: 1rem; font-size: 0.95rem; }
    nav a { margin-right: 1rem; }
    label { display: block; font-weight: 600; margin-top: 0.75rem; }
    input, select { max-width: 100%; padding: 0.35rem 0.5rem; font-size: 1rem; }
    input[type="number"] { max-width: 10rem; }
    .row { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-end; margin-top: 0.5rem; }
    .row > div { flex: 1 1 12rem; }
    button { margin-top: 1rem; margin-right: 0.5rem; padding: 0.45rem 0.9rem; font-size: 1rem; cursor: pointer; }
    #log { width: 100%; height: 12rem; font-family: ui-monospace, monospace; font-size: 0.78rem; overflow: auto; background: #111; color: #e8e8e8; padding: 0.5rem; border-radius: 4px; white-space: pre-wrap; }
    .chart-panel { margin-top: 1rem; max-width: 100%; height: 240px; position: relative; }
    .muted { color: #555; font-size: 0.88rem; }
    .err { color: #a22; }
    .ok { color: #080; }
    .status { margin-top: 0.5rem; font-weight: 600; }
    #metricChecks { margin-top: 0.5rem; padding: 0.5rem 0; border: 1px solid #ddd; border-radius: 4px; max-height: 10rem; overflow: auto; }
    #metricChecks label { display: inline-block; margin: 0 0.75rem 0.35rem 0; font-weight: 400; font-size: 0.88rem; }
    .data-panel { margin-top: 0.5rem; padding: 0.75rem; border: 1px solid #ddd; border-radius: 6px; background: #fafafa; }
    .data-panel h2 { margin: 0 0 0.5rem; font-size: 1.05rem; }
    select[multiple].source-select { width: 100%; min-height: 9rem; font-family: ui-monospace, monospace; font-size: 0.82rem; }
    .work-dir-name-row { display: flex; align-items: stretch; flex-wrap: wrap; max-width: 36rem; margin-top: 0.25rem; }
    .work-dir-prefix {
      display: flex; align-items: center; font-family: ui-monospace, monospace; font-size: 1rem;
      padding: 0.35rem 0.55rem; background: #f0f0f0; border: 1px solid #ccc; border-right: none;
      border-radius: 4px 0 0 4px; color: #333;
    }
    #categoryChecks { margin-top: 0.5rem; padding: 0.5rem 0; border: 1px solid #ddd; border-radius: 4px; max-height: 9rem; overflow: auto; }
    #categoryChecks label { display: inline-block; margin: 0 0.75rem 0.35rem 0; font-weight: 400; font-size: 0.88rem; }
    #work_dir_suffix { flex: 1 1 12rem; min-width: 10rem; border-radius: 0 4px 4px 0; }
  </style>
</head>
<body>
  <nav><a href="/hub">統括</a><a href="/">CVAT export</a><a href="/train">YOLOX 学習</a><a href="/nuclio-deploy">Nuclio デプロイ</a></nav>
  <h1>YOLOX-S ファインチューン</h1>
  <p class="muted">メトリックは <code>work_dirs/.../vis_data/scalars.json</code> から読み取ります（学習開始後しばらくでファイルが現れます）。</p>

  <label for="work_dir_suffix">出力 work_dir 名</label>
  <div class="work-dir-name-row">
    <span class="work-dir-prefix" id="work_dir_prefix">web_train_</span>
    <input id="work_dir_suffix" type="text" autocomplete="off" spellcheck="false" aria-describedby="work_dir_suffix_help" />
  </div>
  <p id="work_dir_suffix_help" class="muted">空欄のときは <code>web_train_YYYY_MMDD_HHMM_xxxxxxxx</code> を自動生成します。接尾辞は英数字と <code>_</code> <code>-</code> のみ使用できます。</p>

  <label for="config_path">MMDetection config</label>
  <select id="config_path"></select>
  <p class="muted">既定は <code>configs/yolox/yolox_s_finetune.py</code>（CVAT エクスポート向け）。他モデルはデータセット構造が異なる場合があります。</p>

  <section class="data-panel">
    <h2>Train データ</h2>
    <p class="muted">Ctrl / Cmd + クリックで複数選択。別のエクスポートフォルダも混在できます。</p>
    <select id="train_sources" class="source-select" multiple aria-label="Train データ"></select>
  </section>

  <section class="data-panel">
    <h2>Val データ</h2>
    <p class="muted">検証用アノテーションを複数選択できます（評価時は COCO JSON をマージします）。</p>
    <select id="val_sources" class="source-select" multiple aria-label="Val データ"></select>
  </section>

  <section class="data-panel">
    <h2>カテゴリ</h2>
    <p class="muted">Train / Val データの選択から自動で候補を出します。チェックを外すとそのクラスは学習・評価しません。</p>
    <div id="categoryChecks"><span class="muted">（Train または Val を選択してください）</span></div>
  </section>

  <label for="pretrained">事前学習重み（work_dirs）</label>
  <select id="pretrained"></select>
  <p class="muted">未選択時は config の COCO 事前学習 URL を使用します。選択時は <code>last_checkpoint</code> の .pth を <code>model.init_cfg.checkpoint</code> に設定します。</p>

  <div class="row">
    <div><label for="max_epochs">max_epochs</label><input id="max_epochs" type="number" min="1" value="50" /></div>
    <div><label for="lr">学習率 (lr)</label><input id="lr" type="number" step="any" value="0.001" /></div>
    <div><label for="batch_size">batch_size</label><input id="batch_size" type="number" min="1" value="4" /></div>
  </div>

  <div>
    <button type="button" id="btnStart">学習開始</button>
    <button type="button" id="btnStop">停止</button>
  </div>
  <div id="status" class="status"></div>
  <div id="workdir" class="muted"></div>
  <div id="scalarpath" class="muted"></div>

  <h2>メトリック（折れ線）</h2>
  <p class="muted">初期表示: <strong>loss</strong> / <strong>memory</strong> / <strong>coco/bbox_mAP</strong> のみ。下のチェックで loss_* ・ memory ・ lr ・ time ・ coco/bbox_mAP_* などを追加できます。</p>
  <div id="metricChecks"></div>

  <h3>loss 系（loss, loss_cls, …）</h3>
  <div class="chart-panel"><canvas id="cLoss"></canvas></div>
  <h3>COCO bbox mAP（validation）</h3>
  <div class="chart-panel"><canvas id="cMap"></canvas></div>
  <h3>memory / lr / time など</h3>
  <div class="chart-panel"><canvas id="cSys"></canvas></div>

  <h2>ログ</h2>
  <div id="log"></div>

  <script>
    const STORAGE_KEY = 'mmplatform_train_selected_metrics';
    let datasets = [];
    let lastState = {};
    let defaultMetrics = ['loss', 'memory', 'coco/bbox_mAP'];
    const chartRefs = { loss: null, map: null, sys: null };
    let prevAvailKey = '';
    let prevWorkDirForCharts = undefined;
    let suggestedCategoryNames = [];
    let categoryFetchTimer = null;

    function buildSourceOptions(dsList) {
      const opts = [];
      for (const d of dsList) {
        const prefixes = d.image_prefixes && d.image_prefixes.length ? d.image_prefixes : ['images/'];
        for (const ann of d.annotations || []) {
          for (const prefix of prefixes) {
            const item = { data_root_rel: d.data_root_rel, ann_file: ann, img_prefix: prefix };
            opts.push({
              value: JSON.stringify(item),
              label: d.id + '  |  ' + ann + '  |  ' + prefix,
            });
          }
        }
      }
      return opts;
    }

    function fillSourceSelect(el, sourceOptions) {
      el.innerHTML = '';
      if (!sourceOptions.length) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.disabled = true;
        opt.textContent = '(data/exports に COCO がありません)';
        el.appendChild(opt);
        return;
      }
      for (const o of sourceOptions) {
        const opt = document.createElement('option');
        opt.value = o.value;
        opt.textContent = o.label;
        el.appendChild(opt);
      }
    }

    function getSelectedSources(selectId) {
      const el = document.getElementById(selectId);
      const out = [];
      for (const opt of el.selectedOptions) {
        if (!opt.value) continue;
        try { out.push(JSON.parse(opt.value)); } catch (e) {}
      }
      return out;
    }

    function getSelectedCategories() {
      const box = document.getElementById('categoryChecks');
      if (!box) return [];
      return [...box.querySelectorAll('input[type=checkbox][data-category]:checked')].map(i => i.dataset.category);
    }

    function renderCategoryChecks(payload) {
      const box = document.getElementById('categoryChecks');
      const rows = (payload && payload.categories) || [];
      suggestedCategoryNames = (payload && payload.names) || [];
      if (!rows.length) {
        box.innerHTML = '<span class="muted">（Train または Val を選択してください）</span>';
        return;
      }
      const prev = new Set(getSelectedCategories());
      const defaultSel = new Set((payload.default_selected || suggestedCategoryNames));
      box.innerHTML = '';
      for (const row of rows) {
        const lab = document.createElement('label');
        const inp = document.createElement('input');
        inp.type = 'checkbox';
        inp.dataset.category = row.name;
        const tags = [];
        if (row.in_train) tags.push('train');
        if (row.in_val) tags.push('val');
        const checked = prev.size ? prev.has(row.name) : defaultSel.has(row.name);
        inp.checked = checked;
        lab.appendChild(inp);
        lab.appendChild(document.createTextNode(' ' + row.name + (tags.length ? ' (' + tags.join(', ') + ')' : '')));
        box.appendChild(lab);
      }
    }

    async function refreshSuggestedCategories() {
      const train_sources = getSelectedSources('train_sources');
      const val_sources = getSelectedSources('val_sources');
      if (!train_sources.length && !val_sources.length) {
        renderCategoryChecks({ categories: [], names: [], default_selected: [] });
        return;
      }
      try {
        const r = await fetch('/api/train/suggest-categories', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ train_sources, val_sources }),
        });
        const j = await r.json();
        if (!r.ok) {
          const detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail);
          document.getElementById('categoryChecks').innerHTML = '<span class="err">' + detail + '</span>';
          return;
        }
        renderCategoryChecks(j);
      } catch (e) {
        document.getElementById('categoryChecks').innerHTML = '<span class="err">カテゴリ取得に失敗しました</span>';
      }
    }

    function scheduleCategoryRefresh() {
      if (categoryFetchTimer) clearTimeout(categoryFetchTimer);
      categoryFetchTimer = setTimeout(refreshSuggestedCategories, 250);
    }

    function populateSourceSelects() {
      const opts = buildSourceOptions(datasets);
      fillSourceSelect(document.getElementById('train_sources'), opts);
      fillSourceSelect(document.getElementById('val_sources'), opts);
      scheduleCategoryRefresh();
    }

    async function loadDefaultWorkDirName() {
      const el = document.getElementById('work_dir_suffix');
      const prefixEl = document.getElementById('work_dir_prefix');
      try {
        const r = await fetch('/api/train/default-work-dir');
        const j = await r.json();
        if (r.ok) {
          if (j.work_dir_prefix) prefixEl.textContent = j.work_dir_prefix;
          const example = j.work_dir_suffix_example || '';
          if (example) el.placeholder = example + '（空欄で自動）';
        }
      } catch (e) {}
    }

    let defaultConfigId = 'configs/yolox/yolox_s_finetune.py';

    async function loadMmdetConfigs() {
      const sel = document.getElementById('config_path');
      sel.innerHTML = '';
      try {
        const r = await fetch('/api/train/configs');
        const j = await r.json();
        const rows = j.configs || [];
        if (j.default_config) defaultConfigId = j.default_config;
        if (!rows.length) {
          const opt = document.createElement('option');
          opt.value = defaultConfigId;
          opt.textContent = defaultConfigId;
          sel.appendChild(opt);
          return;
        }
        const groups = {};
        for (const row of rows) {
          const g = row.group || 'other';
          if (!groups[g]) groups[g] = [];
          groups[g].push(row);
        }
        const groupNames = Object.keys(groups).sort((a, b) => {
          if (a === 'yolox') return -1;
          if (b === 'yolox') return 1;
          return a.localeCompare(b);
        });
        for (const g of groupNames) {
          const og = document.createElement('optgroup');
          og.label = g;
          for (const row of groups[g]) {
            const opt = document.createElement('option');
            opt.value = row.id;
            opt.textContent = row.label;
            if (row.is_default) opt.textContent += ' (default)';
            og.appendChild(opt);
          }
          sel.appendChild(og);
        }
        sel.value = defaultConfigId;
      } catch (e) {
        const opt = document.createElement('option');
        opt.value = defaultConfigId;
        opt.textContent = defaultConfigId + ' (default)';
        sel.appendChild(opt);
      }
    }

    async function loadPretrainedModels() {
      const sel = document.getElementById('pretrained');
      sel.innerHTML = '';
      const def = document.createElement('option');
      def.value = '';
      def.textContent = '（デフォルト: COCO 事前学習 URL）';
      sel.appendChild(def);
      const r = await fetch('/api/train/pretrained-models');
      const j = await r.json();
      const rows = j.pretrained_models || [];
      if (!rows.length) {
        const none = document.createElement('option');
        none.value = '';
        none.disabled = true;
        none.textContent = '（work_dirs に利用可能なチェックポイントがありません）';
        sel.appendChild(none);
        return;
      }
      for (const row of rows) {
        const opt = document.createElement('option');
        opt.value = row.id;
        opt.textContent = row.id + '  —  ' + row.checkpoint;
        sel.appendChild(opt);
      }
    }

    async function loadDatasets() {
      const r = await fetch('/api/train/datasets');
      const j = await r.json();
      datasets = j.datasets || [];
      populateSourceSelects();
    }

    function chartBucket(key) {
      if (key === 'loss' || key.startsWith('loss_')) return 'loss';
      if (key.indexOf('bbox_mAP') !== -1 || key.indexOf('coco/bbox') === 0) return 'map';
      return 'sys';
    }

    function buildChartPayload(metricSeries, selectedKeys, bucket) {
      const steps = new Set();
      for (const k of selectedKeys) {
        if (chartBucket(k) !== bucket) continue;
        (metricSeries[k] || []).forEach(p => steps.add(p.step));
      }
      const labels = Array.from(steps).sort((a, b) => a - b);
      const palette = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
      const datasetsOut = [];
      let ci = 0;
      for (const k of selectedKeys) {
        if (chartBucket(k) !== bucket) continue;
        const pts = metricSeries[k] || [];
        const byStep = {};
        pts.forEach(p => { byStep[String(p.step)] = p.value; });
        const data = labels.map(s => (Object.prototype.hasOwnProperty.call(byStep, String(s)) ? byStep[String(s)] : null));
        datasetsOut.push({
          label: k,
          data: data,
          borderColor: palette[ci++ % palette.length],
          backgroundColor: 'transparent',
          tension: 0.12,
          pointRadius: 0,
          spanGaps: false
        });
      }
      return { labels, datasets: datasetsOut };
    }

    function chartOptions(yTitle, showLegend) {
      return {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { display: showLegend, position: 'bottom' } },
        scales: {
          x: { title: { display: true, text: 'step' } },
          y: { title: { display: true, text: yTitle } }
        }
      };
    }

    function renderOneChart(refKey, canvasId, metricSeries, selectedKeys, yTitle) {
      const el = document.getElementById(canvasId);
      if (!el) return;
      const { labels, datasets } = buildChartPayload(metricSeries, selectedKeys, refKey);
      const ctx = el.getContext('2d');
      const ch = chartRefs[refKey];
      if (!ch) {
        chartRefs[refKey] = new Chart(ctx, {
          type: 'line',
          data: { labels, datasets },
          options: chartOptions(yTitle, datasets.length > 0)
        });
        return;
      }
      ch.data.labels = labels;
      ch.data.datasets = datasets;
      ch.options.plugins.legend.display = datasets.length > 0;
      if (ch.options.scales && ch.options.scales.y && ch.options.scales.y.title) {
        ch.options.scales.y.title.text = yTitle;
      }
      ch.update('none');
    }

    function getSelectedMetrics() {
      const box = document.getElementById('metricChecks');
      if (!box) return [];
      return [...box.querySelectorAll('input[type=checkbox]:checked')].map(i => i.dataset.metric);
    }

    function destroyChart(refKey) {
      if (chartRefs[refKey]) {
        chartRefs[refKey].destroy();
        chartRefs[refKey] = null;
      }
    }

    function renderChartsFromState(j) {
      const wd = j.work_dir != null ? j.work_dir : null;
      if (wd !== prevWorkDirForCharts) {
        prevWorkDirForCharts = wd;
        ['loss', 'map', 'sys'].forEach(destroyChart);
      }
      const ms = j.metric_series || {};
      const sel = getSelectedMetrics();
      const yLoss = 'loss 系';
      const yMap = 'mAP';
      const ySys = 'memory / lr / time …';
      renderOneChart('loss', 'cLoss', ms, sel, yLoss);
      renderOneChart('map', 'cMap', ms, sel, yMap);
      renderOneChart('sys', 'cSys', ms, sel, ySys);
    }

    function rebuildCheckboxes(available) {
      const box = document.getElementById('metricChecks');
      if (!available || !available.length) {
        box.innerHTML = '<span class="muted">（scalars 未生成）</span>';
        return;
      }
      let saved = null;
      try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null'); } catch (e) {}
      const fromSaved = Array.isArray(saved) ? saved.filter(m => available.includes(m)) : [];
      let selected = new Set(fromSaved);
      if (selected.size === 0) {
        defaultMetrics.forEach(m => { if (available.includes(m)) selected.add(m); });
      }
      box.innerHTML = '';
      for (const m of available) {
        const safeId = 'mc_' + m.replace(/[^a-zA-Z0-9]+/g, '_');
        const lab = document.createElement('label');
        const inp = document.createElement('input');
        inp.type = 'checkbox';
        inp.dataset.metric = m;
        inp.id = safeId;
        inp.checked = selected.has(m);
        inp.addEventListener('change', () => {
          const all = [...box.querySelectorAll('input[type=checkbox]:checked')].map(i => i.dataset.metric);
          localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
          renderChartsFromState(lastState);
        });
        lab.appendChild(inp);
        lab.appendChild(document.createTextNode(' ' + m));
        box.appendChild(lab);
      }
    }

    async function poll() {
      const r = await fetch('/api/train/state');
      const j = await r.json();
      lastState = j;
      if (Array.isArray(j.default_metrics) && j.default_metrics.length) defaultMetrics = j.default_metrics;
      const st = j.status || 'idle';
      document.getElementById('status').textContent = '状態: ' + st;
      document.getElementById('status').className = 'status ' + (st === 'failed' ? 'err' : (st === 'completed' ? 'ok' : ''));
      document.getElementById('workdir').textContent = j.work_dir ? ('work_dir: ' + j.work_dir) : '';
      document.getElementById('scalarpath').textContent = j.scalars_path ? ('scalars: ' + j.scalars_path) : '';
      if (j.error) document.getElementById('status').textContent += ' — ' + j.error;
      document.getElementById('log').textContent = (j.lines_tail || []).join('\\n');
      const logEl = document.getElementById('log');
      logEl.scrollTop = logEl.scrollHeight;

      const avail = j.available_metrics || [];
      const key = avail.join('|');
      if (key !== prevAvailKey) {
        prevAvailKey = key;
        rebuildCheckboxes(avail);
      }
      renderChartsFromState(j);
    }

    document.getElementById('train_sources').addEventListener('change', scheduleCategoryRefresh);
    document.getElementById('val_sources').addEventListener('change', scheduleCategoryRefresh);

    document.getElementById('btnStart').addEventListener('click', async () => {
      const train_sources = getSelectedSources('train_sources');
      const val_sources = getSelectedSources('val_sources');
      if (!train_sources.length) { alert('Train データを 1 件以上選択してください'); return; }
      if (!val_sources.length) { alert('Val データを 1 件以上選択してください'); return; }
      const cfgSel = document.getElementById('config_path').value;
      const body = {
        train_sources: train_sources,
        val_sources: val_sources,
        max_epochs: parseInt(document.getElementById('max_epochs').value, 10),
        lr: parseFloat(document.getElementById('lr').value),
        batch_size: parseInt(document.getElementById('batch_size').value, 10),
      };
      if (cfgSel) body.config_path = cfgSel;
      const pw = document.getElementById('pretrained').value;
      if (pw) body.pretrained_work_dir = pw;
      const wdn = document.getElementById('work_dir_suffix').value.trim();
      if (wdn) body.work_dir_name = wdn;
      const r = await fetch('/api/train/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const j = await r.json();
      if (!r.ok) {
        alert(typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail));
        return;
      }
      prevAvailKey = '';
      poll();
    });

    document.getElementById('btnStop').addEventListener('click', async () => {
      await fetch('/api/train/stop', { method: 'POST' });
      poll();
    });

    Promise.all([loadDatasets(), loadMmdetConfigs(), loadPretrainedModels(), loadDefaultWorkDirName()]).then(() => { poll(); setInterval(poll, 1200); });
  </script>
</body>
</html>
"""

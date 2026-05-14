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
  </style>
</head>
<body>
  <nav><a href="/">CVAT export</a><a href="/train">YOLOX 学習</a></nav>
  <h1>YOLOX-S ファインチューン</h1>
  <p class="muted">メトリックは <code>work_dirs/.../vis_data/scalars.json</code> から読み取ります（学習開始後しばらくでファイルが現れます）。</p>

  <label for="dataset">データセット（data/exports）</label>
  <select id="dataset"></select>

  <div class="row">
    <div>
      <label for="train_ann">train アノテーション JSON</label>
      <select id="train_ann"></select>
    </div>
    <div>
      <label for="val_ann">val アノテーション JSON</label>
      <select id="val_ann"></select>
    </div>
  </div>
  <div class="row">
    <div>
      <label for="train_prefix">train 画像プレフィックス</label>
      <select id="train_prefix"></select>
    </div>
    <div>
      <label for="val_prefix">val 画像プレフィックス</label>
      <select id="val_prefix"></select>
    </div>
  </div>

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

    function fillSelect(el, options, getv) {
      el.innerHTML = '';
      for (const o of options) {
        const opt = document.createElement('option');
        opt.value = getv(o);
        opt.textContent = typeof o === 'string' ? o : (o.label || o.value);
        el.appendChild(opt);
      }
    }

    function onDatasetChange() {
      const id = document.getElementById('dataset').value;
      const d = datasets.find(x => x.id === id);
      if (!d) return;
      fillSelect(document.getElementById('train_ann'), d.annotations, x => x);
      fillSelect(document.getElementById('val_ann'), d.annotations, x => x);
      fillSelect(document.getElementById('train_prefix'), d.image_prefixes, x => x);
      fillSelect(document.getElementById('val_prefix'), d.image_prefixes, x => x);
    }

    async function loadDatasets() {
      const r = await fetch('/api/train/datasets');
      const j = await r.json();
      datasets = j.datasets || [];
      const sel = document.getElementById('dataset');
      sel.innerHTML = '';
      if (!datasets.length) {
        sel.innerHTML = '<option value="">(data/exports に COCO がありません)</option>';
        return;
      }
      for (const d of datasets) {
        const opt = document.createElement('option');
        opt.value = d.id;
        opt.textContent = d.id;
        sel.appendChild(opt);
      }
      onDatasetChange();
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

    document.getElementById('dataset').addEventListener('change', onDatasetChange);

    document.getElementById('btnStart').addEventListener('click', async () => {
      const ds = document.getElementById('dataset').value;
      if (!ds) { alert('データセットを選択してください'); return; }
      const d = datasets.find(x => x.id === ds);
      const body = {
        data_root_rel: d.data_root_rel,
        train_ann: document.getElementById('train_ann').value,
        val_ann: document.getElementById('val_ann').value,
        train_img_prefix: document.getElementById('train_prefix').value,
        val_img_prefix: document.getElementById('val_prefix').value,
        max_epochs: parseInt(document.getElementById('max_epochs').value, 10),
        lr: parseFloat(document.getElementById('lr').value),
        batch_size: parseInt(document.getElementById('batch_size').value, 10),
      };
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

    loadDatasets().then(() => { poll(); setInterval(poll, 1200); });
  </script>
</body>
</html>
"""

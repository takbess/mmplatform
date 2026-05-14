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
    body { max-width: 56rem; margin: 1.5rem auto; padding: 0 1rem; line-height: 1.45; }
    nav { margin-bottom: 1rem; font-size: 0.95rem; }
    nav a { margin-right: 1rem; }
    label { display: block; font-weight: 600; margin-top: 0.75rem; }
    input, select { max-width: 100%; padding: 0.35rem 0.5rem; font-size: 1rem; }
    input[type="number"] { max-width: 10rem; }
    .row { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-end; margin-top: 0.5rem; }
    .row > div { flex: 1 1 12rem; }
    button { margin-top: 1rem; margin-right: 0.5rem; padding: 0.45rem 0.9rem; font-size: 1rem; cursor: pointer; }
    #log { width: 100%; height: 14rem; font-family: ui-monospace, monospace; font-size: 0.78rem; overflow: auto; background: #111; color: #e8e8e8; padding: 0.5rem; border-radius: 4px; white-space: pre-wrap; }
    #chartWrap { margin-top: 1rem; max-width: 100%; height: 280px; }
    .muted { color: #555; font-size: 0.88rem; }
    .err { color: #a22; }
    .ok { color: #080; }
    .status { margin-top: 0.5rem; font-weight: 600; }
  </style>
</head>
<body>
  <nav><a href="/">CVAT export</a><a href="/train">YOLOX 学習</a></nav>
  <h1>YOLOX-S ファインチューン</h1>
  <p class="muted">data/exports 配下の COCO 展開ディレクトリを選び、<code>yolox_s_finetune.py</code> 相当の設定で
    <code>python -m webui.mmdet_train_worker</code> を起動します（ログは下記にストリーム表示）。</p>

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

  <h2>loss（train）</h2>
  <div id="chartWrap"><canvas id="lossChart"></canvas></div>

  <h2>ログ</h2>
  <div id="log"></div>

  <script>
    let datasets = [];
    let chart = null;

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

    function ensureChart() {
      const ctx = document.getElementById('lossChart').getContext('2d');
      if (chart) { chart.destroy(); chart = null; }
      chart = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'loss', data: [], borderColor: '#36a', tension: 0.1, pointRadius: 0 }] },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: { x: { title: { display: true, text: 'step' } }, y: { title: { display: true, text: 'loss' } } }
        }
      });
    }

    function updateChart(points) {
      if (!chart) ensureChart();
      chart.data.labels = points.map(p => p.step);
      chart.data.datasets[0].data = points.map(p => p.loss);
      chart.update('none');
    }

    async function poll() {
      const r = await fetch('/api/train/state');
      const j = await r.json();
      const st = j.status || 'idle';
      document.getElementById('status').textContent = '状態: ' + st;
      document.getElementById('status').className = 'status ' + (st === 'failed' ? 'err' : (st === 'completed' ? 'ok' : ''));
      document.getElementById('workdir').textContent = j.work_dir ? ('work_dir: ' + j.work_dir) : '';
      if (j.error) document.getElementById('status').textContent += ' — ' + j.error;
      document.getElementById('log').textContent = (j.lines_tail || []).join('\\n');
      const logEl = document.getElementById('log');
      logEl.scrollTop = logEl.scrollHeight;
      if (j.loss_points && j.loss_points.length) updateChart(j.loss_points);
    }

    document.getElementById('dataset').addEventListener('change', onDatasetChange);

    document.getElementById('btnStart').addEventListener('click', async () => {
      const ds = document.getElementById('dataset').value;
      if (!ds) { alert('データセットを選択してください'); return; }
      const d = datasets.find(x => x.id === ds);
      ensureChart();
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
      poll();
    });

    document.getElementById('btnStop').addEventListener('click', async () => {
      await fetch('/api/train/stop', { method: 'POST' });
      poll();
    });

    loadDatasets().then(() => { ensureChart(); poll(); setInterval(poll, 1200); });
  </script>
</body>
</html>
"""

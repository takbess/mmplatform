# HTML for /nuclio-deploy (separate from server.py for readability).

NUCLIO_DEPLOY_PAGE_HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CVAT Nuclio デプロイ</title>
  <style>
    :root { font-family: system-ui, sans-serif; }
    body { max-width: 56rem; margin: 1.5rem auto; padding: 0 1rem; line-height: 1.5; }
    nav { margin-bottom: 1rem; font-size: 0.95rem; }
    nav a { margin-right: 1rem; }
    label { display: block; font-weight: 600; margin-top: 0.75rem; }
    select { width: 100%; max-width: 100%; padding: 0.4rem 0.5rem; font-size: 1rem; }
    button { margin-top: 0.75rem; padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; }
    #log { margin-top: 1rem; width: 100%; min-height: 14rem; font-family: ui-monospace, monospace; font-size: 0.78rem; white-space: pre-wrap; background: #111; color: #e8e8e8; padding: 0.6rem; border-radius: 4px; overflow: auto; }
    .muted { color: #555; font-size: 0.88rem; }
    .ok { color: #0a6; font-weight: 600; }
    .err { color: #a20; font-weight: 600; }
    #meta { margin-top: 0.75rem; font-size: 0.9rem; }
    #detail { margin-top: 1rem; border: 1px solid #ddd; border-radius: 6px; padding: 0.75rem 1rem; background: #fafafa; font-size: 0.9rem; }
    #detail h3 { margin: 0 0 0.5rem; font-size: 1rem; }
    #detail section + section { margin-top: 1rem; padding-top: 0.75rem; border-top: 1px solid #e5e5e5; }
    .metric-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(10rem, 1fr)); gap: 0.5rem 1rem; }
    .metric-grid dt { font-weight: 600; font-size: 0.82rem; color: #444; margin: 0; }
    .metric-grid dd { margin: 0.1rem 0 0; font-family: ui-monospace, monospace; font-size: 0.88rem; }
    .spec-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    .spec-table th, .spec-table td { text-align: left; padding: 0.25rem 0.5rem 0.25rem 0; vertical-align: top; }
    .spec-table th { color: #444; font-weight: 600; white-space: nowrap; width: 11rem; }
    .spec-table td { font-family: ui-monospace, monospace; word-break: break-all; }
    .badge { display: inline-block; background: #e8f4ea; color: #0a5; padding: 0.1rem 0.45rem; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    .badge-warn { background: #f4f0e8; color: #850; }
  </style>
</head>
<body>
  <nav>
    <a href="/hub">統括</a>
    <a href="/">CVAT export</a>
    <a href="/train">YOLOX 学習</a>
    <a href="/nuclio-deploy">Nuclio デプロイ</a>
  </nav>
  <h1>CVAT 自動アノテ用 Nuclio デプロイ</h1>
  <p class="muted">
    <code>work_dirs/web_train_*</code> を選び、<code>nuctl deploy</code> まで実行します。
    このマシンに <strong>nuctl</strong> と <strong>Docker</strong> があり、CVAT の serverless 用ネットワーク設定が
    <code>third_party/cvat/serverless/deploy_gpu.sh</code> と同じ前提である必要があります。
  </p>

  <label for="wd">work_dir（フォルダ名）</label>
  <select id="wd"></select>
  <div><button type="button" id="btnRefresh">一覧を再読込</button></div>

  <div id="detail" class="muted">モデルを選択すると学習設定と最終メトリックが表示されます。</div>

  <div>
    <button type="button" id="btnDeploy">選択した work_dir をデプロイ</button>
  </div>
  <div id="status"></div>
  <div id="meta"></div>
  <div id="log"></div>

  <script>
    const wd = document.getElementById('wd');
    const log = document.getElementById('log');
    const status = document.getElementById('status');
    const meta = document.getElementById('meta');
    const detail = document.getElementById('detail');
    let workDirs = [];

    async function readResponse(r) {
      const raw = await r.text();
      try {
        return { ok: r.ok, raw, json: raw ? JSON.parse(raw) : {} };
      } catch (_) {
        return { ok: r.ok, raw, json: null };
      }
    }

    function formatHttpError(r, raw, json) {
      if (json && json.detail !== undefined) return typeof json.detail === 'string' ? json.detail : JSON.stringify(json.detail);
      if (raw && raw.trim()) return raw.trim().slice(0, 8000);
      return 'HTTP ' + r.status;
    }

    function fmtNum(v, digits) {
      if (v == null || Number.isNaN(v)) return '—';
      if (typeof v === 'number') return v.toFixed(digits);
      return String(v);
    }

    function optionLabel(row) {
      const m = row.summary && row.summary.metrics && row.summary.metrics.final_val;
      const t = row.summary && row.summary.metrics && row.summary.metrics.final_train;
      let s = row.id;
      if (m && m['coco/bbox_mAP'] != null) s += '  |  mAP ' + fmtNum(m['coco/bbox_mAP'], 3);
      if (t && t.loss != null) s += '  |  loss ' + fmtNum(t.loss, 3);
      return s;
    }

    function renderMetricGrid(title, obj, keys, labels) {
      if (!obj || !Object.keys(obj).length) return '';
      let html = '<section><h3>' + title + '</h3><dl class="metric-grid">';
      for (let i = 0; i < keys.length; i++) {
        const k = keys[i];
        if (obj[k] == null) continue;
        const lab = labels ? labels[i] : k;
        let val = obj[k];
        if (k.indexOf('mAP') !== -1 || k === 'loss' || String(k).indexOf('loss_') === 0) val = fmtNum(val, k.indexOf('mAP') !== -1 ? 3 : 4);
        else if (k === 'epoch' || k === 'iter' || k === 'step') val = String(Math.round(val));
        else val = fmtNum(val, 6);
        html += '<dt>' + lab + '</dt><dd>' + val + '</dd>';
      }
      html += '</dl></section>';
      return html;
    }

    function renderSpecTable(spec) {
      if (!spec) return '<section><h3>学習設定（webui_train_spec.json）</h3><p class="muted">webui_train_spec.json がありません。</p></section>';
      const order = ['max_epochs', 'lr', 'batch_size', 'data_root', 'train_sources', 'val_sources', 'train_ann', 'val_ann', 'train_img_prefix', 'val_img_prefix', 'pretrained_work_dir', 'config_path', 'work_dir'];
      const labels = {
        max_epochs: 'max_epochs', lr: 'lr', batch_size: 'batch_size', data_root: 'data_root',
        train_sources: 'train データ', val_sources: 'val データ',
        train_ann: 'train_ann', val_ann: 'val_ann', train_img_prefix: 'train_img_prefix', val_img_prefix: 'val_img_prefix',
        pretrained_work_dir: '事前学習 work_dir', config_path: 'config_path', work_dir: 'work_dir'
      };
      let html = '<section><h3>学習設定（webui_train_spec.json）</h3><table class="spec-table"><tbody>';
      const seen = new Set();
      for (const k of order) {
        if (spec[k] == null) continue;
        seen.add(k);
        html += '<tr><th>' + (labels[k] || k) + '</th><td>' + String(spec[k]) + '</td></tr>';
      }
      for (const k of Object.keys(spec)) {
        if (seen.has(k)) continue;
        html += '<tr><th>' + k + '</th><td>' + String(spec[k]) + '</td></tr>';
      }
      html += '</tbody></table></section>';
      return html;
    }

    function renderDetail(row) {
      if (!row) {
        detail.innerHTML = '<span class="muted">モデルを選択してください。</span>';
        detail.className = 'muted';
        return;
      }
      const s = row.summary || {};
      const spec = s.train_spec;
      const metrics = s.metrics;
      const m = metrics && metrics.final_val;
      const headline = m && m['coco/bbox_mAP'] != null
        ? '<p><span class="badge">coco/bbox_mAP ' + fmtNum(m['coco/bbox_mAP'], 3) + '</span></p>'
        : '<p><span class="badge badge-warn">検証 mAP 未取得</span></p>';

      let html = headline;
      html += '<p class="muted">ckpt: <code>' + row.checkpoint + '</code>  |  nuclio: <code>' + row.nuclio_function + '</code>';
      if (s.classes && s.classes.length) html += '  |  classes: ' + JSON.stringify(s.classes);
      html += '</p>';
      html += renderSpecTable(spec);
      if (metrics && metrics.scalars_path) {
        html += '<p class="muted">scalars: <code>' + metrics.scalars_path + '</code></p>';
      }
      const trainKeys = ['epoch', 'iter', 'step', 'loss', 'loss_cls', 'loss_bbox', 'loss_obj', 'loss_l1', 'lr'];
      const trainLabels = ['epoch', 'iter', 'step', 'loss', 'loss_cls', 'loss_bbox', 'loss_obj', 'loss_l1', 'lr'];
      const valKeys = ['step', 'coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75', 'coco/bbox_mAP_s', 'coco/bbox_mAP_m', 'coco/bbox_mAP_l'];
      const valLabels = ['step', 'mAP', 'mAP@50', 'mAP@75', 'mAP (s)', 'mAP (m)', 'mAP (l)'];
      if (metrics && metrics.final_train) {
        html += renderMetricGrid('最終 train（scalars.json 末尾付近）', metrics.final_train, trainKeys, trainLabels);
      }
      if (metrics && metrics.final_val) {
        html += renderMetricGrid('最終 validation（COCO bbox）', metrics.final_val, valKeys, valLabels);
      } else if (!metrics) {
        html += '<section><h3>メトリック</h3><p class="muted">scalars.json が見つかりません。</p></section>';
      }
      detail.innerHTML = html;
      detail.className = '';
    }

    function onSelectChange() {
      const id = wd.value;
      const row = workDirs.find(x => x.id === id);
      renderDetail(row);
    }

    async function loadDirs() {
      wd.innerHTML = '';
      const r = await fetch('/api/nuclio/work-dirs', { headers: { Accept: 'application/json' } });
      const { ok, raw, json } = await readResponse(r);
      if (!ok) {
        status.className = 'err';
        status.textContent = formatHttpError(r, raw, json);
        workDirs = [];
        renderDetail(null);
        return;
      }
      status.textContent = '';
      const j = json || {};
      workDirs = j.work_dirs || [];
      if (!workDirs.length) {
        const o = document.createElement('option');
        o.value = '';
        o.textContent = '（デプロイ可能な work_dir がありません）';
        wd.appendChild(o);
        renderDetail(null);
        return;
      }
      for (const row of workDirs) {
        const o = document.createElement('option');
        o.value = row.id;
        o.textContent = optionLabel(row);
        wd.appendChild(o);
      }
      onSelectChange();
    }

    wd.addEventListener('change', onSelectChange);
    document.getElementById('btnRefresh').addEventListener('click', loadDirs);
    document.getElementById('btnDeploy').addEventListener('click', async () => {
      const id = wd.value;
      if (!id) return;
      log.textContent = '';
      meta.textContent = '';
      status.className = '';
      status.textContent = 'デプロイ中…（Docker ビルドで数分かかることがあります）';
      const btn = document.getElementById('btnDeploy');
      btn.disabled = true;
      try {
        const r = await fetch('/api/nuclio/deploy', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
          body: JSON.stringify({ work_dir: id }),
        });
        const { ok: httpOk, raw, json } = await readResponse(r);
        if (!httpOk) {
          status.className = 'err';
          status.textContent = formatHttpError(r, raw, json);
          log.textContent = raw || '';
          return;
        }
        const j = json || {};
        status.className = j.ok ? 'ok' : 'err';
        status.textContent = j.ok ? 'デプロイ完了' : 'デプロイ失敗（終了コード ' + j.exit_code + '）';
        meta.textContent =
          'nuclio: ' + (j.nuclio_function || '') +
          '  |  image: ' + (j.docker_image || '') +
          '  |  classes: ' + JSON.stringify(j.classes || []);
        log.textContent = j.log || '';
        loadDirs();
      } catch (e) {
        status.className = 'err';
        status.textContent = String(e);
      } finally {
        btn.disabled = false;
      }
    });

    loadDirs();
  </script>
</body>
</html>
"""

"""Small FastAPI UI to run CVAT COCO export by task id (extensible for training config later)."""

from __future__ import annotations

import asyncio
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdk.exportv2 import default_export_zip, run_export  # noqa: E402
from webui.train_manager import DEFAULT_CHART_METRICS, list_export_datasets, train_manager  # noqa: E402
from webui.train_page import TRAIN_PAGE_HTML  # noqa: E402

app = FastAPI(title="mmplatform webui", version="0.2.0")
_executor = ThreadPoolExecutor(max_workers=2)


class ExportRequest(BaseModel):
    task_id: int = Field(..., ge=1, description="CVAT task id")


class TrainStartRequest(BaseModel):
    data_root_rel: str = Field(..., description="Relative path under repo root, e.g. data/exports/task_1_coco")
    train_ann: str = Field(..., description="Annotation path relative to data_root")
    val_ann: str = Field(..., description="Annotation path relative to data_root")
    train_img_prefix: str = Field(..., description="Image prefix relative to data_root")
    val_img_prefix: str = Field(..., description="Image prefix relative to data_root")
    max_epochs: int = Field(50, ge=1, le=10000)
    lr: float = Field(0.001, gt=0, le=100.0)
    batch_size: int = Field(4, ge=1, le=256)
    config_path: Optional[str] = Field(
        default=None, description="Optional path to an mmdet config under third_party/mmdetection"
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CVAT export</title>
  <style>
    :root { font-family: system-ui, sans-serif; }
    body { max-width: 42rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
    label { display: block; font-weight: 600; margin-bottom: 0.25rem; }
    input { width: 100%; max-width: 12rem; padding: 0.4rem 0.5rem; font-size: 1rem; }
    button { margin-top: 0.75rem; padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; }
    #preview { margin-top: 1rem; font-size: 0.9rem; color: #333; }
    #result { margin-top: 1rem; white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 0.85rem; }
    .ok { color: #0a6; }
    .err { color: #a20; }
    .muted { color: #666; font-size: 0.85rem; margin-top: 2rem; }
  </style>
</head>
<body>
  <nav style="margin-bottom:1rem;font-size:0.95rem"><a href="/">CVAT export</a> &middot; <a href="/train">YOLOX 学習</a></nav>
  <h1>CVAT COCO export</h1>
  <p class="muted">ZIP は <code>data/exports/task_{task-id}_{YYYY_MMDD_HHMM}.zip</code> を自動で使います。</p>
  <form id="f">
    <label for="task_id">Task ID</label>
    <input id="task_id" name="task_id" type="number" min="1" value="1" required />
    <div><button type="submit" id="go">Export</button></div>
  </form>
  <div id="preview"></div>
  <div id="result"></div>
  <script>
    const taskInput = document.getElementById('task_id');
    const preview = document.getElementById('preview');
    const result = document.getElementById('result');
    const go = document.getElementById('go');

    async function refreshPreview() {
      const id = parseInt(taskInput.value, 10);
      if (!id || id < 1) { preview.textContent = ''; return; }
      const r = await fetch('/api/preview-out?task_id=' + id);
      const j = await r.json();
      if (!r.ok) { preview.textContent = ''; return; }
      preview.innerHTML = '<strong>出力予定</strong><br>ZIP: <code>' + j.out_zip + '</code><br>展開先: <code>' + j.extract_dir + '</code>';
    }
    taskInput.addEventListener('change', refreshPreview);
    taskInput.addEventListener('input', refreshPreview);
    refreshPreview();

    document.getElementById('f').addEventListener('submit', async (e) => {
      e.preventDefault();
      result.textContent = '';
      const task_id = parseInt(taskInput.value, 10);
      go.disabled = true;
      result.textContent = '実行中…（CVAT へ接続・ダウンロードに時間がかかることがあります）';
      try {
        const r = await fetch('/api/export', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ task_id }),
        });
        const j = await r.json();
        if (!r.ok) {
          result.className = 'err';
          result.textContent = j.detail || JSON.stringify(j);
          return;
        }
        result.className = 'ok';
        result.textContent = '完了\\nZIP: ' + j.out_zip + '\\n展開先: ' + j.extract_dir;
        refreshPreview();
      } catch (err) {
        result.className = 'err';
        result.textContent = String(err);
      } finally {
        go.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


@app.get("/api/preview-out")
def preview_out(task_id: int) -> dict:
    if task_id < 1:
        raise HTTPException(status_code=400, detail="task_id must be >= 1")
    out = default_export_zip(task_id)
    extract_dir = out.parent / out.stem
    return {"out_zip": str(out), "extract_dir": str(extract_dir)}


@app.post("/api/export")
async def api_export(body: ExportRequest) -> dict:
    def _job():
        out_zip, extract_dir = run_export(body.task_id, None)
        return str(out_zip), str(extract_dir)

    loop = asyncio.get_running_loop()
    try:
        out_zip_s, extract_dir_s = await loop.run_in_executor(_executor, _job)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{exc!s}\n{tb}") from exc

    return {"ok": True, "out_zip": out_zip_s, "extract_dir": extract_dir_s}


@app.get("/train", response_class=HTMLResponse)
def train_page() -> str:
    return TRAIN_PAGE_HTML


@app.get("/api/train/datasets")
def api_train_datasets() -> dict:
    return {"datasets": list_export_datasets()}


@app.get("/api/train/state")
def api_train_state() -> dict:
    snap = train_manager.active_snapshot()
    if snap is None:
        return {
            "status": "idle",
            "lines_tail": [],
            "loss_points": [],
            "metric_series": {},
            "available_metrics": [],
            "default_metrics": list(DEFAULT_CHART_METRICS),
            "scalars_path": None,
        }
    return snap


@app.post("/api/train/start")
def api_train_start(body: TrainStartRequest) -> dict:
    try:
        job = train_manager.start(
            data_root_rel=body.data_root_rel,
            train_ann=body.train_ann,
            val_ann=body.val_ann,
            train_img_prefix=body.train_img_prefix,
            val_img_prefix=body.val_img_prefix,
            max_epochs=body.max_epochs,
            lr=body.lr,
            batch_size=body.batch_size,
            config_path=body.config_path,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "job_id": job.job_id, "work_dir": job.work_dir}


@app.post("/api/train/stop")
def api_train_stop() -> dict:
    ok = train_manager.stop()
    return {"ok": ok}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8765)


if __name__ == "__main__":
    main()

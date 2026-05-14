"""Training job lifecycle: discover exports, spawn MMDet worker, parse logs."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR = ROOT / "data" / "exports"
MMDET_ROOT = ROOT / "third_party" / "mmdetection"
DEFAULT_CONFIG = MMDET_ROOT / "configs" / "yolox" / "yolox_s_finetune.py"
WORK_DIR_PARENT = ROOT / "work_dirs"

EPOCH_TRAIN_RE = re.compile(
    r"Epoch\(train\)\s+\[(\d+)\]\[\s*(\d+)/(\d+)\]"
)
LOSS_TOKEN_RE = re.compile(r"\bloss:\s+([\d.]+(?:e[+-]?\d+)?)", re.IGNORECASE)


def export_stamp() -> str:
    return datetime.now().strftime("%Y_%m%d_%H%M")


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def list_export_datasets() -> List[Dict[str, Any]]:
    if not EXPORTS_DIR.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for p in sorted(EXPORTS_DIR.iterdir()):
        if not p.is_dir() or p.name.startswith(".") or p.name == ".trash":
            continue
        ann_dir = p / "annotations"
        if not ann_dir.is_dir():
            continue
        jsons = sorted(
            f.relative_to(p).as_posix()
            for f in ann_dir.glob("instances*.json")
        )
        if not jsons:
            continue
        prefixes: List[str] = []
        im = p / "images"
        if im.is_dir():
            for sub in sorted(im.iterdir()):
                if sub.is_dir():
                    prefixes.append(f"images/{sub.name}/")
        if not prefixes:
            prefixes.append("images/")
        rows.append(
            {
                "id": p.name,
                "data_root_rel": str(p.relative_to(ROOT)),
                "annotations": jsons,
                "image_prefixes": prefixes,
            }
        )
    return rows


def _parse_train_loss_line(line: str) -> Optional[Dict[str, Any]]:
    m = EPOCH_TRAIN_RE.search(line)
    if not m:
        return None
    lm = LOSS_TOKEN_RE.search(line)
    if not lm:
        return None
    epoch = int(m.group(1))
    it = int(m.group(2))
    total = int(m.group(3))
    try:
        loss = float(lm.group(1))
    except ValueError:
        return None
    return {"epoch": epoch, "iter": it, "iter_total": total, "loss": loss}


@dataclass
class TrainJob:
    job_id: str
    status: str  # idle starting running completed failed stopped
    lines: Deque[str] = field(default_factory=lambda: deque(maxlen=12000))
    loss_points: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=2000))
    error: Optional[str] = None
    work_dir: Optional[str] = None
    command: Optional[List[str]] = None
    _proc: Optional[subprocess.Popen] = None
    _reader_t: Optional[threading.Thread] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _loss_step: int = field(default=0, init=False, repr=False)

    def append_line(self, text: str) -> None:
        with self._lock:
            self.lines.append(text.rstrip("\n"))
            parsed = _parse_train_loss_line(text)
            if parsed is not None:
                self._loss_step += 1
                self.loss_points.append({"step": self._loss_step, **parsed})

    def snapshot(self, tail: int = 400) -> Dict[str, Any]:
        with self._lock:
            lst = list(self.lines)
            lp = list(self.loss_points)
        if tail > 0 and len(lst) > tail:
            lst = lst[-tail:]
        return {
            "job_id": self.job_id,
            "status": self.status,
            "work_dir": self.work_dir,
            "command": self.command,
            "error": self.error,
            "lines_tail": lst,
            "loss_points": lp,
        }


def _normalize_ann_relpath(data_root: Path, ann: str) -> str:
    """MMDet は data_root からの相対パス（例: annotations/instances_*.json）を想定する。"""
    rel = ann.strip().replace("\\", "/").lstrip("/")
    if (data_root / rel).is_file():
        return rel
    base = Path(rel).name
    under_ann = f"annotations/{base}"
    if (data_root / under_ann).is_file():
        return under_ann
    return rel


class TrainJobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: Optional[TrainJob] = None

    def active_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._job is None:
                return None
            return self._job.snapshot()

    def start(
        self,
        *,
        data_root_rel: str,
        train_ann: str,
        val_ann: str,
        train_img_prefix: str,
        val_img_prefix: str,
        max_epochs: int,
        lr: float,
        batch_size: int,
        config_path: Optional[str] = None,
    ) -> TrainJob:
        with self._lock:
            if self._job is not None and self._job.status == "running":
                raise RuntimeError("既に学習が実行中です。完了するまで待つか、停止してください。")

        data_root = (ROOT / data_root_rel).resolve()
        if not _is_under(data_root, EXPORTS_DIR.resolve()):
            raise ValueError("data_root は data/exports 配下のみ選択できます。")

        train_ann_n = _normalize_ann_relpath(data_root, train_ann)
        val_ann_n = _normalize_ann_relpath(data_root, val_ann)
        for name, rel in (
            (train_ann, data_root / train_ann_n),
            (val_ann, data_root / val_ann_n),
        ):
            if not rel.is_file():
                raise ValueError(f"アノテーションが見つかりません: {name}")

        raw = config_path or str(DEFAULT_CONFIG)
        cfg_path = Path(raw)
        if not cfg_path.is_absolute():
            cfg_path = (ROOT / cfg_path).resolve()
        else:
            cfg_path = cfg_path.resolve()
        if not cfg_path.is_file():
            raise ValueError(f"config が見つかりません: {cfg_path}")
        if not _is_under(cfg_path, MMDET_ROOT.resolve()):
            raise ValueError("config は third_party/mmdetection 配下のファイルを指定してください。")

        WORK_DIR_PARENT.mkdir(parents=True, exist_ok=True)
        work_dir = WORK_DIR_PARENT / f"web_train_{export_stamp()}_{uuid.uuid4().hex[:8]}"
        work_dir.mkdir(parents=True, exist_ok=False)
        if not _is_under(work_dir, WORK_DIR_PARENT.resolve()):
            raise ValueError("work_dir の配置が不正です。")

        spec = {
            "config_path": str(cfg_path),
            "data_root": str(data_root),
            "train_ann": train_ann_n,
            "val_ann": val_ann_n,
            "train_img_prefix": train_img_prefix,
            "val_img_prefix": val_img_prefix,
            "max_epochs": max_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "work_dir": str(work_dir),
        }

        spec_path = work_dir / "webui_train_spec.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "webui.mmdet_train_worker",
            str(spec_path),
        ]
        env = os.environ.copy()
        sep = os.pathsep
        extra = sep.join([str(MMDET_ROOT), str(ROOT)])
        env["PYTHONPATH"] = extra + sep + env.get("PYTHONPATH", "")
        env.setdefault("LOCAL_RANK", "0")

        job = TrainJob(job_id=uuid.uuid4().hex[:12], status="running", work_dir=str(work_dir), command=cmd)

        def _reader() -> None:
            try:
                job._proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert job._proc.stdout is not None
                for line in job._proc.stdout:
                    job.append_line(line)
                code = job._proc.wait()
                with job._lock:
                    if job.status == "stopped":
                        return
                    job.status = "completed" if code == 0 else "failed"
                    if code != 0:
                        job.error = f"プロセス終了コード {code}"
            except Exception as exc:
                with job._lock:
                    job.status = "failed"
                    job.error = str(exc)

        t = threading.Thread(target=_reader, daemon=True)
        job._reader_t = t

        with self._lock:
            self._job = job
        t.start()
        return job

    def stop(self) -> bool:
        with self._lock:
            job = self._job
        if job is None or job.status != "running":
            return False
        proc = job._proc
        if proc is None:
            return False
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        with job._lock:
            job.status = "stopped"
        return True


train_manager = TrainJobManager()

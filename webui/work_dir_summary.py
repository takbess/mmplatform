"""Summarize work_dir training spec and final scalars for Web UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent

SPEC_DISPLAY_KEYS = (
    "config_path",
    "data_root",
    "classes",
    "train_sources",
    "val_sources",
    "train_ann",
    "val_ann",
    "train_img_prefix",
    "val_img_prefix",
    "max_epochs",
    "lr",
    "batch_size",
    "pretrained_work_dir",
    "pretrained_checkpoint",
    "work_dir",
)

TRAIN_METRIC_KEYS = ("epoch", "iter", "step", "loss", "loss_cls", "loss_bbox", "loss_obj", "loss_l1", "lr")
VAL_METRIC_KEYS = (
    "step",
    "coco/bbox_mAP",
    "coco/bbox_mAP_50",
    "coco/bbox_mAP_75",
    "coco/bbox_mAP_s",
    "coco/bbox_mAP_m",
    "coco/bbox_mAP_l",
)


def find_scalars_json(work_dir: Path) -> Optional[Path]:
    if not work_dir.is_dir():
        return None
    candidates = list(work_dir.glob("*/vis_data/scalars.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _rel_under_root(path_str: str) -> str:
    try:
        p = Path(path_str)
        if not p.is_absolute():
            return path_str
        return str(p.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return path_str


def _pick_numeric(obj: Dict[str, Any], keys: tuple) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in keys:
        if k not in obj:
            continue
        v = obj[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        out[k] = v
    return out


def summarize_scalars(path: Path) -> Dict[str, Any]:
    """最終 train 行（loss あり）と最終 validation 行（coco/bbox_mAP あり）を返す。"""
    last_train: Optional[Dict[str, Any]] = None
    last_val: Optional[Dict[str, Any]] = None
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"scalars_path": str(path), "error": str(exc)}

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "coco/bbox_mAP" in obj:
            last_val = obj
        if "loss" in obj:
            last_train = obj

    return {
        "scalars_path": str(path.relative_to(ROOT)) if _is_under(path, ROOT) else str(path),
        "final_train": _pick_numeric(last_train or {}, TRAIN_METRIC_KEYS) or None,
        "final_val": _pick_numeric(last_val or {}, VAL_METRIC_KEYS) or None,
    }


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def load_train_spec_summary(work_dir: Path) -> Optional[Dict[str, Any]]:
    spec_path = work_dir / "webui_train_spec.json"
    if not spec_path.is_file():
        return None
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(spec, dict):
        return None

    out: Dict[str, Any] = {}
    for k in SPEC_DISPLAY_KEYS:
        if k not in spec:
            continue
        v = spec[k]
        if k in ("config_path", "data_root", "work_dir", "pretrained_checkpoint") and isinstance(v, str):
            out[k] = _rel_under_root(v)
        elif k in ("train_sources", "val_sources") and isinstance(v, list):
            rows = []
            for item in v:
                if not isinstance(item, dict):
                    continue
                row = dict(item)
                if isinstance(row.get("data_root"), str):
                    row["data_root"] = _rel_under_root(row["data_root"])
                rows.append(row)
            out[k] = rows
        else:
            out[k] = v
    return out or None


def summarize_work_dir(work_dir: Path, *, classes: Optional[List[str]] = None) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"train_spec": load_train_spec_summary(work_dir)}
    if classes:
        summary["classes"] = classes
    sp = find_scalars_json(work_dir)
    if sp is not None and sp.is_file():
        summary["metrics"] = summarize_scalars(sp)
    else:
        summary["metrics"] = None
    return summary

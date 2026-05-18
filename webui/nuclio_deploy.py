"""Stage MMDetection checkpoints from work_dirs and deploy a Nuclio function (nuctl)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from webui.work_dir_summary import summarize_work_dir

ROOT = Path(__file__).resolve().parent.parent
WORK_DIR_PARENT = ROOT / "work_dirs"
MMDET_ROOT = (ROOT / "third_party" / "mmdetection").resolve()

WORK_DIR_NAME_RE = re.compile(r"^web_train_[A-Za-z0-9_-]+$")

METAINFO_CLASSES_RE = re.compile(
    r"metainfo\s*=\s*dict\s*\(\s*classes\s*=\s*\(([^)]*)\)\s*\)",
    re.DOTALL,
)

NUCLIO_MAIN_PY = r'''# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT
# Staged by mmplatform webui (nuclio_deploy).

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules


def _parse_event_body(body):
    if isinstance(body, (bytes, bytearray)):
        return json.loads(body.decode("utf-8"))
    if isinstance(body, str):
        return json.loads(body)
    return body


def _decode_image_bgr(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64, validate=False)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot decode image bytes (cv2.imdecode returned None)")
    return img


def _find_assets(work_dir: Path) -> tuple[str, str]:
    cfg = work_dir / "model_deploy.py"
    if not cfg.is_file():
        raise RuntimeError(f"Missing model_deploy.py under {work_dir}")
    last_file = work_dir / "last_checkpoint"
    if not last_file.is_file():
        raise RuntimeError(f"Missing last_checkpoint under {work_dir}")
    raw = last_file.read_text(encoding="utf-8").strip()
    if not raw:
        raise RuntimeError(f"Empty last_checkpoint in {last_file}")
    ckpt = Path(raw) if Path(raw).is_absolute() else work_dir / raw
    if not ckpt.is_file():
        raise RuntimeError(f"Checkpoint not found: {ckpt} (from last_checkpoint {raw!r})")
    return str(cfg), str(ckpt)


def init_context(context):
    context.logger.info("Init context...  0%")
    register_all_modules()
    config_file, checkpoint_file = _find_assets(Path("/opt/nuclio"))
    context.logger.info("Loading config=%s checkpoint=%s", config_file, checkpoint_file)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = init_detector(config_file, checkpoint_file, device=device)
    context.user_data.model = model
    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run MMDetection staged model")
    data = _parse_event_body(event.body)
    threshold = float(data.get("threshold", 0.5))
    img = _decode_image_bgr(data["image"])
    model = context.user_data.model
    result = inference_detector(model, img)
    pred_instances = result.pred_instances
    classes = model.dataset_meta["classes"]
    h, w = img.shape[:2]
    results = []
    if len(pred_instances) > 0:
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy().astype(int)
        for box, score, label_idx in zip(bboxes, scores, labels):
            if float(score) < threshold:
                continue
            xtl, ytl, xbr, ybr = box
            xtl = max(int(xtl), 0)
            ytl = max(int(ytl), 0)
            xbr = min(int(xbr), w)
            ybr = min(int(ybr), h)
            label_name = classes[int(label_idx)]
            results.append(
                {
                    "confidence": str(float(score)),
                    "label": label_name,
                    "points": [xtl, ytl, xbr, ybr],
                    "type": "rectangle",
                }
            )
    return context.Response(
        body=json.dumps(results), headers={}, content_type="application/json", status_code=200
    )
'''


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _validate_work_dir_basename(name: str) -> str:
    n = name.strip()
    if not n or not WORK_DIR_NAME_RE.match(n):
        raise ValueError(
            "work_dir は work_dirs 直下のフォルダ名のみ指定できます（例: web_train_2026_0515_0016_0b4e73df）。"
        )
    return n


def resolve_work_dir(name: str) -> Path:
    n = _validate_work_dir_basename(name)
    wd = (WORK_DIR_PARENT / n).resolve()
    if not wd.is_dir():
        raise ValueError(f"work_dir が見つかりません: {n}")
    if wd.parent != WORK_DIR_PARENT.resolve():
        raise ValueError("不正な work_dir パスです。")
    return wd


def find_training_config(work_dir: Path) -> Path:
    dumped = work_dir / "yolox_s_finetune.py"
    if dumped.is_file():
        return dumped
    spec_path = work_dir / "webui_train_spec.json"
    if not spec_path.is_file():
        raise ValueError(
            "学習用 config が見つかりません。"
            " work_dir に yolox_s_finetune.py があるか、webui_train_spec.json がある必要があります。"
        )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    raw = spec.get("config_path")
    if not raw:
        raise ValueError("webui_train_spec.json に config_path がありません。")
    cfg = Path(raw)
    if not cfg.is_absolute():
        cfg = (ROOT / cfg).resolve()
    else:
        cfg = cfg.resolve()
    if not cfg.is_file():
        raise ValueError(f"config が見つかりません: {cfg}")
    if not _is_under(cfg, MMDET_ROOT):
        raise ValueError("config は third_party/mmdetection 配下のみ許可されています。")
    return cfg


def infer_classes_from_config_py(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = METAINFO_CLASSES_RE.search(text)
    if not m:
        return ["person"]
    inner = m.group(1)
    pairs = re.findall(r"'([^']*)'|\"([^\"]*)\"", inner)
    classes = [a or b for a, b in pairs if (a or b)]
    return classes if classes else ["person"]


def _resolve_checkpoint_path(work_dir: Path) -> Tuple[Path, str]:
    """Return (absolute checkpoint path, basename for staging)."""
    last = work_dir / "last_checkpoint"
    raw = last.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError("last_checkpoint が空です。")
    if Path(raw).is_absolute():
        src = Path(raw)
    else:
        src = (work_dir / raw).resolve()
    if not src.is_file():
        raise ValueError(f"チェックポイントが見つかりません: {src}")
    return src, src.name


def nuclio_function_id(work_dir_basename: str) -> str:
    h = hashlib.sha256(work_dir_basename.encode("utf-8")).hexdigest()[:12]
    return f"pth-mmplt-wt-{h}"


def docker_image_tag(work_dir_basename: str) -> str:
    h = hashlib.sha256(work_dir_basename.encode("utf-8")).hexdigest()[:12]
    return f"cvat.pth.mmplatform.wt.{h}:latest-gpu"


def build_cvat_detector_spec(classes: List[str]) -> str:
    rows = [
        {"id": i, "name": name, "type": "rectangle"}
        for i, name in enumerate(classes)
    ]
    return json.dumps(rows, ensure_ascii=False, indent=2)


def render_function_gpu_yaml(
    *,
    nuclio_name: str,
    display_name: str,
    spec_json: str,
    image_tag: str,
) -> str:
    # spec_json is pre-formatted JSON array; indent inside YAML block scalar
    indented = "\n".join("      " + line for line in spec_json.splitlines())
    return f"""metadata:
  name: {nuclio_name}
  namespace: cvat
  annotations:
    name: {display_name}
    type: detector
    spec: |
{indented}

spec:
  description: MMDetection checkpoint staged from mmplatform work_dir
  runtime: 'python:3.10'
  handler: main:handler
  eventTimeout: 30s
  build:
    image: {image_tag}
    baseImage: pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

    directives:
      preCopy:
        - kind: RUN
          value: apt-get update && apt-get install --no-install-recommends -y wget libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 libgl1 && rm -rf /var/lib/apt/lists/*
        - kind: RUN
          value: pip install --no-cache-dir -U openmim "numpy<2"
        - kind: RUN
          value: mim install mmengine "mmcv>=2.0.1,<2.2.0" "mmdet==3.3.0"
        - kind: RUN
          value: pip install --no-cache-dir "numpy<2"
        - kind: WORKDIR
          value: /opt/nuclio

  triggers:
    myHttpTrigger:
      numWorkers: 1
      kind: 'http'
      workerAvailabilityTimeoutMilliseconds: 10000
      attributes:
        maxRequestBodySize: 33554432

  resources:
    limits:
      nvidia.com/gpu: 1

  platform:
    attributes:
      restartPolicy:
        name: always
        maximumRetryCount: 3
      mountMode: volume
"""


def stage_bundle(work_dir: Path) -> Path:
    """Write _nuclio_bundle under work_dir; returns bundle path."""
    cfg_src = find_training_config(work_dir)
    classes = infer_classes_from_config_py(cfg_src)
    ckpt_src, ckpt_base = _resolve_checkpoint_path(work_dir)

    bundle = work_dir / "_nuclio_bundle"
    if bundle.is_dir():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True, exist_ok=False)

    shutil.copy2(cfg_src, bundle / "model_deploy.py")
    shutil.copy2(ckpt_src, bundle / ckpt_base)
    (bundle / "last_checkpoint").write_text(f"{ckpt_base}\n", encoding="utf-8")
    (bundle / "main.py").write_text(NUCLIO_MAIN_PY, encoding="utf-8")

    nid = nuclio_function_id(work_dir.name)
    display = f"YOLOX {work_dir.name}"[:120]
    spec_body = build_cvat_detector_spec(classes)
    yaml_text = render_function_gpu_yaml(
        nuclio_name=nid,
        display_name=display,
        spec_json=spec_body,
        image_tag=docker_image_tag(work_dir.name),
    )
    (bundle / "function-gpu.yaml").write_text(yaml_text, encoding="utf-8")

    meta = {
        "work_dir": work_dir.name,
        "nuclio_function": nid,
        "docker_image": docker_image_tag(work_dir.name),
        "classes": classes,
        "checkpoint": ckpt_base,
    }
    (bundle / "bundle_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return bundle


def run_nuctl_deploy(bundle_dir: Path, timeout_sec: int = 3600) -> Tuple[int, str]:
    nuctl = shutil.which("nuctl")
    if not nuctl:
        raise RuntimeError("nuctl が PATH にありません。Nuclio CLI をインストールしてください。")
    cfg = bundle_dir / "function-gpu.yaml"
    if not cfg.is_file():
        raise RuntimeError(f"function-gpu.yaml がありません: {bundle_dir}")

    cmd = [
        nuctl,
        "deploy",
        "--project-name",
        "cvat",
        "--path",
        str(bundle_dir),
        "--file",
        str(cfg),
        "--platform",
        "local",
        "--env",
        "CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk",
        "--env",
        "CVAT_FUNCTIONS_REDIS_PORT=6666",
        "--platform-config",
        '{"attributes": {"network": "cvat_cvat"}}',
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    out = (proc.stdout or "") + ("\n--- stderr ---\n" if proc.stderr else "") + (proc.stderr or "")
    return proc.returncode, out


def list_deployable_work_dirs() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not WORK_DIR_PARENT.is_dir():
        return rows
    for p in sorted(WORK_DIR_PARENT.iterdir(), key=lambda x: x.name, reverse=True):
        if not p.is_dir() or not p.name.startswith("web_train_"):
            continue
        if not WORK_DIR_NAME_RE.match(p.name):
            continue
        if not (p / "last_checkpoint").is_file():
            continue
        try:
            find_training_config(p)
        except ValueError:
            continue
        try:
            ckpt, ck_base = _resolve_checkpoint_path(p)
        except ValueError:
            continue
        try:
            cfg = find_training_config(p)
            classes = infer_classes_from_config_py(cfg)
        except ValueError:
            classes = None
        row: Dict[str, Any] = {
            "id": p.name,
            "checkpoint": ck_base,
            "checkpoint_path": str(ckpt),
            "nuclio_function": nuclio_function_id(p.name),
        }
        row["summary"] = summarize_work_dir(p, classes=classes)
        rows.append(row)
    return rows


def deploy_work_dir(work_dir_basename: str, *, timeout_sec: int = 3600) -> Dict[str, Any]:
    wd = resolve_work_dir(work_dir_basename)
    bundle = stage_bundle(wd)
    meta = json.loads((bundle / "bundle_meta.json").read_text(encoding="utf-8"))
    try:
        code, log = run_nuctl_deploy(bundle, timeout_sec=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "exit_code": -1,
            "work_dir": wd.name,
            "bundle_dir": str(bundle),
            "nuclio_function": meta.get("nuclio_function"),
            "docker_image": meta.get("docker_image"),
            "classes": meta.get("classes"),
            "log": f"nuctl deploy が {timeout_sec}s でタイムアウトしました: {exc}",
        }
    ok = code == 0
    return {
        "ok": ok,
        "exit_code": code,
        "work_dir": wd.name,
        "bundle_dir": str(bundle),
        "nuclio_function": meta.get("nuclio_function"),
        "docker_image": meta.get("docker_image"),
        "classes": meta.get("classes"),
        "log": log[-120000:] if log else "",
    }

"""Run MMDetection training from a JSON spec (invoked as subprocess by the Web UI)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _mmdet_root() -> Path:
    return _repo_root() / "third_party" / "mmdetection"


def _ensure_paths() -> None:
    root = _repo_root()
    mmdet = _mmdet_root()
    os.environ.setdefault("LOCAL_RANK", "0")
    for p in (str(mmdet), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(str(mmdet))


def _apply_spec(cfg, spec: dict) -> None:
    from pathlib import Path as P

    data_root = P(spec["data_root"]).resolve()
    dr = str(data_root)
    train_ann = spec["train_ann"]
    val_ann = spec["val_ann"]
    train_img_prefix = spec["train_img_prefix"]
    val_img_prefix = spec["val_img_prefix"]
    max_epochs = int(spec["max_epochs"])
    lr = float(spec["lr"])
    batch_size = int(spec["batch_size"])
    work_dir = spec["work_dir"]

    cfg.train_dataloader.batch_size = batch_size
    cfg.val_dataloader.batch_size = batch_size

    cfg.train_dataloader.dataset.dataset.data_root = dr
    cfg.train_dataloader.dataset.dataset.ann_file = train_ann
    cfg.train_dataloader.dataset.dataset.data_prefix["img"] = train_img_prefix

    cfg.val_dataloader.dataset.data_root = dr
    cfg.val_dataloader.dataset.ann_file = val_ann
    cfg.val_dataloader.dataset.data_prefix["img"] = val_img_prefix

    if cfg.get("test_dataloader") is not None:
        cfg.test_dataloader.dataset.data_root = dr
        cfg.test_dataloader.dataset.ann_file = val_ann
        cfg.test_dataloader.dataset.data_prefix["img"] = val_img_prefix

    ann_val_abs = str(data_root / val_ann)
    cfg.val_evaluator.ann_file = ann_val_abs
    cfg.test_evaluator.ann_file = ann_val_abs

    cfg.train_cfg.max_epochs = max_epochs
    cfg.optim_wrapper.optimizer.lr = lr

    sched = cfg.param_scheduler
    if len(sched) < 3:
        raise ValueError("config param_scheduler が想定と異なります（要素が3未満）。")

    nel = int(cfg.get("num_last_epochs", 5))
    nel = max(1, min(nel, max_epochs - 1))
    cos_end = max_epochs - nel
    warm_end = min(5, max(1, cos_end - 1))
    cos_begin = 5 if cos_end > 6 else max(1, cos_end - 1)

    sched[0]["end"] = warm_end
    sched[1]["begin"] = cos_begin
    sched[1]["end"] = cos_end
    sched[1]["T_max"] = cos_end
    sched[1]["eta_min"] = lr * 0.05
    sched[2]["begin"] = cos_end
    sched[2]["end"] = max_epochs

    asl = cfg.get("auto_scale_lr")
    if asl is not None:
        asl["base_batch_size"] = batch_size

    cfg.work_dir = work_dir


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m webui.mmdet_train_worker <spec.json>", flush=True)
        return 2
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    _ensure_paths()

    from mmengine.config import Config
    from mmengine.registry import RUNNERS
    from mmengine.runner import Runner

    from mmdet.utils import setup_cache_size_limit_of_dynamo

    setup_cache_size_limit_of_dynamo()

    cfg = Config.fromfile(spec["config_path"])
    cfg.launcher = "none"
    _apply_spec(cfg, spec)

    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    runner.train()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise SystemExit(1) from exc

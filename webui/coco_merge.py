"""Merge multiple COCO instance JSON files (for validation evaluator)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from webui.coco_categories import read_coco_category_names


def merge_coco_instance_jsons(paths: Sequence[Path]) -> Dict[str, Any]:
    """Merge COCO JSON dicts; reassign image/annotation ids. Categories must match."""
    if not paths:
        raise ValueError("マージする COCO JSON がありません。")
    merged: Dict[str, Any] = {"images": [], "annotations": []}
    categories: Any = None
    next_img_id = 1
    next_ann_id = 1

    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if categories is None:
            categories = raw.get("categories")
        elif raw.get("categories") != categories:
            raise ValueError(
                f"カテゴリ定義が一致しません（マージ不可）: {path.name}"
            )
        id_map: Dict[int, int] = {}
        for img in raw.get("images") or []:
            old_id = int(img["id"])
            new_img = dict(img)
            new_img["id"] = next_img_id
            id_map[old_id] = next_img_id
            next_img_id += 1
            merged["images"].append(new_img)
        for ann in raw.get("annotations") or []:
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = id_map[int(ann["image_id"])]
            next_ann_id += 1
            merged["annotations"].append(new_ann)

    merged["categories"] = categories or []
    return merged


def filter_coco_by_class_names(
    coco: Dict[str, Any],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """Keep only categories / annotations for the given class names."""
    allowed = set(class_names)
    cats = [c for c in coco.get("categories") or [] if c.get("name") in allowed]
    cat_ids = {int(c["id"]) for c in cats}
    anns = [
        a
        for a in coco.get("annotations") or []
        if int(a.get("category_id", -1)) in cat_ids
    ]
    out = dict(coco)
    out["categories"] = cats
    out["annotations"] = anns
    return out


def write_merged_val_annotations(
    items: Sequence[Tuple[Path, str]],
    out_path: Path,
    *,
    class_names: Optional[Sequence[str]] = None,
) -> str:
    """items: (data_root, ann_relpath). Returns ann path relative to first data_root."""
    paths: List[Path] = []
    for root, ann_rel in items:
        p = root / ann_rel
        if not p.is_file():
            raise ValueError(f"アノテーションが見つかりません: {p}")
        paths.append(p)
    merged = merge_coco_instance_jsons(paths)
    if class_names is not None:
        merged = filter_coco_by_class_names(merged, class_names)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged), encoding="utf-8")
    return str(out_path.resolve())


def prepare_val_evaluator_ann_file(
    val_resolved: Sequence[Dict[str, str]],
    work_dir: Path,
    class_names: Sequence[str],
) -> str:
    """Write filtered (or merged) COCO JSON for val_evaluator.ann_file."""
    items = [(Path(s["data_root"]), s["ann_file"]) for s in val_resolved]
    all_names: Set[str] = set()
    for root, ann_rel in items:
        all_names.update(read_coco_category_names(root / ann_rel))
    use_subset = set(class_names) != all_names

    if len(items) == 1 and not use_subset:
        return str(items[0][0] / items[0][1])

    if len(items) == 1:
        raw = json.loads((items[0][0] / items[0][1]).read_text(encoding="utf-8"))
        filtered = filter_coco_by_class_names(raw, class_names)
        out = work_dir / "_val_eval_annotations.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(filtered), encoding="utf-8")
        return str(out.resolve())

    return write_merged_val_annotations(
        items,
        work_dir / "_merged_val_annotations.json",
        class_names=class_names if use_subset else None,
    )

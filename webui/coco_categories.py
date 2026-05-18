"""Read and merge COCO category names from annotation JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def read_coco_category_names(ann_path: Path) -> List[str]:
    """Return category names sorted by COCO category id."""
    raw = json.loads(ann_path.read_text(encoding="utf-8"))
    cats = raw.get("categories") or []
    cats_sorted = sorted(cats, key=lambda c: int(c.get("id", 0)))
    names: List[str] = []
    for c in cats_sorted:
        name = c.get("name")
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    return names


def _names_from_resolved_sources(
    resolved: Sequence[Dict[str, str]],
) -> Set[str]:
    names: Set[str] = set()
    for s in resolved:
        ann_path = Path(s["data_root"]) / s["ann_file"]
        for n in read_coco_category_names(ann_path):
            names.add(n)
    return names


def suggest_categories(
    train_resolved: Sequence[Dict[str, str]],
    val_resolved: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    """Union of category names from train / val COCO JSONs (stable order)."""
    train_set = _names_from_resolved_sources(train_resolved)
    val_set = _names_from_resolved_sources(val_resolved)
    ordered: List[str] = []
    seen: Set[str] = set()

    def _append_from(resolved: Sequence[Dict[str, str]]) -> None:
        for s in resolved:
            ann_path = Path(s["data_root"]) / s["ann_file"]
            for name in read_coco_category_names(ann_path):
                if name not in seen:
                    seen.add(name)
                    ordered.append(name)

    _append_from(train_resolved)
    _append_from(val_resolved)

    rows = [
        {
            "name": name,
            "in_train": name in train_set,
            "in_val": name in val_set,
        }
        for name in ordered
    ]
    return {
        "categories": rows,
        "names": ordered,
        "default_selected": list(ordered),
    }


def validate_selected_classes(
    selected: Optional[Sequence[str]],
    *,
    allowed_names: Sequence[str],
) -> List[str]:
    allowed = list(allowed_names)
    if not allowed:
        raise ValueError("選択したデータからカテゴリを読み取れませんでした。")
    if selected is None:
        return allowed
    sel = [str(c).strip() for c in selected if str(c).strip()]
    if not sel:
        raise ValueError("カテゴリを 1 つ以上選択してください。")
    allowed_set = set(allowed)
    extra = [c for c in sel if c not in allowed_set]
    if extra:
        raise ValueError(f"データに存在しないカテゴリです: {', '.join(extra)}")
    return sel

# MMPlatform からの MMDetection 学習

Web UI（`/train`）は `third_party/mmdetection` 配下の config を読み込み、画面で選んだデータとハイパーパラメータで上書きして学習します。

## MMDetection config に必要なこと

### 置き場所

- `third_party/mmdetection/configs/` 以下の `.py` ファイル
- 学習画面のドロップダウンに自動表示（`_base_` 配下の断片 config は除外）

### データセット構造（必須）

Web UI が **train / val のパスを上書き** できるのは、次のいずれかの形だけです。

| 役割 | 想定構造 |
|------|----------|
| train | `train_dataloader.dataset` が `CocoDataset`、または `MultiImageMixDataset` → 内側が `CocoDataset` / `ConcatDataset` |
| val | `val_dataloader.dataset` が `CocoDataset`（`ann_file`・`data_prefix` を持つ dict） |

内側の `CocoDataset` には **`ann_file` と `data_prefix.img`** が定義されていること。画面で選んだ `data/exports/...` の COCO JSON と画像プレフィックスに差し替わります。

**非対応の例**: `OpenImagesDataset`、画像フォルダのみの設定、`ann_file` のないカスタム Dataset など。

### クラス数・事前学習

- **`metainfo.classes`**（または同等の `num_classes`）は、COCO JSON のカテゴリと一致させる
- 画面で「事前学習 work_dir」を選ぶ場合は **`model.init_cfg`（`type='Pretrained'`）** が必要

### 評価（推奨）

- **`val_evaluator`**（と `test_evaluator`）に COCO 形式の `ann_file` 指定があること  
  → val が複数のときは `work_dir/_merged_val_annotations.json` にマージして設定

### 学習ループ（推奨）

- **`train_cfg.max_epochs`** … 画面の max_epochs で上書き
- **`optim_wrapper.optimizer.lr`** … 画面の lr で上書き
- **`train_dataloader.batch_size` / `val_dataloader.batch_size`** … 画面の batch_size で上書き

### YOLOX 向け scheduler（任意）

`param_scheduler` が **3 要素以上** のときだけ、YOLOX 用の epoch スケジュール調整を行います。それ以外の config では scheduler は config 記載のままです。

---

## 既定 config（そのまま使えるテンプレート）

**`configs/yolox/yolox_s_finetune.py`**

- CVAT COCO エクスポート（`data/exports/<task>/annotations/instances_*.json` + `images/.../`）向け
- `metainfo.classes`、COCO 事前学習 URL、`MultiImageMixDataset` + YOLOX 用 aug を含む
- **新規タスクはこのファイルをコピーして `num_classes` / `metainfo` だけ変える**のが最短

```python
# カスタム例: クラスを増やすだけ
metainfo = dict(classes=('person', 'helmet'))
num_classes = len(metainfo['classes'])
model = dict(
    bbox_head=dict(num_classes=num_classes),
    init_cfg=dict(type='Pretrained', checkpoint='...'),
)
```

config 内の `data_root` / `train_ann` 等は **プレースホルダ** でよい（学習開始時に UI 選択で置き換わる）。

---

## カスタム config チェックリスト

1. `configs/<your_model>/your_job.py` として配置
2. train / val が **CocoDataset 系** で `ann_file` を持つ
3. クラス数と COCO JSON の categories が一致
4. 必要なら `model.init_cfg` で事前学習チェックポイントを定義
5. `val_evaluator` で bbox mAP 評価できるようにする
6. 学習画面で config を選び、Train / Val データを別々に（複数可）指定して実行

---

## 画面・API（参考）

| 項目 | 内容 |
|------|------|
| work_dir 名 | 常に `web_train_` + 接尾辞（空欄で自動生成） |
| Train / Val | 別リスト・複数選択可。複数時は `ConcatDataset`、val 評価は JSON マージ |
| config 一覧 | `GET /api/train/configs` |
| 学習開始 | `POST /api/train/start`（`config_path`: `configs/yolox/yolox_s_finetune.py` 形式） |
| カテゴリ提案 | `POST /api/train/suggest-categories`（Train / Val 選択から COCO `categories` を読み取り） |
| カテゴリ絞り込み | `classes`: 学習するカテゴリ名のリスト（未指定時は検出された全カテゴリ） |

学習後の `work_dir` は Nuclio デプロイ画面（`web_train_*` のみ一覧）から利用できます。

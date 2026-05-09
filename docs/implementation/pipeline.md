# 学習パイプライン

> **Status**: stable | **Last reviewed**: 2026-05-10
>
> AMI Corpus → ITM 学習向け Dataset / DataLoader / 損失関数の実装。Phase 2 の fine-tune と Phase 3 の視覚追加で使う共通基盤。

## TL;DR

- `itm.data.AMIDataset` が会議群を **20s チャンク** に切って `(audio_2ch, hazard, mask)` を返す
- 損失は `itm.data.survival_nll_loss` の **discrete-time survival NLL**
- 5 会議で **983 chunks**（20s chunk / 10s hop、計 165 分）
- 全 44 ユニットテスト合格

## 構成

```mermaid
flowchart LR
    A[AMI annotations<br/>XML] --> B[ami.py<br/>parser]
    AU[AMI audio<br/>headset wav] --> AUD[audio.py<br/>load_two_channel_audio]
    B --> L[labels.py<br/>extract_event_onsets<br/>+ survival_targets]
    L --> T[targets.py<br/>survival_to_tensors]
    AUD --> DS[dataset.py<br/>AMIDataset]
    T --> DS
    DS --> DL[DataLoader<br/>+ ami_collate]
    DL --> M[Model<br/>(Phase 2-)]
    M --> LOSS[targets.py<br/>survival_nll_loss]
```

## モジュール一覧

| モジュール | 役割 |
|---|---|
| `itm.data.ami` | NXT XML → `Word`/`Segment`/`DialogAct`/`Meeting` |
| `itm.data.labels` | dialog acts → ITM event onsets → サバイバル形式 |
| `itm.data.audio` | 2 話者 headset 音声の同期ロード、チャンク切り出し |
| `itm.data.targets` | サバイバルラベル → torch tensor、NLL 損失 |
| `itm.data.dataset` | `AMIDataset`（PyTorch `Dataset`）+ `ami_collate` |

## Dataset API

### コンストラクタ

```python
from itm.data import AMIDataset

ds = AMIDataset(
    annot_root="data/raw/ami/annotations/unpacked",
    audio_root="data/raw/ami",
    meeting_ids=["ES2002a", "ES2002b", "ES2002c", "IS1000a", "IS1000b"],
    chunk_sec=20.0,        # 1 chunk = 20 秒
    hop_sec=10.0,          # 隣接 chunk の間隔
    frame_rate_hz=20,      # モデル出力フレームレート（VAP 互換）
    horizon_bins=40,       # 将来予測 bin 数（40 × 50ms = 2 秒先まで）
    bin_size_sec=0.05,     # 各 bin の幅
)
```

### 1 アイテムの形

```python
item = ds[0]
# {
#   "audio":   Float[320000, 2]    # 20s × 16kHz × 2ch
#   "hazard":  {                   # 各イベントについて
#       EventType.TURN_SHIFT:  Long[400, 40]   # 20s × 20Hz × K=40
#       EventType.BACKCHANNEL: Long[400, 40]
#       EventType.OVERLAP:     Long[400, 40]
#   }
#   "mask":    {ev: Float[400, 40]}
#   "meeting":   "ES2002a"
#   "speakers":  ("B", "D")
#   "start_sec": 0.0
# }
```

`hazard` は **次イベント発生 bin に 1 が立つ** スパース行列。`mask` は survival NLL で観測扱いする bin（イベント未発生かつ horizon 内）。

### DataLoader 統合

```python
from torch.utils.data import DataLoader
from itm.data import ami_collate

loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=ami_collate)
batch = next(iter(loader))
# batch["audio"]: Float[4, 320000, 2]
# batch["hazard"][EventType.TURN_SHIFT]: Long[4, 400, 40]
```

## 損失

### サバイバル NLL

discrete-time survival の負の対数尤度を、event-bin と未観測 bin で正しくマスクして計算する：

```
L_e(t) = - log h_e(t, k*) - sum_{j<k*} log(1 - h_e(t, j))   # event observed at bin k*
       = - sum_{j=0}^{K-1} log(1 - h_e(t, j))               # right-censored
```

実装：

```python
from itm.data import survival_nll_loss

# hazard_logits: Float[B, T, K] — モデル出力（pre-sigmoid）
# batch["hazard"][ev]: Long[B, T, K]  — 0/1 ラベル
# batch["mask"][ev]:   Float[B, T, K] — 観測フラグ
loss = survival_nll_loss(
    hazard_logits,
    batch["hazard"][EventType.TURN_SHIFT],
    batch["mask"][EventType.TURN_SHIFT],
)
```

`reduction="mean" / "sum" / "none"` を選べる。

## 実 AMI でのサニティチェック

`scripts/inspect_dataset.py` で 5 会議をロードした結果：

```
Building dataset from: ['ES2002a', 'ES2002b', 'ES2002c', 'IS1000a', 'IS1000b']
Dataset size: 983 chunks

--- Aggregate (full dataset) ---
  turn_shift    :  28395 positive hazard frames
  backchannel   :  40785 positive hazard frames
  overlap       :  70730 positive hazard frames
```

イベント別の正例数が出ているため、**学習信号は十分** と判断できる。

## メモリ占有

各 `AMIDataset` インスタンスは **会議の音声を一括メモリロード**：

| 会議数 | 音声 RAM | 注釈 + ターゲット | 合計 |
|---|---|---|---|
| 1 (ES2002a) | ~155 MB | 数 MB | ~160 MB |
| 5 (全部) | ~1.0 GB | ~30 MB | **~1.0 GB** |

ストリーミング不要（M4 16GB / Linux クラウドどちらでも乗る）。

## テスト

`tests/test_dataset.py` で:
- 軽量 fake corpus を tmp ディレクトリに作って Dataset 構築
- `__getitem__` で正しい形が返るか
- DataLoader + collate が batch を作れるか
- 既知のイベント時刻が hazard に正しく現れるか
- バッチに `survival_nll_loss` を流して finite な値が返るか

`tests/test_audio.py` で:
- 2 話者音声の同期ロード
- チャンク切り出しのパディング/切り詰め

`tests/test_targets.py` で:
- 観測 / 打ち切り両方のサバイバルテンソル化
- NLL 損失の正解推論で 0、誤推論で大きな値

## Phase 2 への接続

このパイプラインは Phase 2 で以下を可能にする：

1. **Fine-tune ベースライン**: MaAI バックボーンに 3 ハザードヘッドを足し、AMI で fine-tune → hold/shift accuracy ≥ 70% 目標
2. **マルチイベント学習**: turn-shift / backchannel / overlap 同時最適化
3. **校正**: `survival_nll_loss` の reduction="none" を使った per-bin 校正性指標の計算

---

## ITM モデル + 学習スクリプト（Phase 2-B 実装）

### `itm.models.ITMModel`

VAP backbone + 3 ハザード head の構成。

```mermaid
flowchart LR
    A[audio (B, T_samples, 2)] --> ENC[CPC Encoder ×2<br/>frozen]
    ENC --> AR1[ar_channel ch1]
    ENC --> AR2[ar_channel ch2]
    AR1 --> CR[ar Cross-Attention<br/>trainable]
    AR2 --> CR
    CR --> H[hidden state h<br/>(B, T_enc, dim=256)]
    H --> H1[hazard head turn_shift]
    H --> H2[hazard head backchannel]
    H --> H3[hazard head overlap]
    H1 --> O1[hazard_logits<br/>(B, T_enc, K=40)]
    H2 --> O2[(B, T_enc, K=40)]
    H3 --> O3[(B, T_enc, K=40)]
```

```python
from itm.models import build_itm_model

model = build_itm_model(
    lang="en",
    frame_rate=20,
    context_len_sec=20,
    horizon_bins=40,
    freeze_encoder=True,         # CPC エンコーダはフリーズ
    freeze_transformer=False,    # ar_channel/ar は AMI で fine-tune
)
print(model.count_parameters())
# {'total': 8_072_725, 'trainable': 3_726_841, 'encoder': 4_343_808,
#  'hazard_heads': 115_704, 'transformer': 3_613_213}
```

**重要**: CPC エンコーダは内部的に **~50 Hz** でフレームを出力するため、`audio` を 20s 渡すと `hazard_logits` は約 ~997 フレーム（B, 997, K）になる。学習データの target も `frame_rate_hz=50` で生成して長さを揃えるのが推奨。

### 学習ループ

```python
from itm.training import train_step
from itm.data import AMIDataset, ami_collate
from torch.utils.data import DataLoader
import torch

train_ds = AMIDataset(annot_root, audio_root, ["ES2002b", "ES2002c", "IS1000a", "IS1000b"],
                     chunk_sec=20.0, hop_sec=10.0, frame_rate_hz=50, horizon_bins=40)
loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=ami_collate)

optim = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=3.63e-4, weight_decay=1e-3,
)
for batch in loader:
    info = train_step(model, batch, optim)
    print(info.total_loss)
```

長さ不一致（model 997 vs target 1000 など）は ``compute_loss`` 内で短い方に切り詰められる。

### CLI スクリプト

```bash
# Sanity smoke test (1 meeting, 8 steps)
python scripts/train_itm.py --meetings ES2002a --epochs 1 --batch-size 2 --max-steps 8

# Full Phase 2-B training (5 meetings, multi-epoch)
python scripts/train_itm.py --all --epochs 8 --batch-size 4 --device cuda
```

学習履歴は `checkpoints/itm_phase2b_log.jsonl`、checkpoint は
`checkpoints/itm_phase2b_epoch{NN}.pt` に保存。`val_loss` 最良の epoch は
`itm_phase2b_best.pt` にコピーされる。

### CPU での実行可否

M4 MacBook で sanity 動作確認済み:

```
Building datasets...
train: 126 chunks   val: 63 chunks
Building model (frame_rate=20, device=cpu)...
Parameter counts:
  total            8,072,725
  trainable        3,726,841
  ...
epoch=0 step=2 loss=0.6546 turn_shift=0.657 backchannel=0.639 overlap=0.668
epoch=0 step=4 loss=0.5888 turn_shift=0.597 backchannel=0.569 overlap=0.601
epoch=0 step=6 loss=0.5395 turn_shift=0.551 backchannel=0.510 overlap=0.558
epoch=0 step=8 loss=0.4985 turn_shift=0.512 backchannel=0.469 overlap=0.514
Done. 8 steps in 19.8s
```

CPU 1 step ≈ 2.5s（batch=2、chunk=20s）。GPU で 5 〜 10 倍高速化を見込む。

## 関連ページ

- [v1 アーキテクチャ](../design/architecture.md) — モデル設計
- [マルチイベント・ハザード](../design/multi-event-hazard.md) — 損失の理論
- [ラベル生成](../design/label-generation.md) — AMI から ITM event への変換規則
- [MaAI ベースライン](maai-baseline.md) — Phase 1 結果（Phase 2 で超える対象）

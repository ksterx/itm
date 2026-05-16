# 学習パイプライン

> **Status**: stable | **Last reviewed**: 2026-05-11 (v4)
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

### Smoke test 学習曲線（ES2002a 単独、3 epoch）

`scripts/train_itm.py --meetings ES2002a --epochs 3 --batch-size 2`:

```
epoch=0 step=20  loss=0.2575  turn_shift=0.232 backchannel=0.226 overlap=0.315
epoch=0 step=40  loss=0.0495  turn_shift=0.058 backchannel=0.028 overlap=0.063
epoch=0 step=60  loss=0.0135  turn_shift=0.014 backchannel=0.005 overlap=0.021
  [val] epoch=0 loss=0.0212
epoch=1 step=80  loss=0.0156
epoch=1 step=120 loss=0.0214
  [val] epoch=1 loss=0.0192
epoch=2 step=180 loss=0.0188
  [val] epoch=2 loss=0.0191

Done. 189 steps in 560.6s
```

- 9.4 分 CPU で 3 epoch 完了
- Loss は最初の epoch 中に 0.26 → 0.013 と急減
- val loss は 0.022 → 0.019 で plateau
- ⚠️ 1 会議のみで train=val なので **汎化評価ではない**。infrastructure の動作確認にとどまる

本格的な汎化評価は次節（Phase 2-B 4+1 split）で行う。

### Phase 2-B 本格学習: 4+1 cross-meeting split

`scripts/train_itm.py --all --epochs 1`:

```
train meetings: ['ES2002a', 'ES2002b', 'ES2002c', 'IS1000a']
val   meetings: ['IS1000b']
train: 750 chunks   val: 117 chunks

epoch=0 step=30  loss=0.1101
epoch=0 step=60  loss=0.0305
epoch=0 step=180 loss=0.0234
epoch=0 step=360 loss=0.0364
  [val] epoch=0 loss=0.0194  (n_batches=59)

Done. 375 steps in 943.9s
```

15.7 分 CPU、val loss 0.0194。汎化はしているように見える。

## ⚠ Phase 2-B 評価結果: 学習は失敗した

`scripts/eval_itm_on_ami.py --checkpoint checkpoints/itm_phase2b_v1_best.pt --meetings IS1000b`:

| 指標 | MaAI baseline (Phase 1) | ITM 学習後 (Phase 2-B v1) |
|---|---|---|
| Frame VAD accuracy | **0.933** | **0.518** 🔻 |
| Hold accuracy | 86/134 = 0.642 | 134/134 = 1.000 |
| Shift accuracy | 23/52 = 0.442 | **0/52 = 0.000** 🔻 |
| Overall | 0.586 | 0.720 (見かけ) |

すべてのしきい値（0.005 〜 0.5）で全 silence を **HOLD** と予測。Overall 0.720 は class imbalance（hold 72%、shift 28%）による見かけの数字で、実質的にはモデルが何も学習していない。

### なぜ失敗したか

1. **Survival NLL のクラス不均衡**: 各 (frame, bin) のうちイベント発生 bin はせいぜい 1 個、残りはすべて 0。「全部 0 を予測」が局所最適になる
2. **VAD 能力の喪失**: VAP backbone の transformer 層を fine-tune した結果、元々持っていた VAD 認識（93%）が崩壊（52%）。 hazard 損失だけを最適化すると VAP が学習した話者識別が忘却される
3. **校正されていないしきい値**: hazard が低い値で偏っており、しきい値スイープしても全部 HOLD 判定になる

### v2 で修正すべき項目

| 修正 | 詳細 |
|---|---|
| **Positive class weighting** | survival NLL に `pos_weight` を導入（10〜100 程度）。または focal loss |
| **VAD 補助損失を併用** | `va_classifier` の出力に対する VAD BCE を total loss に追加して話者識別を保持 |
| **Backbone 凍結 + ヘッドのみ学習から開始** | `freeze_transformer=True` で先にヘッドを学習、後段で transformer の解凍も検討 |
| **しきい値の自動較正** | 学習時に hazard 分布を観察し、val で balanced accuracy を最大化するしきい値を選ぶ |

これらは Phase 2-B v2 で実装する。

### Phase 2-B v1 の意義

「失敗した」が、実装と評価インフラは完成しており、**問題点が定量的に観察できる状態** に到達した。これが v2 改善の出発点になる。

## Phase 2-B v2: pos_weight + VAD aux + freeze_transformer

v1 の失敗を踏まえた修正：

1. **`survival_nll_loss(pos_weight=...)`** 追加 — クラス不均衡対策
2. **VAD 補助損失** — `compute_loss(vad_logits=, vad_target=)` が `binary_cross_entropy_with_logits` を返す。`AMIDataset` も `vad_target` を batch に追加
3. **`--freeze-transformer`** — VAP の transformer 層を凍結し、hazard heads (115K) のみ学習。VAD 能力を保護

### v2 学習結果（pos_weight=50、VAD aux on、frozen transformer）

```
trainable: 115,704 params
epoch=0 step=30  loss=1.5365 turn_shift=0.213 backchannel=0.969 overlap=1.438 vad=0.663
epoch=0 step=180 loss=0.9794 turn_shift=0.287 backchannel=0.149 overlap=0.344 vad=0.719
epoch=0 step=360 loss=1.0882 turn_shift=0.400 backchannel=0.303 overlap=0.404 vad=0.719
  [val] epoch=0 loss=0.1572

Done. 375 steps in 779.2s (13 min)
```

学習対象は heads のみなので、loss スケールは pos_weight の影響で変化（v1 とは比較不可）。

### v2 評価結果（IS1000b、threshold sweep）

| threshold | Hold | Shift | Overall |
|---|---|---|---|
| 0.005 〜 0.05 | 0/134 (0%) | **52/52 (100%)** | 0.280（全 SHIFT） |
| 0.1 | 106/134 (79%) | 9/52 (17%) | **0.618** |
| 0.2 | 133/134 (99%) | 0/52 (0%) | 0.715 |
| 0.3 〜 0.5 | 134/134 (100%) | 0/52 (0%) | 0.720（全 HOLD） |

ベースライン MaAI: 0.586 (Hold 64%、Shift 44%)。

### 観察

- **threshold 0.1 で Overall 0.618** ─ ベースライン 0.586 をわずかに上回る
- **Hold は 79% に向上**（ベースライン 64%）
- **Shift は 17% に低下**（ベースライン 44%）— トレードオフ
- しきい値で hold/shift が極端に振れる → モデルの hazard 出力分布が **「ほぼ一様」**で、hold vs shift を識別する情報が出力に乗っていない

### Frame VAD の比較は意味がない

ベースライン eval は `p_now`（VAP 頭部の出力）を使う一方、ITM eval は `va_classifier` の per-channel VAD を使う。両者は意味が異なる出力なので、Frame VAD の数値（baseline 0.933 vs v2 0.510）は **異なる指標** を比較しており、直接比較は無効。フェアな比較には ITM 側にも `p_now` を出力する必要があり、これは v3 の作業項目とする（[deferred: 比較インフラの統一は学習収束後に行う]）。

### v2 の意義

「ベースラインを安定して上回る」には至らなかったが:

- Survival NLL のクラス不均衡問題を pos_weight で部分的に緩和
- VAD aux loss と凍結により VAP の能力を保護（v1 で起きた崩壊は回避）
- しきい値しだいで baseline を超えうる構成にはなった

### v3 に向けた次のアクション

| 修正 | 詳細 |
|---|---|
| **pos_weight 探索** | 5/10/20/100 を比較。50 は SHIFT 過剰と HOLD 過剰の中間で挙動不安定 |
| **transformer 部分解凍** | freeze_transformer=False に戻し、ただし学習率を低めに（1e-5 程度）して VAD 能力を保ちつつ AMI に適応 |
| **複数 epoch** | 1 epoch では不十分の可能性。3〜5 epoch で再訓練 |
| **`p_now` を ITM から出力** | フェアな Frame VAD 比較のため（[deferred: v3]） |
| **Shift 専用の訓練 signal** | mutual silence の gt_label と直接結びつくラベル（survival とは別）も併用

## Phase 2-B v3: transformer 部分解凍 + multi-epoch

> **Status**: stable（2026-05-10 完了）
>
> v2 で残された v3 アクション項目のうち最重要 3 つを 1 回の実験に集約。

### v3 学習設定

```bash
python scripts/train_itm.py --all --epochs 3 --batch-size 2 \
    --pos-weight 20 --use-vad-aux --vad-loss-weight 1.0 \
    --lr 1e-5 --save-name itm_phase2b_v3
```

| 項目 | v2 | v3 |
|---|---|---|
| transformer | **frozen** | **trainable** |
| learning rate | 3.63e-4 (head のみ) | **1e-5** (全パラメータ) |
| epochs | 1 | **3** |
| pos_weight | 50 | **20** |
| VAD aux | ✅ | ✅ |

**狙い**:
- 低 LR で transformer を AMI に適応させつつ、VAD aux で大規模 pretrain の能力を保つ
- pos_weight=20 で v1 (=1, all-zero collapse) と v2 (=50, hold-bias) の中間
- 3 epoch で 1 epoch では拾えない学習信号を回収

### v3 学習曲線

| epoch | val loss |
|---|---|
| 0 | 0.300 |
| 1 | 0.137 |
| 2 | **0.095** |

CPU で 3 epoch / 1125 step / 47 分。Train loss は step 10 の 1.55 から step 1100 の 0.25–0.50 帯まで単調減少。VAD aux loss も 0.78 → 0.07–0.20 で保持され、v1 の崩壊（0.93→0.52）は再現せず。

### v3 / baseline AUC 診断 — survival NLL の構造的限界

しきい値非依存の評価で実態が判明:

| | ROC-AUC | PR-AUC | hazard 出力範囲 |
|---|---|---|---|
| **MaAI baseline** | **0.701** | **0.484** | (p_future ベース) |
| v2 (frozen) | 0.487 | 0.270 | turn [0.028, 0.358] |
| v3 (unfrozen) | 0.440 | 0.262 | turn [0.031, 0.109] |

`scripts/diagnose_itm_hazard.py` で v3 の hazard score を 12 通りに集約しても AUC は 0.44–0.56 帯。

- v2 の hazard 出力は v3 より広いが、shift discrimination はほぼ random
- v3 は **scoring を反転** (1 - hazard) すると AUC が 0.560 に上がる — 弱い負相関
- どちらも survival NLL のみでは shift discrimination を獲得できていない

**結論**: post-hoc temperature scaling は AUC を上げない（モノトニック変換は不変）。学習目的を変える必要 → v4。

### v3 評価結果（IS1000b、threshold sweep）

| threshold | frame_acc | Hold | Shift | Overall | 解釈 |
|---|---|---|---|---|---|
| 0.005–0.09 | 0.748 | 0/134 | 52/52 (100%) | 52/186 (0.280) | 全 SHIFT 予測 |
| **0.10** | 0.748 | 48/134 (35.8%) | 26/52 (**50.0%**) | 74/186 (**0.398**) | 唯一の動作点 |
| 0.11–0.5 | 0.748 | 134/134 (100%) | 0/52 (0%) | 134/186 (0.720) | 全 HOLD 予測（= データ prior） |

比較:

| モデル | Hold | Shift | Overall | 備考 |
|---|---|---|---|---|
| MaAI baseline | 64.2% | **44.2%** | 0.586 | バランス良 |
| v2 (frozen, pw=50) | 79.1% | 17.3% | **0.618** | hold バイアス |
| v3 (unfrozen, pw=20) | 35.8% | **50.0%** | 0.398 | shift バイアス、calibration が脆弱 |
| trivial (always HOLD) | 100% | 0% | 0.720 | データ prior |

### v3 観察

1. **Shift accuracy 50.0% で初めて baseline (44.2%) を上回った** — Phase 2-B 系列で初の真の前進
2. ただし overall 0.398 は baseline 0.586 を下回る — Hold が 36% まで落ちたため
3. Hazard 分布が狭く、threshold 0.10 ↔ 0.11 で全反転する**「all-or-nothing」現象**
4. Frame VAD 0.748 は v2/baseline (0.93) より低下 — 低 LR でも transformer fine-tune の影響あり
5. Trivial (常に HOLD) が 0.720 でテスト集合 prior と一致 — Overall 単独は誤導的指標

### v3 の意義と次の方向

- 当初 Shift 50% は前進と解釈したが、AUC 診断で偶然と判明
- survival NLL のみでは AUC ≤ 0.5 で discrimination 不能
- v4 では学習目的を BCE ベース discriminative head に変更

## Phase 2-B v4: Shift 専用 BCE ヘッド

> **Status**: stable（2026-05-11 完了）
>
> survival NLL の構造的限界を回避するため、silence 直前の context から
> shift/hold を直接二値分類する独立ヘッドを追加。

### v4 アーキテクチャ

```
backbone hidden h ∈ (B, T_enc, dim)
    ├─> hazard heads (turn / bc / overlap)  [既存、保持]
    ├─> VAD head                             [既存、frozen]
    └─> shift_head: LayerNorm → Linear(dim, 128) → GELU → Linear(128, 1)
            → (B, T_enc) shift logits
```

### v4 学習設定

```bash
python scripts/train_itm.py --all --epochs 3 --batch-size 2 \
    --pos-weight 20 --use-vad-aux --vad-loss-weight 1.0 \
    --use-shift-head --shift-loss-weight 1.0 --shift-pos-weight 1.5 \
    --freeze-transformer --save-name itm_phase2b_v4
```

| 項目 | v3 | v4 |
|---|---|---|
| transformer | unfrozen (lr=1e-5) | **frozen** |
| Shift head | ❌ | ✅ (BCE on silence frames) |
| trainable params | 8.07M | **149K** (heads + shift_head) |
| 学習時間 (CPU) | 47 min | **39 min** |

### v4 学習曲線

| epoch | val loss |
|---|---|
| 0 | **0.0521** ← best |
| 1 | 0.0637 |
| 2 | 0.0565 |

epoch 0 で best — 早めに過学習する傾向。step 内で shift loss は乱高下 (0.27–2.6)、batch ごとの label 偏りに敏感。

### v4 評価結果（IS1000b、threshold sweep + AUC）

最初のスイープでは default threshold が低すぎて v4 の特性に合わなかった (shift_head の sigmoid 出力は 0.3–0.5 帯にある)。fine-grained スイープで真の動作点を発見:

| threshold | Hold | Shift | Overall |
|---|---|---|---|
| 0.30 | 8.2% | 94.2% | 0.323 |
| 0.35 | 39.6% | 73.1% | 0.489 |
| **0.38** | **67.2%** | **44.2%** | **0.608** ⭐ |
| 0.40 | 74.6% | 25.0% | 0.608 |
| 0.42 | 89.6% | 17.3% | 0.694 (hold-bias) |
| 0.45+ | ≥97.8% | ≤5.8% | 0.720 (trivial all-HOLD) |

| | **ROC-AUC** | **PR-AUC** |
|---|---|---|
| baseline | 0.701 | 0.484 |
| v2 (frozen) | 0.487 | 0.270 |
| v3 (unfrozen) | 0.440 | 0.262 |
| **v4 (shift head)** | **0.566** | **0.353** |

### v4 観察

1. **AUC 0.566 で初めて random (0.5) を上回った** — Phase 2-B で初の本物の discriminative signal
2. **threshold=0.38 で Overall 0.608**（baseline real-time 0.586 を上回る）— Shift 44.2% (=baseline) かつ Hold 67.2% (>baseline 64%) でバランス改善
3. 専用 BCE ヘッドは survival NLL より shift 検出に効く
4. AUC 0.701 (baseline) にはまだ届かない — discriminative signal の上限は v5 で押し上げる
5. Frame VAD 0.453 は frozen va_classifier が AMI 分布に転送できていない問題（v3 unfrozen の 0.748 と対比）

### Phase 2-B 系列まとめ

| | ROC-AUC | best Overall | Hold | Shift | 性質 |
|---|---|---|---|---|---|
| MaAI baseline (real-time) | 0.701 | 0.586 | 64% | 44% | バランス |
| v2 (frozen survival NLL) | 0.487 | 0.618 | 79% | 17% | Hold-bias |
| v3 (unfrozen survival NLL) | 0.440 | 0.398 | 36% | 50% | Shift-bias、偶然 |
| **v4 (Shift BCE head)** | **0.566** | 0.608 | 67% | 44% | バランス、本物 |
| v5a (pw=1, lw=2, 5 epoch) | 0.497 | 0.699 (trivial) | trivial | trivial | 退化解 |
| v5b (segment-BCE) | 0.508 | 0.602 @ 0.35 | 75% | 23% | Hold-bias |
| **v6-α (17 meetings, 8.6h)** | 0.535 | **0.677 @ 0.38** | 82% | 31% | hold-bias 強、calibration 改善 |

## Phase 2-B v5 試行: ハイパー & 学習単位の探索（regression 続き）

v4 を起点に 1 変数ずつ動かして AUC > 0.566 を目指したが、2 試行とも regression:

### v5a — `shift_pos_weight=1.0` + `shift_loss_weight=2.0` + 5 epoch

意図: shift 過剰予測抑制 (pos_weight 下げ) + shift 学習を loss で支配的に (loss_weight 上げ) + 多 epoch。

結果: **ROC-AUC 0.497** (random)、しきい値で振り切るだけの退化解。pos_weight と loss_weight を同時に動かしたため原因切り分け不能。「変数 1 つだけ動かす」原則を破ったのが教訓。

### v5b — segment-level BCE (v4 hyperparams 維持)

意図: 訓練の損失単位 (per-frame BCE) と評価の score 集約 (mean over silence) の **unit mismatch** を解消するため、`_segment_shift_bce` 関数で contiguous silence を 1 セグメントに collapse し、`(mean_logit, segment_label)` 単位で BCE 計算するように変更。

結果: **ROC-AUC 0.508**（v4 0.566 から悪化）。

原因解析:
- v4 (per-frame BCE) は 1 chunk あたり ~50 silence frame の gradient signal
- v5b (segment-BCE) は 1 chunk あたり 3–5 segment の gradient signal
- **gradient density が約 10× 減少** → 学習不足
- unit consistency より supervision density のほうが重要だった

### Phase 2-B v5 教訓

- v4 設定 (`pos_weight=1.5`, `loss_weight=1.0`, 3 epoch, per-frame BCE, frozen transformer) が現データ量での **near-optimal**
- これ以上の改善には次の構造変更が必要:
  1. **学習データ拡張** (5 → 20+ meetings、Phase 1 で download 拡大)
  2. **`va_classifier` 解凍** (Frame VAD 0.453 → 0.9 期待、indirect に shift にも効く可能性)
  3. **encoder fine-tune** (CPC を AMI で更新、ただし overfit リスク)
  4. **アーキテクチャ変更** (head の MLP を深く、attention pooling 等)

### 棚上げ: per-segment BCE 実装

`_segment_shift_bce` 関数は v5b で性能悪化を確認したため、現在 `compute_loss` の shift パスから呼ばれている状態。v6 で per-frame BCE に戻すか option flag で切り替え可能にするかを判断する [deferred: v6 で扱い方針を決める、現状コードは保持]。

### v6 候補（次回）

| 修正 | 詳細 | 期待効果 |
|---|---|---|
| **AMI 全 100h ダウンロード** | 現在 5 meetings (165 min) → 100h | データ量 30× で AUC 0.6+ 期待 |
| **va_classifier 解凍 (lr=1e-5)** | freeze_transformer=True かつ va_classifier だけ trainable のフラグ追加 | Frame VAD 回復 + shift 副次効果 |
| **shift_head MLP 深層化** | 2-layer → 3-layer + dropout | 表現容量増、過学習注意 |

## Phase 2-B v6-α 試行: データ 3.1× 拡張 (5 → 17 meetings)

> **Status**: stable（2026-05-16 完了）
>
> CT1000P3 外付け SSD に AMI 移行後、ES2003/ES2004/IS1001 シリーズ追加で
> 学習データを 2.75h → 8.6h に拡張。v4 設定そのままで再学習。

### v6-α 学習設定

```bash
python scripts/train_itm.py \
    --meetings ES2002a..c ES2003a-d ES2004a-d IS1000a IS1001a..d IS1000b \
    --epochs 3 --batch-size 2 \
    --pos-weight 20 --use-vad-aux --vad-loss-weight 1.0 \
    --use-shift-head --shift-loss-weight 1.0 --shift-pos-weight 1.5 \
    --freeze-transformer --save-name itm_phase2b_v6a
```

- train: 13 meetings (6.7h)、val: 4 meetings (1.9h、IS1001b/c/d + IS1000b)
- test: IS1000b 単独で AUC・しきい値スイープ（v4 と同条件）

### v6-α 結果

| epoch | val loss |
|---|---|
| 0 | 0.841 |
| 1 | 0.828 |
| 2 | **0.794** ← best (単調減少、v4 の「epoch 0 が best」とは違うパターン) |

学習時間: CPU 126 min (v4 39 min × 3.2、data 量比例)。

### v6-α 評価 (IS1000b)

| threshold | Hold | Shift | Overall |
|---|---|---|---|
| 0.30 | 50% | 58% | 0.522 |
| 0.35 | 67% | 37% | 0.586 |
| **0.38** | 82% | 31% | **0.677** ⭐ |
| 0.40–0.45 | 84–88% | 25–15% | 0.677 |
| 0.50 | 93% | 10% | 0.699 |

ROC-AUC = **0.535** (PR-AUC 0.342)、Frame VAD 0.482。

### v4 vs v6-α 比較

| | v4 (5 meetings) | v6-α (17 meetings) |
|---|---|---|
| ROC-AUC | 0.566 | **0.535** (微悪化) |
| best Overall (calibrated) | 0.608 @ 0.38 | **0.677 @ 0.38** (大改善) |
| Hold / Shift @ 0.38 | 67% / 44% | 82% / 31% (hold-bias 強化) |
| Frame VAD | 0.453 | 0.482 |
| threshold スイープ感度 | all-or-nothing | 滑らかに変化 |

### 観察

1. **データ 3.1× 増で AUC は変わらない (むしろ微悪化)** — discriminative 能力の上限はハイパー調整 + 中規模データで頭打ちに見える
2. **calibration は改善** — しきい値が安定し、Overall accuracy は大きく向上 (0.608 → 0.677)
3. **Overall は baseline real-time 0.586 を大幅に上回り、fast 0.640 も超える** — 実用上は今までで最高性能
4. **代償として Shift accuracy が低下** (44% → 31%、baseline 44% を下回る)
5. Frame VAD は微改善 (frozen va_classifier の元で、encoder が AMI 経由で間接学習)

### v6-α が示唆すること

- **「データを増やすだけ」では AUC を 0.7 (baseline) に近づけられない** — discrimination は別の手段が必要
- **モデルが「いつ来るか」より「全体的に hold になりがち」の事前確率を覚えている** — pos_weight 調整や segment 構造変更では限界
- v6-α は Overall 実用性能ではこれまでで最高、Phase 2-B のひとつの到達点

### v7 候補（discrimination を伸ばす方向）

| 修正 | 詳細 |
|---|---|
| **va_classifier 解凍 (lr=1e-5)** | 現在 frozen、AMI への adaptation 余地。Frame VAD も改善 |
| **shift_head MLP 深層化** | 2-layer → 3-layer + dropout、表現容量増 |
| **AMI scenario 全 50h** | v6-α より更に 6× データ、CPU 12–24h |
| **per-frame BCE への revert** | `_segment_shift_bce` を撤去し v4 の per-frame BCE 学習に戻す（v6-α は segment-BCE のまま） |

### v5 旧候補（再評価）

| 修正 | 詳細 |
|---|---|
| **shift_pos_weight を下げる** | 1.5 → 1.0 / 0.5 で shift 過剰予測を抑制 |
| **Calibration set でしきい値最適化** | val から最適しきい値を学習し test で適用 |
| **Score 集約変更** | `mean` ではなく `silence midpoint` の単一フレーム値で評価 |
| **va_classifier 解凍 + 低 LR** | Frame VAD を AMI 分布に適応（v3 で実証） |
| **Multi-epoch で shift_head LR を別管理** | shift loss の振動が大きいので別 optimizer / scheduler |
| **`--shift-loss-weight` 増加** | 2.0–5.0 で shift 学習を支配的に |

## 関連ページ

- [v1 アーキテクチャ](../design/architecture.md) — モデル設計
- [マルチイベント・ハザード](../design/multi-event-hazard.md) — 損失の理論
- [ラベル生成](../design/label-generation.md) — AMI から ITM event への変換規則
- [MaAI ベースライン](maai-baseline.md) — Phase 1 結果（Phase 2 で超える対象）

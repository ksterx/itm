# MaAI ベースライン

> **Status**: stable | **Last reviewed**: 2026-05-09
>
> MaAI を ITM のベースラインとして使う方法。

## なぜ MaAI

[既存モデル](../research/existing-models.md) で詳述した通り:

- ErikEkstedt のオリジナル VAP は依存劣化、`vap_dataset` が private で再現困難
- MaAI は京大・井上研の **現役メンテ** 後継、`pip install maai` で即動作
- HuggingFace に **29 モデル** 公開（VAP / VAP_BC / VAP_Nod / VAP_MC / VAP_Prompt）
- 言語: 英・日・中・仏・trilingual

## 利用可能なモデル

`maai.util.get_available_models()` で一覧取得可能。代表例:

| モデル名 | タスク | 用途 |
|---|---|---|
| `maai-kyoto/vap_en` | turn-taking 二値 | **ITM ベースライン** |
| `maai-kyoto/vap_bc_en` | backchannel 予測 | 比較対象 |
| `maai-kyoto/vap_nod_jp` | うなずき予測 | 視覚タスクの参考 |
| `maai-kyoto/vap_mc_en` | ノイズ耐性版 | 比較対象 |

各モデルは frame_rate (5 / 10 / 12.5 / 20 Hz) と context_len (2.5 / 3 / 5 / 10 / 20 秒) の組合せで複数バリアント存在。

## 基本的な使い方

```python
from maai import Maai, MaaiInput, MaaiOutput

maai = Maai(
    mode="vap",
    lang="en",
    frame_rate=10,
    context_len_sec=5,
    audio_ch1=MaaiInput.Wav("speaker_a.wav"),
    audio_ch2=MaaiInput.Wav("speaker_b.wav"),
    device="cpu",
)
maai.start()

# 結果を取得
result_q = maai.result_dict_queue
while True:
    try:
        r = result_q.get(timeout=0.5)
    except Exception:
        break
    print(r["p_now"], r["p_future"], r["vad"])
```

## 出力の意味

毎フレーム（frame_rate に応じて）以下の dict が返る:

| キー | 型 | 内容 |
|---|---|---|
| `t` | float | unix timestamp |
| `x1`, `x2` | np.float32 (1600,) | 各話者の生音声チャンク（100ms @ 16kHz） |
| `p_now` | list[float] (2,) | `[話者1, 話者2]` 現在の発話確率 |
| `p_future` | list[float] (2,) | 近未来の発話確率（**VAP の核**） |
| `vad` | list[float] (2,) | 現在のVAD 出力 |

### 解釈

- `p_now=[0.8, 0.2]`: 話者1 がいま発話中、話者2 は無音
- `p_future=[0.3, 0.7]`: 近未来は話者2 が話す可能性が高い
- `p_now` と `p_future` の差分から **turn-shift の予兆** を読める

## マイク入力でリアルタイム動作

```python
maai = Maai(
    mode="vap",
    lang="en",
    frame_rate=10,
    audio_ch1=MaaiInput.Mic(),         # マイク
    audio_ch2=MaaiInput.Zero(),        # 無音（自分の発話のみ評価）
    device="cpu",
)
maai.start()
out = MaaiOutput.ConsoleBar()
out.update(maai.get_result())  # ターミナルにバー表示
```

## モデルの内部構造

```
maai/
├── encoder.py              # CPC エンコーダ
├── encoder_components.py
├── input.py                # Mic / Wav / Tcp / Zero 入力
├── model.py                # Maai クラス（メインオーケストレーション）
├── modules.py              # Transformer 層
├── objective.py            # VAP 目的関数
├── output.py               # ConsoleBar / GuiBar / GuiPlot 等
├── util.py                 # モデル列挙、ダウンロード
└── models/
    ├── config.py           # VapConfig
    ├── vap.py              # VAP モデル
    ├── vap_bc.py           # Backchannel VAP
    ├── vap_bc_2type.py
    ├── vap_nod.py          # うなずき
    └── vap_prompt.py       # プロンプト条件付け
```

## ITM v1 における利用方針

### Phase 1: ベースライン推論

MaAI をそのまま使い、AMI 上で標準指標（hold/shift accuracy）を再現。我々のマルチイベントヘッドはまだ載せない。

### Phase 2: マルチイベント拡張

VAP モデルから hidden state を取り出し、その上に独自の **3 つのハザードヘッド** を載せる:

```python
import torch
from maai.models.vap import VapModel  # 仮称

class ITMModel(torch.nn.Module):
    def __init__(self, vap_backbone: VapModel):
        super().__init__()
        self.backbone = vap_backbone
        # backbone の出力次元を取得
        d = self.backbone.hidden_dim  # 仮
        self.hazard_turn = torch.nn.Linear(d, 40)
        self.hazard_bc = torch.nn.Linear(d, 40)
        self.hazard_overlap = torch.nn.Linear(d, 40)

    def forward(self, x1, x2):
        h = self.backbone(x1, x2, return_hidden=True)
        return {
            "h_turn": torch.sigmoid(self.hazard_turn(h)),
            "h_bc": torch.sigmoid(self.hazard_bc(h)),
            "h_overlap": torch.sigmoid(self.hazard_overlap(h)),
        }
```

これは Phase 2 の主要実装作業。

### Phase 3: 視覚追加

MediaPipe で抽出した顔特徴を、上記 hidden state にクロスアテンションで融合してからハザードヘッドに渡す。

## ライセンス注意

| 要素 | ライセンス |
|---|---|
| MaAI コード | MIT |
| MaAI モデル重み | **academic only** |
| ITM 派生コード | BSD 2-Clause（独立） |
| ITM 派生モデル重み | **未決定**（重要） |

ITM が MaAI 重みから派生する fine-tune モデルを作る場合、MaAI のライセンス制約を継承する可能性がある。**完全な BSD 2-Clause リリースを目指す場合は、CPC + Self/Cross Attention のスクラッチ学習** が必要になる。

これは v1 公開前に詰めるべき論点。

## Phase 1 ベースライン数値（AMI 5 ミーティング）

`scripts/eval_maai_on_ami.py --all` の結果（5 ミーティング、計 165 分）:

| 会議 | 評価 ch | dur (s) | Frame VAD | Hold | Shift | Overall |
|---|---|---:|---:|---:|---:|---:|
| ES2002a | B, D | 1273 | 0.936 | 30/55 = 0.545 | 34/54 = 0.630 | **0.587** |
| ES2002b | C, D | 2280 | 0.935 | 40/101 = 0.396 | 42/67 = 0.627 | **0.488** |
| ES2002c | C, D | 2424 | 0.958 | 56/111 = 0.505 | 19/36 = 0.528 | **0.510** |
| IS1000a | A, D | 1583 | 0.904 | 96/127 = 0.756 | 26/53 = 0.491 | **0.678** |
| IS1000b | C, B | 2344 | 0.933 | 86/134 = 0.642 | 23/52 = 0.442 | **0.586** |
| **POOLED** | — | **9904** | **0.935** | **308/528 = 0.583** | **144/262 = 0.550** | **0.572** |

### 解釈

- **Frame VAD 精度 93.5% (pooled)** ─ MaAI の `argmax(p_now)` は GT と高い一致を示す。VAD としては十分実用レベル
- **Hold/shift 精度 57.2% (pooled)** ─ VAP 論文の Switchboard 数値（75〜80%）より大幅に低い
- **会議ごとのばらつき大** （48.8% 〜 67.8%）。IS1000a だけ突出して良いが評価サンプル数も少ない

### 想定通りの低スコア要因

1. **ドメイン差**: MaAI 英語 VAP は **電話会話（Switchboard）** で学習、AMI は **4 人会議**。音響条件と会話動態が大きく異なる
2. **2 of 4 ch のみ評価**: 残り 2 名（C/A など）が話している時、ground truth がノイジーになる。「mutual silence」も実際は別話者が発話中の可能性
3. **AMI で fine-tune していない素のベースライン**

### Hold が低めで Shift がさらに低い

会議全体での話し手交替頻度が低い（Hold 528 vs Shift 262）。MaAI のデフォルト出力は中央値付近にあり、Shift の検出感度が足りていない可能性。

### ITM v1 の目標

このベースライン（57.2%）を v1 で超えること。**目標 hold/shift accuracy ≥ 70%**。Phase 2 で AMI を使った fine-tune + マルチイベント拡張で達成を狙う。

## 関連ページ

- [既存モデル](../research/existing-models.md) — MaAI 詳細
- [v1 アーキテクチャ](../design/architecture.md) — ITM での使い方全体図
- [環境構築](environment.md) — セットアップ
- [AMI Corpus](ami-corpus.md) — 学習データ

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

## 関連ページ

- [既存モデル](../research/existing-models.md) — MaAI 詳細
- [v1 アーキテクチャ](../design/architecture.md) — ITM での使い方全体図
- [環境構築](environment.md) — セットアップ
- [AMI Corpus](ami-corpus.md) — 学習データ

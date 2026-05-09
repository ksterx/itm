# データ戦略

> **Status**: stable | **Last reviewed**: 2026-05-09
>
> ITM v1 でどのデータをどのフェーズで使うかの戦略。個人入手可能性が制約。

## TL;DR

- **メインデータ**: AMI Corpus（CC BY 4.0、即時 DL 可、5 ミーティング ~3GB から開始）
- **補助データ**: Smart Turn v3.1 train（HF、BSD、filler 補強）
- **将来候補**: CANDOR（個人申請可能性）、Multi-TPC（三者拡張）
- **死んだ計画**: 日本語データ全般（アカデミック署名/購入が必須で断念）

## フェーズ別データ計画

| Phase | 主データ | 補助 | 用途 |
|---|---|---|---|
| Phase 1（ベースライン） | AMI 5 ミーティング | — | MaAI 推論・指標再現 |
| Phase 2（マルチイベント） | AMI 全 168 ミーティング (~100GB) | Smart Turn v3.1 (filler 補助) | 主損失 + 補助損失 |
| Phase 3（視覚追加） | AMI 全 (映像込み) | — | MediaPipe + late fusion |
| Phase 4（量子化） | Phase 3 の重み | — | int8 QAT、評価のみ |
| v2（rPPG 等） | AMI + 自前小規模 | — | 呼吸シグナル統合 |

## AMI Corpus（メイン）

### 選定理由

- **CC BY 4.0**: 商用 OK、個人 OK、登録不要、即 DL
- **マルチモーダル**: 音声（per-speaker）+ 映像（per-speaker close-up）
- **詳細注釈**: 16 種 dialog act、segments、words、headGesture 等
- **規模**: 100 時間、ターンテイキング研究の世界標準
- **再現性**: 公開データなので査読対象になる場合の比較が容易

### 既知の制約

- **会議シナリオ**: 日常会話とは性質が異なる（タスク指向、4 人参加）
- **英語のみ**: 多言語対応は v2 以降
- **音響条件**: ヘッドセットマイクで録音。雑音耐性は別途必要

### サブセット選定

Phase 1 の 5 ミーティング（~3GB）の選び方:

```
ES2002a, ES2002b, ES2002c  # Edinburgh group 1, 連続セッション
IS1000a, IS1000b           # Idiap group 1, 別グループでの検証
```

異なるグループ・異なるセッションを混ぜることで、speaker overfitting を防ぐ。

## Smart Turn v3.1 (補助)

### 選定理由

- **BSD 2-Clause**: 最も寛容なライセンス
- **HuggingFace 即 DL**: `datasets.load_dataset("pipecat-ai/smart-turn-data-v3.1-train")`
- **filler ラベル**: midfiller / endfiller が AMI にない補完情報
- **大規模**: 270k samples で補助タスクの学習に十分

### 重要な制約

- **単一話者データ**: turn-shift / backchannel / overlap は学習できない
- **83% 合成音声**: TTS 由来でドメイン差あり
- **23 言語**: 英語以外は別途 fine-tune 検討

### 用途の絞り込み

ITM v1 では **filler 補助タスクのみ** に使う:

```python
# 補助損失（Smart Turn データから学習）
L_filler = BCE(pred_midfiller, gt_midfiller) + BCE(pred_endfiller, gt_endfiller)
```

主タスク（マルチイベントハザード）には AMI のみ使う。

## 個別アクセス手順

### AMI Corpus

```bash
# 注釈のみ（22.9MB）
python scripts/download_ami_subset.py --annotations-only

# 5 ミーティング、音声のみ（~640MB）
python scripts/download_ami_subset.py --audio-only

# 5 ミーティング、音声+映像（~3GB）
python scripts/download_ami_subset.py
```

詳細は [AMI Corpus 詳細](../implementation/ami-corpus.md)。

### Smart Turn v3.1

```python
from datasets import load_dataset
ds = load_dataset("pipecat-ai/smart-turn-data-v3.1-train", split="train", streaming=True)
# 36.8GB なので streaming 推奨
```

## v2 で検討するデータ

### CANDOR（挑戦価値あり）

- 1,656 会話、850 時間、ビデオ通話の自然対話
- 個人申請: https://betterup-data-requests.herokuapp.com
- "Independent researcher" で記入してみる価値あり
- 通れば VAP 系の標準学習データに乗れる

### Multi-TPC

- Nature Sci Data 公開、Zenodo
- 三者対話 + 視線 + モーション
- ITM の三者拡張 (v2/v3) で活用候補

### 自前収録（最終手段）

- 自分 + 友人で 10〜30 時間収録
- 倫理：被験者の同意必須、データ公開時はデータポリシー策定
- 時間がかかる（数ヶ月）

## 死んだ計画（参考）

| 候補 | 死因 |
|---|---|
| ~~CEJC~~ | ¥50,000 + 機関契約 |
| ~~NoXi+J~~ | 永続職アカデミック EULA |
| ~~Hazumi 完全版~~ | NII IDR 機関契約 |
| ~~TEIDAN~~ | 未公開、京大研究室問合せ必要 |
| ~~Obi & Funakoshi VREi~~ | 公開チャネル不在、著者問合せ必要 |
| ~~Switchboard / Fisher~~ | LDC 有料 |

## 関連ページ

- [データセット](../research/datasets.md) — 全データセットの詳細
- [AMI Corpus 詳細](../implementation/ami-corpus.md) — ダウンロード手順
- [ラベル生成](label-generation.md) — AMI からのラベル変換
- [v1 アーキテクチャ](architecture.md) — どのデータをどう使うか全体図

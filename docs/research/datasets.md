# データセット

> **Status**: stable | **Last reviewed**: 2026-05-09
>
> ターンテイキング学習・評価に使用可能なデータセットと、ITM プロジェクトでの個人入手可能性。

## 個人入手可能性まとめ

著者は **アカデミック所属なし、購入予算なし** の独立研究者。この制約下で使えるかを優先指標とする。

| データセット | 規模 | 言語 | モダリティ | 個人入手 | ITM での用途 |
|---|---|---|---|---|---|
| **AMI Corpus** | 100h | 英 | 音声+映像+注釈 | ✅ CC BY 4.0 | **メイン** |
| **VoxConverse** | 50h+ | 英 | 音声 (映像 YouTube 別取得) | ✅ CC BY 4.0 | 補助 |
| **AVA-ActiveSpeaker** | 38.5h | 英 | 映像 + 音声活動 | ✅ CC BY 4.0 | 補助 |
| **Multi-TPC (2025)** | 三者対話 | 英 | 音声+映像+モーション+視線 | ✅ Zenodo | 補助 (将来) |
| **Smart Turn v3.1 train** | 270k samples | 23言語 | 音声 | ✅ HF, BSD | 補助（endpoint+filler のみ） |
| **CANDOR** | 850h | 英 | 音声+映像 | ⚠️ 個人申請可能性 | （挑戦価値あり） |
| ~~Switchboard~~ | 260h | 英 | 音声 | ❌ LDC 有料 | 不可 |
| ~~Fisher~~ | 2000h | 英 | 音声 | ❌ LDC | 不可 |
| ~~CEJC~~ | 200h | 日 | 音声+映像+対話行為 | ❌ ¥50,000 + 機関契約 | 不可 |
| ~~NoXi+J~~ | 11.6h | 日 | 音声+映像 | ❌ 永続職アカデミック EULA | 不可 |
| ~~Hazumi~~ | 181人 | 日 | マルチモーダル | ❌ NII IDR 機関契約 | GitHub 公開分のみ可 |
| ~~TEIDAN~~ | 三者会話 | 日 | 音声+映像 | ❌ 未公開 | 不可 |
| ~~Obi & Funakoshi VREi~~ | 30人 | 日 | 顔+呼吸ベルト | ❌ 公開チャネル不在 | 不可 |

## AMI Corpus（メインデータ）

### 概要

- 100 時間の会議録、4 人参加（基本）、シナリオ会議とフリー会議
- 英語、Edinburgh / Idiap / Brno など複数機関
- ライセンス: **CC BY 4.0**（商用利用可、個人利用 OK）

### 内容

| ストリーム | 内容 |
|---|---|
| **音声 Headset-{0,1,2,3}** | per-speaker 音声、各 ~40MB / 30 分会議 |
| **音声 Mix-Headset** | ミックス音声、~40MB |
| **映像 Closeup{1,2,3,4}** | per-speaker 顔映像、各 40〜60MB |
| **映像 Corner** | 全体俯瞰、~50MB |

1 ミーティングあたり ~600MB、全 168 ミーティングで ~100GB。

### URL パターン

```
Audio (per-speaker):
  https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<MID>/audio/<MID>.Headset-{0,1,2,3}.wav

Video:
  https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<MID>/video/<MID>.Closeup{1,2,3,4}.avi

Annotations:
  https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
```

### アノテーション

manual_1.6.2 (CC BY 4.0、22.9MB) に以下が含まれる:

- **dialogueActs/** — 16 種の対話行為（bck=Backchannel, stl=Stall, fra=Fragment, inf=Inform, sug=Suggest, ass=Assess, など）
- **disfluency/** — 言いよどみ
- **segments/** — IPU/発話単位の時間境界
- **words/** — 単語レベル時間整合
- **headGesture/** / **handGesture/** / **movement/** — 動作
- **focus/** — 注視
- **topics/** — 話題境界

ITM のマルチイベントラベル生成に必要なものがすべて揃っている。詳細は [ラベル生成](../design/label-generation.md)。

## Smart Turn v3.1 (補助データ)

### 概要

- HuggingFace `pipecat-ai/smart-turn-data-v3.1-train`
- 270,429 samples、36.8GB
- 23 言語、英語 25%
- BSD 2-Clause

### スキーマ

```python
{
    "id": str,
    "audio": {"array": np.array, "sampling_rate": 16000},
    "language": str,
    "endpoint_bool": bool,    # 発話完了か
    "midfiller": bool,         # 発話中のフィラー (NaN 20%)
    "endfiller": bool,         # 発話末のフィラー
    "synthetic": bool,         # 83% が True (Chirp3 TTS)
    "spoken_text": str,
    "dataset": str,
}
```

### 重要な制約

**Smart Turn は単一話者のエンドポイント+フィラー検出データ**。turn-shift / backchannel / overlap の二者会話イベントは含まれない。ITM では補助的に endpoint + filler 認識の補強に使う。

## Multi-TPC (2025)

- Nature Scientific Data 公開、Zenodo にDOI
- 三者対話、音声+全身モーション+視線追跡
- ターンテイキング+多人数会話のリアル研究に直結
- ITM v1 では使わないが、v2 の三者拡張で活用候補

## VoxConverse / AVA-ActiveSpeaker

### VoxConverse

- 50時間+、政治討論・ニュース番組
- 話者ダイアリゼーション RTTM
- 音声は GitHub 直 DL、映像は YouTube ID + タイムスタンプから自前抽出
- 補助評価用

### AVA-ActiveSpeaker

- 38.5時間、YouTube 映画
- 顔バウンディング + 音声活動アノテーション
- CVDF が AWS S3 で動画配信
- アクティブスピーカー検出の標準ベンチ

## CANDOR（挑戦価値あり）

- 1,656 会話、850時間、ビデオ通話の自然対話
- 個人申請フォームに「Independent researcher」記入で通る可能性あり
- 申請: https://betterup-data-requests.herokuapp.com
- ダメ元で申請して、通れば VAP 系の標準学習データに乗れる

## ITM v1 のデータ計画

| Phase | 主データ | 補助 |
|---|---|---|
| Phase 1〜2 (ベースライン + マルチイベント) | AMI Corpus（5 ミーティング、~3GB） | — |
| Phase 3 (視覚追加) | AMI 映像 | — |
| Phase 4〜5 (量子化・公開) | AMI 全体 | Smart Turn v3.1 (filler 補強) |
| v2 (rPPG・V-JEPA) | AMI + 自前小規模映像 | — |

詳細は [データ戦略](../design/data-strategy.md)。

## 関連ページ

- [調査の概観](overview.md) — データセットを含む全体像
- [AMI Corpus 詳細](../implementation/ami-corpus.md) — ダウンロード手順とディレクトリ構造
- [ラベル生成](../design/label-generation.md) — AMI から ITM ラベルへの変換
- [データ戦略](../design/data-strategy.md) — どこで何を使うか

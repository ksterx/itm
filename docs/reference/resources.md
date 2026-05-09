# リソース

> **Status**: stable | **Last reviewed**: 2026-05-09
>
> ツール・OSS・データセット・コミュニティへのリンク集。

## OSS リポジトリ

### ベース実装

| プロジェクト | URL | 用途 |
|---|---|---|
| **MaAI (旧 VAP-Realtime)** | https://github.com/maai-kyoto/maai | **ITM ベースライン** |
| Erik Ekstedt VAP | https://github.com/ErikEkstedt/VoiceActivityProjection | オリジナル参考（保守停止） |
| TurnGPT | https://github.com/ErikEkstedt/TurnGPT | テキストベース（保守停止） |
| Smart Turn | https://github.com/pipecat-ai/smart-turn | 軽量参考、BSD-2 |
| TurnSense | https://github.com/latishab/turnsense | 超軽量、Raspberry Pi |
| Easy Turn | https://github.com/ASLP-lab/Easy-Turn | 4状態分類 |
| Moshi | https://github.com/kyutai-labs/moshi | フルデュプレックス基盤 |
| VAPwithAudioFaceEncoders | https://github.com/sagatake/VAPwithAudioFaceEncoders | 顔エンコーダ統合 |

### 視覚・呼吸

| プロジェクト | URL | 用途 |
|---|---|---|
| **rPPG-Toolbox** | https://github.com/ubicomplab/rPPG-Toolbox | NeurIPS 2023、rPPG SOTA 集 |
| MediaPipe | https://github.com/google-ai-edge/mediapipe | 顔・姿勢検出 |
| OpenFace 2.0 | https://github.com/TadasBaltrusaitis/OpenFace | FAU 抽出標準ツール |

### 音声フロントエンド・データ

| プロジェクト | URL | 用途 |
|---|---|---|
| Pipecat | https://github.com/pipecat-ai/pipecat | リアルタイム対話フレームワーク |
| Silero VAD | https://github.com/snakers4/silero-vad | エッジ VAD 標準 |

## HuggingFace モデル

### MaAI 系列（29 モデル）

| モデル | 用途 |
|---|---|
| `maai-kyoto/vap_en` | 英語 VAP（複数 frame_rate / context_len） |
| `maai-kyoto/vap_jp` | 日本語 VAP |
| `maai-kyoto/vap_bc_*` | バックチャネル予測 |
| `maai-kyoto/vap_nod_*` | うなずき予測 |
| `maai-kyoto/vap_mc_*` | ノイズ耐性 |
| `maai-kyoto/vap_prompt_*` | プロンプト条件付け |

### Smart Turn

- `pipecat-ai/smart-turn-v2` (94.8M)
- `pipecat-ai/smart-turn-v3` (8M, int8)
- `onnx-community/smart-turn-v3-ONNX`

### その他

- `ASLP-lab/Easy-Turn` (4状態)
- `kyutai/moshiko-pytorch-bf16` (7B)

## データセット

### ITM で実際に使うもの

| データセット | URL | アクセス |
|---|---|---|
| **AMI Corpus** | https://groups.inf.ed.ac.uk/ami/corpus/ | 即 DL、CC BY 4.0 |
| **Smart Turn v3.1 train** | https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.1-train | HF、BSD |
| Multi-TPC | Nature Sci Data 2026 | Zenodo |
| AVA-ActiveSpeaker | https://research.google.com/ava/ | CVDF S3 |
| VoxConverse | https://github.com/joonson/voxconverse | GitHub 直 DL |

### 個人不可（参考）

- CEJC: https://www2.ninjal.ac.jp/conversation/cejc.html
- NoXi+J: https://multimediate.perceptualui.org/datasets/Dataset_NoXi/
- Hazumi: https://www.nii.ac.jp/dsc/idr/rdata/Hazumi/
- CANDOR: https://www.betterup.com/research/candor-research（個人申請の可能性あり）

## 評価ベンチマーク

| 名前 | 用途 |
|---|---|
| Full-Duplex-Bench (arXiv:2503.04721) | フルデュプレックスシステム評価 |

## ドキュメンテーション・公式サイト

- VAP デモ: https://erikekstedt.github.io/VAP/
- Pipecat Docs: https://docs.pipecat.ai/
- Smart Turn Blog: https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/

## サーベイ・レビュー

- Castillo-López et al. (IWSDS 2025): A Survey of Recent Advances on Turn-taking Modeling — https://aclanthology.org/2025.iwsds-1.27
- Skantze (2021): Turn-taking in conversational systems and HRI: a review — Computer Speech & Language

## コミュニティ・議論

- HuggingFace Hub の `turn-taking` タグ
- Pipecat Discord（Smart Turn のコミュニティが活発）

## ITM プロジェクト

- リポジトリ: https://github.com/ksterx/itm
- ドキュメント: https://ksterx.github.io/itm/
- Issue: https://github.com/ksterx/itm/issues

## 関連ページ

- [論文リスト](papers.md) — 学術論文
- [用語集](glossary.md) — 専門用語
- [調査の概観](../research/overview.md) — リサーチ全体像

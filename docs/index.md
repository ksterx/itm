# It's My Turn (ITM)

> **Status**: draft | **Last reviewed**: 2026-05-16
>
> 動画（音声 + 映像）からリアルタイムに、対話相手が話そうとしているか・どう話そうとしているかを発声前に予測する、エッジ実装可能なマルチモーダル AI。

## 一言で

対話で「いま相手が喋り出しそうか」を **発声前** に当てる小さい AI を作っている。音声しか見ない既存研究（VAP / Smart Turn 系）に対して、**(a) 話者交代・相づち・割り込みを同じモデルで同時に予測**し、**(b) 顔映像も使い**、**(c) スマホ・PC で動く軽量さ**を狙う。学習・評価には公開会議録音 **AMI Corpus**（後述）を使用。

## TL;DR

- **目的**: ターンテイキング（話者交代）を **発声前 (proactive)** に予測する。VAP / MM-VAP / DualTurn が解いている問題の延長。
- **独自性**: (1) **マルチイベント・サバイバルハザード**（turn-shift / backchannel / overlap を統一的にモデル化）、(2) **エッジ実装** (< 10M params、CPU リアルタイム)、(3) **顔のみからの呼吸推定**（rPPG 等の派生信号）を視覚シグナルに組み込む。
- **戦略**: HuggingFace 先行リリース (v1) → 反応を見て論文化 (v2)。
- **現状**: Phase 1 完了（baseline AUC 0.701 確立）、Phase 2-B v4 で AUC 0.566 + Overall 0.608 を達成（baseline real-time 0.586 を上回る）。

## 用語の最小セット

| 用語 | 1 行解説 | 詳細 |
|---|---|---|
| **AMI Corpus** | エディンバラ大学が公開した約 100 時間の英語 4 人会議録（音声＋注釈、CC BY 4.0）。本プロジェクトの学習・評価のメインデータ | [データセット調査](research/datasets.md) / [AMI 取り扱い](implementation/ami-corpus.md) |
| **VAP** | Voice Activity Projection (Ekstedt & Skantze 2022)。「次に誰が喋るか」を音声から予測する代表的モデル。本プロジェクトのベースライン | [既存モデル](research/existing-models.md) |
| **MaAI** | VAP 系をリアルタイム化したオープン実装（京大）。我々はこれを backbone として使用 | [MaAI ベースライン](implementation/maai-baseline.md) |
| **turn-shift / hold** | 沈黙のあとに「別の話者」が喋れば turn-shift、「同じ話者」が続ければ hold。これが評価指標 | [ターンテイキング 101](research/turn-taking-101.md) |

## このサイトの読み方

| もし... | 読むべきページ |
|---|---|
| プロジェクトの動機を知りたい | [プロジェクト概要](about/motivation.md) → [解く問題](about/problem.md) |
| 既存モデルと我々の差分を知りたい | [新規性](design/novelty.md) → [既存モデル](research/existing-models.md) |
| Real-time VAP / MM-VAP / Smart Turn v3 とのアーキテクチャ差分 | [アーキテクチャ § 先行モデルとの実装差分](design/architecture.md#先行モデルとの実装差分) |
| ターンテイキング AI の研究領域を知りたい | [調査の概観](research/overview.md) |
| 我々のモデル設計を知りたい | [v1 アーキテクチャ](design/architecture.md) |
| 自分でも動かしたい | [環境構築](implementation/environment.md) → [MaAI ベースライン](implementation/maai-baseline.md) |
| 専門用語が分からない | [用語集](reference/glossary.md) |

## クイックスタート

```bash
git clone https://github.com/ksterx/itm.git
cd itm
brew install portaudio
uv venv -p 3.11 .venv && source .venv/bin/activate
uv pip install -e .
python scripts/test_maai_inference.py
```

詳細は [環境構築](implementation/environment.md) を参照。

## ドキュメント方針

このサイトは **Living Docs**: 仮説や設計が変わるのは前提で、古い決定は明示的にアーカイブして痕跡を残す。詳細は [ドキュメント方針](meta/documentation-policy.md)。

## ライセンス

リポジトリのコードは BSD 2-Clause、ドキュメントは CC BY 4.0。詳細は [LICENSE](https://github.com/ksterx/itm/blob/main/LICENSE)。

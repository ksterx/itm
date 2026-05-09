# It's My Turn (ITM)

> **Status**: draft | **Last reviewed**: 2026-05-09
>
> 動画（音声 + 映像）からリアルタイムに、対話相手が話そうとしているか・どう話そうとしているかを発声前に予測する、エッジ実装可能なマルチモーダル AI。

## TL;DR

- **目的**: ターンテイキング（話者交代）を **発声前 (proactive)** に予測する。VAP / MM-VAP / DualTurn が解いている問題の延長。
- **独自性**: (1) **マルチイベント・サバイバルハザード**（turn-shift / backchannel / overlap を統一的にモデル化）、(2) **エッジ実装** (< 10M params、CPU リアルタイム)、(3) **顔のみからの呼吸推定**（rPPG 等の派生信号）を視覚シグナルに組み込む。
- **戦略**: HuggingFace 先行リリース (v1) → 反応を見て論文化 (v2)。
- **現状**: Phase 0 完了。MaAI を baseline として動かし、AMI 注釈構造を解明済み。

## このサイトの読み方

| もし... | 読むべきページ |
|---|---|
| プロジェクトの動機を知りたい | [プロジェクト概要](about/motivation.md) → [解く問題](about/problem.md) |
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

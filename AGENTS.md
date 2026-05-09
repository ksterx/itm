# AGENTS.md

このリポジトリで AI エージェント（Claude Code 等）と協働するためのガイド。

## プロジェクト概要

**It's My Turn (ITM)** — 動画からリアルタイムにターンテイキングを proactive 予測するマルチモーダル AI。

詳細は [README.md](README.md) と [docs/index.md](docs/index.md)。

## エージェント向けヒント

### まず読むべき場所

1. [docs/about/motivation.md](docs/about/motivation.md) — なぜこのプロジェクトか
2. [docs/design/architecture.md](docs/design/architecture.md) — v1 アーキテクチャ
3. [docs/about/roadmap.md](docs/about/roadmap.md) — 現状と次のフェーズ
4. [docs/meta/changelog.md](docs/meta/changelog.md) — 直近の変更

### コーディング規約

- Python 3.11+、`str | None` 等の現代的な型ヒント
- `uv` でパッケージ管理（`uv pip install`, `uv add`）
- `pydantic` v2
- `anyio` for async
- フォーマット: `ruff format`、リント: `ruff check`

### ドキュメント編集時

- [docs/meta/documentation-policy.md](docs/meta/documentation-policy.md) を厳守
- WHY を書く、WHAT は書かない
- 古くなった内容は削除か `## 過去の判断（破棄）` 節へ
- 新しい慣行はメタドキュメントを更新

### Bash コマンド

- `git status` 等の読み取り系は自由に
- ファイルや branch の削除など破壊的操作は事前確認
- `git push` / `gh pr create` はユーザー指示時のみ

### 避けるべきこと

- ErikEkstedt/VoiceActivityProjection を直接 fork する設計提案（依存劣化のため MaAI を使う）
- 日本語データセット前提の設計（個人入手不可、英語 AMI を使う）
- Smart Turn データで turn-shift 学習（データの実体は単一話者 endpoint）

詳細は [docs/about/roadmap.md](docs/about/roadmap.md) の「死んだ計画」セクション。

# ドキュメント方針

このプロジェクトのドキュメントをどう書き、どう運用するかの指針。**書く前にこのページを読むこと**。

---

## 6 つの原則

### 1. Docs as Code

ドキュメントはソースコードと同じレポジトリに、同じ Pull Request フローで更新する。`docs/` 配下の Markdown が唯一の真実 (Single Source of Truth)。Wiki やスプレッドシートに散逸させない。

### 2. Living Docs

ドキュメントは「完成品」ではなく「現在地のスナップショット」。研究プロジェクトとして仮説や設計が変わるのは前提。古くなったページは **削除か明示的にアーカイブ** する。`status` メタデータで状態を示す:

- `draft` — 議論中、変更頻繁
- `stable` — 安定、参照に使える
- `deprecated` — 古い決定、参考までに残す
- `archived` — 過去の議論、現在は別ページが正

### 3. WHY を書く、WHAT は書かない

「コードは WHAT を語る、ドキュメントは WHY を語る」。実装の詳細やパラメータ表は大半が陳腐化する。**意思決定の理由・背景・代替案・却下理由** を残すことに価値がある。

### 4. 1ページ 1テーマ

長大なページは読まれない。1 ページ 1 テーマ、長くて A4 換算 3〜5 ページ。横断的に関連するなら相互リンクで繋ぐ。

### 5. 二重化禁止

同じ事実を複数ページに書かない。事実は 1 箇所に書き、他は参照する。**README と docs を二重化しない** — README はクイックスタート＋docs への入口に絞る。

### 6. 国際的な公開を意識した言語選択

| 場所 | 言語 | 理由 |
|------|------|------|
| README.md | 英語 | GitHub 上での発見性、海外コミュニティ |
| docs/ | 日本語（現状） | 著者の作業効率、現段階は内部議論ログ |
| コード・コミット | 英語 | 標準慣行 |
| Issue/PR | 日本語可 | 個人プロジェクトとして |

将来 v1 を HuggingFace 公開する段階で、`docs/` を英語化するか、`docs-en/` を別途作るかを再検討する。

---

## ディレクトリ構造

```
docs/
├── index.md                # トップ：ITM とは何か、入口
├── about/                  # プロジェクトの位置づけ
│   ├── motivation.md       # なぜやるか
│   ├── problem.md          # 何の問題を解くか
│   └── roadmap.md          # Phase 0/1/2/v2/v3 のロードマップ
├── research/               # 調査・先行研究
│   ├── overview.md         # サーベイの概観
│   ├── turn-taking-101.md  # ターンテイキングとは
│   ├── existing-models.md  # VAP, MM-VAP, DualTurn, Moshi, Smart Turn
│   ├── visual-cues.md      # 視覚シグナル（FAU, gaze, 呼吸）
│   ├── datasets.md         # 利用可能データセット
│   └── related-work.md     # Obi & Funakoshi, Włodarczak など
├── design/                 # 我々の設計判断
│   ├── architecture.md     # v1 アーキテクチャ
│   ├── multi-event-hazard.md  # マルチイベント・サバイバルハザード
│   ├── label-generation.md # AMI dialog act → ITM event のマッピング
│   ├── data-strategy.md    # AMI + Smart Turn 補助
│   └── novelty.md          # 既存研究との差別化
├── implementation/         # 実装ログ
│   ├── environment.md      # 環境構築
│   ├── maai-baseline.md    # MaAI を baseline で動かす
│   ├── ami-corpus.md       # AMI ダウンロード・構造
│   └── pipeline.md         # 学習パイプライン（Phase 2 以降）
├── reference/              # 参照資料
│   ├── glossary.md         # 用語集
│   ├── papers.md           # 論文リスト（実在確認済みフラグ付き）
│   └── resources.md        # ツール・OSS・データセットのリンク集
└── meta/
    ├── documentation-policy.md  # このページ
    └── changelog.md             # ドキュメントの主要変更履歴
```

---

## ページ書式

各ページの冒頭に Front Matter ではなく、見出し下の説明段落で **status** と **last reviewed** を書く:

```markdown
# ページタイトル

> **Status**: stable | **Last reviewed**: 2026-05-09
>
> このページの一文サマリ。

本文...
```

### 必須セクション

長いページは以下を備えると読まれやすい:
- **TL;DR**: 3〜5 行のまとめ
- **本文**: 構造化された記述
- **未解決事項 / 次のアクション**: 該当する場合
- **関連ページ**: 相互リンク

### 図表

- **図**: Mermaid を使う（コードブロックでバージョン管理可能）
- **数式**: LaTeX (`$...$`、`$$...$$`)、MathJax で描画
- **画像**: `docs/assets/` に置き、相対パスで参照
- **表**: Markdown table で十分。複雑なら HTML table を許容

---

## 何を書く / 何を書かない

### 書く
- 設計判断の **理由と却下した代替案**
- 仮説と、それを検証する実験計画
- データセット選定の理由・制約・代替策
- API の使い方（Quickstart）
- 概念の図解（特に視覚的に説明できるもの）

### 書かない
- API の網羅的リファレンス（コード docstring と Sphinx-style 自動生成に任せる）
- ビルドログ・実験結果の生 dump（必要なら別リポジトリに artifact 置く）
- 個人的な作業メモ（`tmp/` か別の private ノート）
- 一時的な ToDo（Issue または GitHub Projects に）

---

## 更新の流れ

1. **新しい知見が出たら**: 該当ページを編集して PR
2. **設計判断が変わったら**: 旧設計を `## 過去の判断（破棄）` 節に残し、新設計を本文化。理由を必ず書く
3. **長くなりすぎたら**: 分割。元ページは目次＋各ページへのリンクに

---

## ビルドとデプロイ

- ローカル確認: `mkdocs serve`（ホットリロード）
- ビルド: `mkdocs build` → `site/`
- 公開: `main` ブランチへの push で GitHub Actions が自動ビルド・GitHub Pages にデプロイ
- URL: `https://ksterx.github.io/itm/`

詳細は [`.github/workflows/docs.yml`](https://github.com/ksterx/itm/blob/main/.github/workflows/docs.yml) を参照。

---

## このページの位置づけ

> **Status**: stable | **Last reviewed**: 2026-05-09
>
> ドキュメントを書く全員（現状は著者のみ）が出発点とすべきメタドキュメント。
> 新しい慣行を導入したらここを更新する。

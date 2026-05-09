# 変更履歴

> **Status**: stable | **Last reviewed**: 2026-05-09
>
> ドキュメント・設計の主要変更を記録する。コードの詳細は git log / GitHub Releases に任せる。

## 2026-05-09

### Phase 1 ベースライン数値を取得

AMI ES2002a で MaAI 英語 VAP を実行（`scripts/eval_maai_on_ami.py`）:

- **Frame VAD 精度 93.6%**（MaAI の `p_now` argmax が GT と一致）
- **Hold/shift 精度 58.7%**（109 mutual silence、22 分のメタ会議）

VAP 論文の Switchboard 数値（75〜80%）より低いのは想定内（ドメイン差 + 4 人会議の 2 ch 評価）。
詳細は [MaAI ベースライン](../implementation/maai-baseline.md#phase-1-ベースライン数値ami-es2002a)。

### MkDocs Material でドキュメントサイトを構築

- 構造: about / research / design / implementation / reference / meta の 6 セクション
- GitHub Actions による gh-pages デプロイ
- ドキュメント方針を `meta/documentation-policy.md` で明示
- これまでの議論内容を docs に統合

### v1 アーキテクチャを確定

- ベース実装: **MaAI** (旧 VAP-Realtime)
- 出力: **マルチイベント・サバイバルハザード**（turn-shift / backchannel / overlap）
- 視覚統合: MediaPipe + 後期融合
- データ: AMI Corpus (メイン) + Smart Turn v3.1 (filler 補助)

### 死んだ計画

以下を断念し、明示的に痕跡を残す:

- 日本語データ（CEJC, NoXi+J, Hazumi, TEIDAN, Obi&Funakoshi VREi）→ 個人入手不可
- ErikEkstedt/VoiceActivityProjection 直接 fork → 依存劣化
- Smart Turn データで turn-shift 学習 → データの実体は単一話者
- 上半身 optical flow による呼吸 → 顔のみシナリオで動かない（rPPG 等に変更、v2 で）

### Phase 0 完了

- 環境構築（Python 3.11 venv + maai）
- MaAI 動作確認（出力構造把握: `t, x1, x2, p_now, p_future, vad`）
- AMI 注釈構造解明（16 種 dialog act）
- Smart Turn データ構造確認（endpoint+filler、turn-shift ではない）

## 2026-05-01（プロジェクト開始）

### 初回技術調査

- ターンテイキング AI の包括サーベイ実施
- 結果を report.html / report.pdf にまとめる
- VAP / MM-VAP / Moshi / DualTurn / Smart Turn 等の主要モデルを把握
- 視覚シグナル（FAU・視線・呼吸）の重要性を確認

## 関連ページ

- [ロードマップ](../about/roadmap.md) — Phase 別計画と現状
- [ドキュメント方針](documentation-policy.md) — どう書くか

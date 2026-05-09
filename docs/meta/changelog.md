# 変更履歴

> **Status**: stable | **Last reviewed**: 2026-05-09
>
> ドキュメント・設計の主要変更を記録する。コードの詳細は git log / GitHub Releases に任せる。

## 2026-05-10

### Phase 2-B：ITM モデル + 学習スクリプト実装

- `src/itm/models/itm_model.py`: VAP backbone (MaAI) + 3 hazard heads
  + freeze encoder option。`build_itm_model()` ヘルパで MaAI 重み読み込み
- `src/itm/training.py`: `compute_loss` (multi-event survival NLL),
  `train_step`, `eval_step`
- `scripts/train_itm.py`: AMIDataset → DataLoader → 学習ループ →
  checkpoint。`--all`, `--max-steps`, `--device {cpu,cuda,mps}` 等の CLI
- 動作確認: CPU 8 step で loss 0.65 → 0.50 と単調減少
- パラメータ数: total 8.1M, trainable 3.7M (CPC encoder frozen)

ユニットテストを追加（合計 48 件）。`@pytest.mark.slow` で MaAI 重みを
ロードする統合テストを切り分け、CI でデフォルトはスキップ。

詳細は [学習パイプライン](../implementation/pipeline.md#itm-モデル--学習スクリプトphase-2-b-実装)。

### Phase 2 基盤：データ・学習パイプライン実装

`src/itm/data/` に学習基盤を整備:

- `audio.py`: 2 話者 headset 音声の同期ロード + チャンク切り出し
- `targets.py`: サバイバルラベル → torch tensor 変換 + survival NLL 損失
- `dataset.py`: `AMIDataset` (`torch.utils.data.Dataset`) + `ami_collate`

5 会議で **983 chunks** (20s chunk / 10s hop)、turn_shift 28k / backchannel 41k / overlap 71k positive frames。

ユニットテスト 23 件追加（合計 44 件、全合格）。

詳細は [学習パイプライン](../implementation/pipeline.md)。

## 2026-05-09

### Phase 1 ベースライン数値を取得（5 会議）

AMI ES2002a/b/c, IS1000a/b の 5 会議（計 165 分）で MaAI 英語 VAP を実行（`scripts/eval_maai_on_ami.py --all`）:

- **Frame VAD 精度 93.5% (pooled)** ─ MaAI の `p_now` argmax が GT と高い一致
- **Hold/shift 精度 57.2% (pooled)** ─ 528 hold + 262 shift の判定
- **会議ごとのばらつき**: 48.8% 〜 67.8%

VAP 論文の Switchboard 数値（75〜80%）より低いのは想定内（ドメイン差 + 4 人会議の 2 ch 評価 + AMI 未 fine-tune）。
これが Phase 2 で超えるべきフロア（目標 ≥ 70%）。
詳細は [MaAI ベースライン](../implementation/maai-baseline.md#phase-1-ベースライン数値ami-5-ミーティング)。

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

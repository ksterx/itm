# 変更履歴

> **Status**: stable | **Last reviewed**: 2026-05-11
>
> ドキュメント・設計の主要変更を記録する。コードの詳細は git log / GitHub Releases に任せる。

## 2026-05-11

### Phase 2-B v4 評価: Shift 専用 BCE ヘッドで初めて AUC > 0.5 獲得

`shift_head` を ITMModel に追加し、silence 直前の context から shift/hold を
二値分類する BCE 損失で学習（pos_weight=1.5、frozen transformer、3 epoch）:

- **ROC-AUC = 0.566**（v3 0.440、v2 0.487、baseline 0.701 — Phase 2-B で初の AUC > random）
- PR-AUC = 0.353（base rate 0.280 を超える）
- ただし shift 過剰予測の傾向（threshold ≤ 0.2 で全 SHIFT、0.3 で Hold 8% / Shift 94%）
- shift_pos_weight=1.5 が強すぎた可能性
- baseline 0.701 にはまだ大きな差

意義:
- survival NLL のみでは AUC ≤ 0.5 という構造的限界を突破
- 専用 discriminative head が必要なことを実証
- v5 候補: shift_pos_weight 調整、calibration、va_classifier 解凍

詳細は [学習パイプライン](../implementation/pipeline.md#phase-2-b-v4-shift-専用-bce-ヘッド)。

### v3 AUC 診断: 50% Shift は偶然だった

`scripts/diagnose_itm_hazard.py` で 12 通りの hazard 集約を試した結果:

- **v3 ROC-AUC = 0.440**（random 0.5 を下回る）— v2 も 0.487 で同様
- どの集約・horizon・event combo でも AUC ≤ 0.56
- INVERT (1 - hazard) が AUC 0.56 — 弱い負相関しか拾えていない
- Hazard sigmoid 出力が極めて狭い帯域 [0.03, 0.16] に飽和

baseline (MaAI) ROC-AUC = 0.701 と比較すると **v2/v3 は random discrimination**。
post-hoc temperature scaling は AUC 不変なので無効 → 学習目的の変更が必須 → v4。

`scripts/eval_itm_on_ami.py` と `scripts/eval_maai_on_ami.py` に `--auc` フラグを追加。

詳細は [学習パイプライン](../implementation/pipeline.md#v3--baseline-auc-診断--survival-nll-の構造的限界)。

## 2026-05-10

### Phase 2-B v3 評価: Shift 検出能力を初獲得、calibration が新たな課題

transformer 部分解凍 (lr=1e-5) + 3 epoch + pos_weight=20 + VAD aux:

- **Val loss は 3 epoch で単調減少**: 0.300 → 0.137 → 0.095
- **Shift accuracy 50.0%（baseline 44.2% を初めて上回る）** — Phase 2-B 系列で質的前進
- ただし Hold が 35.8% に落ち、Overall は 0.398 で baseline 0.586 を下回る
- しきい値 0.10 ↔ 0.11 で全反転する「all-or-nothing」現象 — hazard 分布が狭い
- Frame VAD 0.748（baseline 0.933 より低いが v1 の 0.518 崩壊は回避）

意義:
- v1: all-zero collapse、v2: hold-bias、**v3: shift-bias** の流れで初めて Shift 信号を獲得
- 残る課題は hazard 出力の calibration（散らばりが狭い）
- Overall 単独は trivial (always HOLD) = 0.720 と一致するため誤導的、AUC 等が必要

v4 候補: temperature scaling / calibration loss / score 集約変更 / AUC 評価 / Shift 専用ヘッド。

詳細は [学習パイプライン](../implementation/pipeline.md#phase-2-b-v3-transformer-部分解凍--multi-epoch)。

### Phase 2-B v2 評価: ベースラインをわずかに上回る (0.618 vs 0.586)

修正（pos_weight=50、VAD aux loss、`--freeze-transformer` で hazard heads のみ学習）後:

- **threshold=0.1 で Overall 0.618**（ベースライン 0.586 上回る）
- Hold accuracy 0.791（baseline 0.642 より改善）
- Shift accuracy 0.173（baseline 0.442 より悪化、トレードオフ）

しきい値で挙動が極端に振れる（0.05 以下で全 SHIFT、0.2 以上で全 HOLD）。
モデル出力分布が hold vs shift を強く区別できていない。

v3 で transformer 部分解凍 + multi-epoch + pos_weight 探索を実施予定。

詳細は [学習パイプライン](../implementation/pipeline.md#phase-2-b-v2-pos_weight--vad-aux--freeze_transformer)。

### Phase 2-B v1 評価: 失敗を確認、原因特定

`scripts/eval_itm_on_ami.py` で IS1000b に対する hold/shift accuracy を測定:

- **Frame VAD: 0.518**（baseline 0.933 から大幅悪化）
- **Shift: 0/52 = 0.000**（モデルが何も検出できない）
- すべての silence を HOLD と予測

原因: survival NLL のクラス不均衡 → 「全部 0 を予測」が局所最適。
追加で、VAP transformer の素朴な fine-tune により VAD 能力も喪失。

→ Phase 2-B v2 で `pos_weight`、VAD 補助損失、ヘッド先行学習で対処。

詳細は [学習パイプライン](../implementation/pipeline.md#phase-2-b-評価結果-学習は失敗した)。

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

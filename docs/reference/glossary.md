# 用語集

> **Status**: stable | **Last reviewed**: 2026-05-09
>
> このプロジェクトで使う専門用語の定義。アルファベット順。

## A

### AMI Corpus

エディンバラ大学・Idiap・Brno が共同で構築した会議データセット。100 時間、4 人参加、CC BY 4.0。詳細は [データセット](../research/datasets.md)。

### AUC (Area Under the ROC Curve)

二値分類の評価指標。閾値に依存せず、precision / recall のトレードオフを総合的に評価。

## B

### Backchannel

聞き手が出す短い相槌（「うん」「なるほど」「I see」）。発話権は移譲されない。我々のマルチイベントの 1 つ。

### Brier Score

確率予測の評価指標。$\frac{1}{N}\sum_i (p_i - y_i)^2$。校正性の評価に使う。

## C

### CANDOR Corpus

BetterUp Labs が構築した 850 時間のビデオ通話会話データセット。VAP 系研究で標準的に使われるが、個人申請の可能性は不明。

### CPC (Contrastive Predictive Coding)

van den Oord et al. (2018) による自己教師あり音声表現学習。VAP のエンコーダとして使われる。詳細は[CPC とは](../research/turn-taking-101.md)。

### Cross-Attention

Transformer の attention 機構の一種。Query を一方のソースから、Key/Value をもう一方から取る。VAP では 2 話者間の相互作用モデリングに使う。

## D

### Dialog Act

発話の意図・機能の分類。AMI では 16 種（Backchannel、Stall、Inform、Suggest 等）。詳細は [AMI Corpus](../implementation/ami-corpus.md)。

### Diarization

「誰がいつ話したか」を音声から推定する技術。話者識別 + VAD の組合せ。

### DualTurn

2026 年提案のフルデュプレックス対話モデル。Mimi コーデック + 0.5B LLM で 220ms 早期予測。arXiv:2603.08216。

## E

### ECE (Expected Calibration Error)

確率予測の校正性を測る指標。出力確率と実際の正解率のズレを bin ごとに計算。

### Edge Deployment

エッジデバイス（スマホ、Raspberry Pi、ノート PC）での実行可能性。ITM の重要な制約。

### Endpoint

発話の終了。Smart Turn の主タスク。

### Easy Turn

2025 年の音響+言語マルチモーダル turn-taking モデル。4 状態分類（complete / incomplete / backchannel / wait）。

## F

### FAU (Facial Action Unit)

顔の筋肉動作の標準的分類。AU01〜AU45 など。MM-VAP の研究で turn-taking 予測に最も寄与する視覚特徴と判明。

### Filler

「えー」「あー」「uh」「um」などの言いよどみ。Smart Turn データには midfiller / endfiller として記録されている。

## H

### Hazard Function

サバイバル分析の概念。「ある時点まで生存している条件下で、次の単位時間で事象が起きる確率」。ITM の出力定式化。

### Hold (Turn Hold)

現在の話者が発話権を継続保持すること。Turn-shift の対になる概念。

### HuBERT

Facebook の自己教師あり音声表現モデル。MM-F2F で使われている。

## I

### IPU (Inter-Pausal Unit)

ポーズで区切られた発話の単位。AMI の `segments/` で記録。

## L

### Lead Time

予測の先取り時間。「正解時刻に対してどれだけ早く予測できたか」(ms)。ITM の主要評価指標の 1 つ。

## M

### MaAI

京大・井上研の VAP 実装、`pip install maai` で利用可能。29 モデルを HF で公開。詳細は [既存モデル](../research/existing-models.md)。

### Mamba

State Space Model (SSM) ベースの系列モデル。線形時間で長系列処理可能。Coupled-Mamba (NeurIPS 2024) などで使われる。ITM v2 で検討。

### MediaPipe

Google の顔・手・姿勢検出ライブラリ。ITM の視覚特徴抽出に使う。

### MM-VAP

VAP に視覚特徴（FAU、視線、頭部姿勢）を後期融合したモデル。Inoue et al. (IEICE 2024 / arXiv:2506.03980)。精度 79% → 84%。

### Moshi

Kyutai の 7B フルデュプレックス音声テキスト基盤モデル。arXiv:2410.00037。エッジ不可。

## O

### Onset

イベントの開始時刻。turn-shift onset = 話者交代の開始時刻。

### Overlap

2 人以上が同時に発話する状態。我々のマルチイベントの 1 つ。

## P

### Proactive Prediction

事象が起きる前に予測すること。Reactive（事後検出）の対。ITM の中核アプローチ。

## Q

### QAT (Quantization-Aware Training)

量子化を考慮した学習。Smart Turn v3 は int8 static QAT で CPU 12ms 推論を実現。

## R

### rPPG (remote photoplethysmography)

顔の色の微小変化から心拍・呼吸を非接触で抽出する技術。ITM v2 で活用。

### RIIV (Respiratory-Induced Intensity Variation)

呼吸による血流変動が肌色に反映される現象。rPPG で呼吸を抽出する物理基盤。

## S

### Smart Turn

pipecat-ai の音声 turn detection モデル。BSD 2-Clause、8M params、CPU 12ms。ITM のアーキテクチャ参考。

### Survival Analysis

時間-イベントデータの統計手法。ハザード関数・サバイバル関数を扱う。ITM の出力定式化に応用。

## T

### TRP (Turn Relevant Point)

ターン交代が起こりうる時点。TurnGPT が予測する対象。

### Turn-shift

話者が交代する事象。我々のマルチイベントの中心。

### TurnGPT

Ekstedt & Skantze (EMNLP 2020 Findings) のテキストベース turn-taking 予測モデル。GPT-2 ベース。

## V

### VAD (Voice Activity Detection)

音声中の発話区間検出。Silero VAD が標準。Smart Turn は VAD + 自身の二段構成。

### VAP (Voice Activity Projection)

Ekstedt & Skantze (Interspeech 2022) の自己教師あり turn-taking モデル。**現在の最重要ベースライン**。詳細は [既存モデル](../research/existing-models.md#vap-ekstedt-skantze-interspeech-2022)。

### V-JEPA

Meta の自己教師あり映像表現モデル。V-JEPA 2 (arXiv:2506.09985)、V-JEPA 2.1 (arXiv:2603.14482)。ITM v2 で蒸留教師として検討。

### VRWE (Video-based Respiratory Waveform Estimation)

Obi & Funakoshi (ICMI 2023) が提案したタスク。顔・上半身映像から呼吸波形を回帰推定。

## 関連ページ

- [ターンテイキング 101](../research/turn-taking-101.md) — 概念の入門
- [既存モデル](../research/existing-models.md) — 各モデルの詳細
- [論文リスト](papers.md) — 参照論文

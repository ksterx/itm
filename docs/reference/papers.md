# 論文リスト

> **Status**: stable | **Last reviewed**: 2026-05-09
>
> ITM プロジェクトで参照する主要論文。すべて WebFetch で実在確認済み。

## ターンテイキング基盤

| arXiv ID | タイトル | 著者 | 発表 | 備考 |
|---|---|---|---|---|
| 2205.09812 | Voice Activity Projection: Self-supervised Learning of Turn-taking Events | Ekstedt, Skantze | Interspeech 2022 | **VAP 原典** |
| 2010.10874 | TurnGPT: a Transformer-based Language Model for Predicting Turn-taking | Ekstedt, Skantze | EMNLP 2020 Findings | テキストベース |
| 2401.04868 | Real-time and Continuous Turn-taking Prediction Using Voice Activity Projection | Inoue et al. | IWSDS 2024 | リアルタイム VAP |
| 2403.06487 | Multilingual Turn-taking Prediction Using Voice Activity Projection | Inoue et al. | LREC-COLING 2024 | 英・中・日 |
| 2410.15929 | Yeah, Un, Oh: Backchannel Prediction with Fine-tuning of VAP | Inoue et al. | NAACL 2025 | バックチャネル |
| 2401.14717 | Turn-taking and Backchannel Prediction with Acoustic and LLM Fusion | Wang et al. | ICASSP 2024 | Amazon |
| 2507.07518 | Triadic Multi-party Voice Activity Projection | Elmers et al. | Interspeech 2025 | 三者会話 |
| 2509.23938 | Easy Turn: Integrating Acoustic and Linguistic Modalities | — | 2025 | 4状態分類 |
| 2603.08216 | DualTurn: Learning Turn-Taking from Dual-Channel Generative Speech Pretraining | Shangeth Rajaa | 2026 | 220ms早期予測 |

## マルチモーダル / 視覚統合

| arXiv ID / DOI | タイトル | 著者 | 発表 | 備考 |
|---|---|---|---|---|
| 2506.03980 | Voice Activity Projection Model with Multimodal Encoders | Saga, Pelachaud | 2025 | **MM-VAP 関連** |
| 10.1587/transinf.2024HCP0002 | Multimodal Voice Activity Projection for Turn-taking | Inoue et al. | IEICE 2024 | MM-VAP 原典 |
| 2505.21043 | Visual Cues Enhance Predictive Turn-Taking | — | ACL 2025 Findings | FAU+視線+頭部 |
| 2505.12654 | MM-F2F: Predicting Turn-Taking and Backchannel | — | ACL 2025 Findings | 3モーダル融合 |
| 2505.13688 | Gaze-Enhanced Multimodal Turn-Taking Prediction in Triadic Conversations | Heo et al. | Interspeech 2025 | 視線 |
| 10.1145/3577190.3614154 | Video-based Respiratory Waveform Estimation in Dialogue (VRWE) | **Obi, Funakoshi** | **ICMI 2023** | **最重要先行** |

## エッジ / 軽量化

| arXiv ID | タイトル | 備考 |
|---|---|---|
| 2503.23439 | Speculative End-Turn Detector | 投機的二段推論 |
| — | Smart Turn v3 (pipecat-ai) | 8M, BSD-2, CPU 12ms |

## フルデュプレックス基盤モデル

| arXiv ID | タイトル | 備考 |
|---|---|---|
| 2410.00037 | Moshi: a speech-text foundation model for real-time dialogue | Kyutai, 7B, OSS |

## 関連分野（v2 で参考）

### Mamba / SSM

| arXiv ID | タイトル | 備考 |
|---|---|---|
| 2405.18014 | Coupled Mamba: Enhanced Multi-modal Fusion with Coupled SSM | NeurIPS 2024 |
| 2502.13145 | Multimodal Mamba: Decoder-only Multimodal SSM via Quadratic to Linear Distillation | — |
| 2409.12031 | PhysMamba: Efficient Remote Physiological Measurement | rPPG |
| 2503.10898 | Trajectory Mamba: Efficient Attention-Mamba Forecasting | — |
| 2504.07654 | ms-Mamba: Multi-scale Mamba for Time-Series | — |

### V-JEPA / 視覚表現

| arXiv ID | タイトル | 備考 |
|---|---|---|
| 2506.09985 | V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning | Meta |
| 2603.14482 | V-JEPA 2.1: Unlocking Dense Features in Video SSL | — |
| 2506.03885 | Video, How Do Your Tokens Merge? | token merging |

### 早期行動認識・意図予測

| arXiv ID | タイトル | 備考 |
|---|---|---|
| 2410.14045 | Human Action Anticipation: A Survey | — |
| 2507.07734 | EEvAct: Early Event-Based Action Recognition | spiking NN |
| 2507.13425 | CaTFormer: Causal Temporal Transformer for Driving Intention | ※元の引用は CaSTFormer だが正式名は CaTFormer |
| 2510.09200 | Towards Safer and Understandable Driver Intention Prediction | DAAD-X 略称はアブスト未確認 |
| 2603.19533 | Pedestrian Crossing Intent Prediction via Psychological Features | — |
| 2603.10061 | Decision-Aware Uncertainty Evaluation of VLM-Based Early Action Anticipation for HRI | — |
| 2604.05843 | EEG-MFTNet: Cross-Session Motor Imagery Decoding | EEG |

## 呼吸研究

| 出典 | タイトル | 備考 |
|---|---|---|
| Interspeech 2016 | Respiratory Turn-Taking Cues | Włodarczak, Heldner |
| Sci Rep 2025 (s41598-025-15776-1) | CNS control of breathing in turn-taking | fMRI 200ms 先行 |
| 2006.03790 | MTTS-CAN: On-Device Vitals | rPPG, モバイル |
| 2111.12082 | PhysFormer | rPPG, CVPR 2022 |

## ストリーミング・リアルタイム音声

| arXiv ID | タイトル | 備考 |
|---|---|---|
| 2510.00982 | Spiralformer: Low Latency Encoder for Streaming Speech | — |
| 2504.02302 | Causal Self-supervised Pretrained Frontend with Predictive Code | speech separation |
| 2503.04721 | Full-Duplex-Bench | 評価ベンチマーク |
| 2509.14515 | From Turn-Taking to Synchronous Dialogue: Full-Duplex Survey | — |

## サーベイ

| 出典 | タイトル | 備考 |
|---|---|---|
| ACL Anthology 2025.iwsds-1.27 | A Survey of Recent Advances on Turn-taking Modeling | Castillo-López et al. |
| Computer Speech & Language 2021 | Turn-taking in conversational systems and HRI: a review | Skantze |

## 引用上の注意

- **2506.03980 の "MM-VAP" 略称**: 論文本文での略称か要確認、引用時は本文確認推奨
- **2507.13425**: 元の引用名 "CaSTFormer" は誤り、正式名は "CaTFormer"
- **2510.09200 の "DAAD-X" 略称**: アブストでは確認できず

## 関連ページ

- [調査の概観](../research/overview.md) — 論文を踏まえた全体整理
- [既存モデル](../research/existing-models.md) — 主要モデルの詳細
- [関連研究](../research/related-work.md) — 特に Obi & Funakoshi シリーズ

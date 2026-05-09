# 関連研究

> **Status**: stable | **Last reviewed**: 2026-05-09
>
> ITM に最も関連の深い先行研究を、特に Obi & Funakoshi シリーズを中心に整理。

## 最重要先行: Obi & Funakoshi シリーズ

東京工業大学（現 Institute of Science Tokyo, ISCT）船越研による一連の研究。**我々が「顔→呼吸→ターンテイキング」と呼んでいた研究方向を、すでに体系的に進めている**。

### Obi & Funakoshi (ICMI 2023)

- **タイトル**: "Video-based Respiratory Waveform Estimation in Dialogue: A Novel Task and Dataset for Human-Machine Interaction"
- **DOI**: 10.1145/3577190.3614154
- **ベニュー**: ICMI 2023, Paris

#### 貢献

1. **VRWE タスク定義** — 対話中の RGB 映像から呼吸波形を回帰推定する新タスク。amplitude estimation と gradient estimation の 2 サブタスクに分解
2. **VREi データセット** — 30 人（80 人中の subset）、日本語、安静セッション 20 分 + 対話セッション 15 分、胸郭+腹部の 2 本の呼吸ベルトと同期
3. **3DCNN-ConvLSTM ベースライン** — 入力 10 frames × 256×256 RGB
4. **下流有用性の実証** — VRWE 出力（特に gradient）が **voice activity の 200ms 先行予測** に有効

#### ITM との関係

我々が「顔のみから呼吸を取って 200ms 先取りでターンテイキング予測」と言っていた構想の **本丸** を既に実証している。"世界初の視覚→呼吸→ターンテイキング" を主張するのは不可能。

### Obi & Funakoshi (HRI 2024 Companion, LBR)

- "Respiration-enhanced Human-Robot Communication"
- pp. 813-816
- ロボット応用の概念実証

### Obi & Funakoshi (HAI 2024, poster)

- "Can Respiration Make Spoken Interactions Better?"
- pp. 423-425
- 仮説提示

### Obi & Funakoshi (SIGDIAL 2024 Demo)

- "Using Respiration for Enhancing Human-Robot Dialogue"
- pp. 325-328
- VRWE 動画推定をロボット対話システムに統合
- speech collision 回避と pseudo-respiration 提示

### Obi & Funakoshi (IWSDS 2025)

- **タイトル**: "Integrating Respiration into Voice Activity Projection for Enhancing Turn-taking Performance"
- **VAP モデルに呼吸を統合**し、audio-only より性能向上を実証
- ただし呼吸は **接触型ベルト計測**（VRWE 動画推定ではない可能性が高い）

### Obi & Funakoshi (IEEE RA-L 2025)

- Vol 10(9), pp. 9581-9588
- "Breathe and Speak Attentively: Implementing Respiratory Awareness Into Conversational Robots"
- 26 名で SCA + 同期 PRP の効果検証

## ITM のポジショニング修正

Obi & Funakoshi シリーズを踏まえると、我々の差分は以下に絞られる:

| 軸 | Obi & Funakoshi シリーズ | ITM |
|---|---|---|
| 呼吸の取得 | 接触型ベルト (IWSDS 2025) または直接 3DCNN 推定 (ICMI 2023) | **rPPG / 顔 micro-motion / 鼻孔の派生信号** から |
| イベント粒度 | VAP の二値 voice activity | **マルチイベント** (turn-shift / backchannel / overlap) |
| エッジ実装 | 議論なし | エッジ最適化（< 10M params） |
| 言語 | 日本語 | **英語**（公開データの制約から） |

主張のフレーズ案:

> Obi & Funakoshi (ICMI 2023) は対話中の RGB 映像から呼吸波形を回帰推定する VRWE タスクを定式化し、推定した呼吸（特に gradient）が voice activity の 200ms 先行予測に有効であることを示した。本研究はこの**単一モダリティ・連続値回帰・二値 voice activity** という設計を出発点とし、(i) ターンテイキングの **多クラスイベント** (turn-shift / backchannel / overlap) への拡張、(ii) **音声との multimodal fusion**、(iii) 生 RGB 直接学習に代わる **rPPG ベースの呼吸表現**、(iv) **エッジ実行**、の 4 点で差分化する。

## その他の関連研究

### Włodarczak & Heldner (Interspeech 2016)

- "Respiratory Turn-Taking Cues" pp. 1275-1279
- 呼吸ローカルマキシマと turn-taking の関連を体系化
- 失敗した割り込みは preparatory inhalation を欠くことを発見
- **吸気深度・吸気持続時間・呼吸 range が予測子**として有意

### Di Pasquasio et al. (Sci Rep 2025)

- doi:10.1038/s41598-025-15776-1
- **fMRI** で自然会話を解析
- 呼吸 200ms 先行を **脳活動レベル** で実証
- 前運動皮質・補足運動野の活動を確認
- bioRxiv 2024.07.17.603521 が prior

### Skantze (2021)

- "Turn-taking in conversational systems and human-robot interaction: a review"
- *Computer Speech & Language*
- ターンテイキング研究のレビュー論文、200ms ギャップの普遍性を整理

### Castillo-López et al. (IWSDS 2025)

- "A Survey of Recent Advances on Turn-taking Modeling in Spoken Dialogue Systems"
- ACL Anthology: 2025.iwsds-1.27
- レビューした研究の **72% が先行研究と比較していない** ことを指摘
- 統一ベンチマークの欠如を問題視

## 関連ページ

- [視覚シグナル](visual-cues.md) — 呼吸シグナルの取得経路
- [新規性](../design/novelty.md) — 既存研究との差別化の最終整理
- [論文リスト](../reference/papers.md) — 実在確認済み arXiv ID

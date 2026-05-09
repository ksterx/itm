# 解く問題

> **Status**: draft | **Last reviewed**: 2026-05-09
>
> ITM が解こうとしている問題の精密な定義。

## 入力

時刻 $t$ における、過去 $T$ 秒分の同期入力ストリーム:

- **音声**: 16kHz、二者会話の各話者チャンネル分離（または mixed + diarization）
- **映像**: 30fps、顔 ROI（上半身は任意）

## 出力

各時刻 $t$ で、複数のターンテイキング・イベントについて **将来 2 秒以内のハザード関数** を出力する:

$$
h_e(t, k) = P\bigl(\text{event } e \text{ occurs in } [t + k\Delta,\ t + (k+1)\Delta) \mid x_{\le t}\bigr)
$$

- $e \in \{\text{turn-shift},\ \text{backchannel},\ \text{overlap}\}$
- $\Delta = 50\,\text{ms}$、$k \in \{0, 1, \ldots, 39\}$（2 秒先まで）

詳細は [マルチイベント・ハザード](../design/multi-event-hazard.md) を参照。

## イベント定義

| イベント | 定義 |
|---|---|
| **Turn-shift** | 相手が **発話権を取って** 持続的に話し始める |
| **Backchannel** | 相手が短い相槌（「うん」「なるほど」）を入れる、発話権は移譲されない |
| **Overlap** | 自分の発話継続中に相手が割り込み発話を始める |

未起こしの定義はラベル生成方針（[ラベル生成](../design/label-generation.md)）で精緻化する。

## 評価

| 指標 | 内容 |
|---|---|
| **Per-event Hazard AUC** | 各 horizon $k$ での AUC |
| **Lead Time @ FPR=5%** | 誤検出 5% 時の平均先取り時間 (ms) |
| **Confusion Matrix** | turn-shift / backchannel / hold 間の混同行列 |
| **Brier Score / ECE** | 確率の校正性 |
| **CPU Inference Latency** | M4 / x86_64 でのフレーム処理時間 |

## 制約

- **計算資源**: NVIDIA A100 × 48h（学習総計）
- **モデルサイズ**: < 10M params 目標
- **推論**: CPU リアルタイム（フレーム処理 < 100ms @ 10Hz）
- **データ**: アカデミック署名・購入なしで入手可能なデータセットのみ

## 関連ページ

- [プロジェクト概要](motivation.md) — なぜこの問題か
- [v1 アーキテクチャ](../design/architecture.md) — どう解くか
- [評価指標の詳細](../design/multi-event-hazard.md) — ハザード形式の選択理由

# ロードマップ

> **Status**: stable | **Last reviewed**: 2026-05-09
>
> Phase 別の作業計画。状況により再評価する。

## v1: HuggingFace 先行リリース

論文より先に「動くものを公開する」段取り。

### Phase 0 — 環境構築・基礎調査 ✅完了

期間: 1 日

- [x] Python 3.11 venv + maai インストール
- [x] MaAI で英語 VAP の推論動作確認
- [x] AMI Corpus 注釈構造解明（dialog act ontology 16種）
- [x] Smart Turn データの実体把握
- [x] プロジェクト構造（pyproject.toml, src/, scripts/）

成果は [Phase 0 メモリ](https://github.com/ksterx/itm/blob/main/docs/implementation/environment.md) を参照。

### Phase 1 — ベースライン再現

期間: 約 1 週間

- [ ] AMI Corpus 部分ダウンロード（5 ミーティング、~3GB）
- [ ] AMI XML 注釈パーサ実装（dialogueActs + segments + words の時間整合）
- [ ] MaAI VAP モデルを AMI で推論
- [ ] VAP の標準指標（hold/shift accuracy）を再現

### Phase 2 — マルチイベント拡張

期間: 約 2 週間

- [x] AMI dialog act → ITM event のラベル変換実装（[ラベル生成](../design/label-generation.md)）
- [x] データ層: AMIDataset + survival NLL 損失
- [x] サバイバルハザード head の実装（`itm.models.ITMModel`）
- [x] 学習スクリプト + sanity smoke test（loss 単調減少）
- [x] Phase 2-B v1: 4+1 split で fine-tune → **失敗を確認**（Shift 0%、VAD 崩壊）
- [x] Phase 2-B v2: pos_weight=50 + VAD aux + frozen transformer → **ベースライン微改善**（0.618 vs 0.586）
- [ ] **Phase 2-B v3**: transformer 部分解凍 + multi-epoch + pos_weight 探索
- [ ] 既存 VAP_BC、VAP_Nod との比較

### Phase 3 — 視覚追加

期間: 約 1 週間

- [ ] MediaPipe 顔特徴抽出（FAU、頭部姿勢、視線、口開度）
- [ ] 後期融合 (late fusion) のクロスアテンション
- [ ] AMI 映像で fine-tune
- [ ] アブレーション（音声のみ vs 音声+視覚）

### Phase 4 — 量子化・デプロイ

期間: 約 1 週間

- [ ] ONNX export
- [ ] int8 static QAT (Smart Turn の方式踏襲)
- [ ] CPU レイテンシ計測（M4、x86_64）

### Phase 5 — HuggingFace 公開

期間: 数日

- [ ] モデルカード作成
- [ ] Gradio Space デモ
- [ ] GitHub README 整備
- [ ] アナウンス（Twitter / Reddit / HF Discord）

## v2: 研究強化 + 論文

### Phase 6 — rPPG 呼吸統合

期間: 1〜2 ヶ月

- [ ] rPPG-Toolbox の PhysMamba/EfficientPhys を呼吸エンコーダとして統合
- [ ] アブレーション（呼吸あり/なし）
- [ ] [視覚シグナル](../research/visual-cues.md) で議論した3経路統合

### Phase 7 — V-JEPA 蒸留

期間: 数週間

- [ ] V-JEPA 2 dense features を顔エンコーダに蒸留
- [ ] エッジサイズ維持

### Phase 8 — Coupled-Mamba 融合

期間: 数週間

- [ ] 後期融合を Coupled-Mamba (NeurIPS 2024) に置換
- [ ] アブレーション

### Phase 9 — 論文執筆

期間: 1〜2 ヶ月

- [ ] arXiv プレプリント
- [ ] ICMI / Interspeech / IWSDS のいずれかに投稿

## 死んだ計画（参考）

過去に検討して採用しなかったもの:

- 日本語 CEJC / NoXi+J / Hazumi をメインデータに → **アカデミック署名/購入が必須で断念**、英語 AMI に変更
- ErikEkstedt/VoiceActivityProjection を直接 fork → **依存劣化、vap_dataset が private** で MaAI に変更
- Smart Turn データで turn-taking 学習 → **データの実体は単一話者 endpoint** で AMI に変更
- 呼吸を上半身 optical flow から取る → **顔のみシナリオで動かない** ので顔由来 (rPPG, micro-motion) に変更

## 関連ページ

- [プロジェクト概要](motivation.md) — 全体像
- [v1 アーキテクチャ](../design/architecture.md) — Phase 1〜5 の技術詳細

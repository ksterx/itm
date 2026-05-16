# AMI Corpus

> **Status**: stable | **Last reviewed**: 2026-05-09
>
> AMI Corpus のダウンロード手順とディレクトリ構造、ITM での利用方法。

## 概要

- 100 時間の会議録、4 人参加、英語
- ライセンス: **CC BY 4.0**（個人 OK、商用 OK、登録不要）
- 公式: https://groups.inf.ed.ac.uk/ami/corpus/
- ITM のメインデータソース

詳細は [データセット](../research/datasets.md) を参照。

## ダウンロード手順

### 注釈のみ（22.9MB、まず最初に取る）

```bash
python scripts/download_ami_subset.py --annotations-only
```

これで `data/raw/ami/annotations/ami_public_manual_1.6.2.zip` が取れる。

### 5 ミーティング、音声のみ（~640MB、Phase 1 の最小構成）

```bash
python scripts/download_ami_subset.py --audio-only
```

### 5 ミーティング、音声+映像（~3GB、Phase 3 で必要）

```bash
python scripts/download_ami_subset.py
```

### 任意のミーティング指定

```bash
python scripts/download_ami_subset.py --meetings ES2002a IS1000a TS3003a
```

## URL パターン

スクリプトの内部仕様:

```
Audio (per-speaker, 4 ch):
  https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<MID>/audio/<MID>.Headset-{0,1,2,3}.wav

Audio (mixed):
  https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<MID>/audio/<MID>.Mix-Headset.wav

Video (per-speaker close-up, 4 cam):
  https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<MID>/video/<MID>.Closeup{1,2,3,4}.avi

Video (overview):
  https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<MID>/video/<MID>.Corner.avi

Annotations:
  https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
```

## ストレージ配置（2026-05-16 以降）

リポジトリ直下の `data/raw/ami` は **外付け SSD (CT1000P3) へのシンボリックリンク**:

```
data/raw/ami  ──symlink──>  /Volumes/CT1000P3/datasets/turn-taking/ami/
```

理由: AMI 全 100h 取得時に内蔵 SSD (228GB, 空き ~86GB) では余裕が少なく、外付け 1TB SSD (空き 826GB) に逃がすため。コード側の `ANNOT_ROOT` / `AUDIO_ROOT` (`data/raw/ami/...`) はそのまま使える。

**注意**:
- 学習中は CT1000P3 をマウント解除しない
- 再現する場合: 外付けが無ければ `mkdir -p data/raw/ami` でリポ内にそのまま展開して OK

## ディレクトリ構造（DL 後）

```
data/raw/ami/   (symlink 先 = /Volumes/CT1000P3/datasets/turn-taking/ami/)
├── annotations/
│   ├── ami_public_manual_1.6.2.zip
│   └── unpacked/
│       ├── 00README_MANUAL.txt
│       ├── LICENCE.txt
│       ├── dialogueActs/        # 16 種の対話行為タグ
│       ├── disfluency/
│       ├── segments/            # IPU/発話単位の境界
│       ├── words/               # 単語レベル時間整合
│       ├── headGesture/
│       ├── handGesture/
│       ├── movement/
│       ├── focus/               # 注視
│       ├── topics/
│       ├── words/
│       └── ontologies/
│           └── da-types.xml     # dialog act 階層
├── ES2002a/
│   ├── audio/
│   │   ├── ES2002a.Headset-0.wav  # 各 ~40MB
│   │   ├── ES2002a.Headset-1.wav
│   │   ├── ES2002a.Headset-2.wav
│   │   └── ES2002a.Headset-3.wav
│   └── video/
│       ├── ES2002a.Closeup1.avi   # 40〜60MB
│       ├── ES2002a.Closeup2.avi
│       ├── ES2002a.Closeup3.avi
│       └── ES2002a.Closeup4.avi
└── ...
```

## 注釈の中身（NXT 形式）

XML ベースの **standoff annotation**: 各注釈ファイルが `nite:child href="..."` で他ファイルを参照。

### words.xml の例

```xml
<w nite:id="ES2002a.A.words0" starttime="0.34" endtime="0.52">Hello</w>
<w nite:id="ES2002a.A.words1" starttime="0.53" endtime="0.65">everyone</w>
```

### segments.xml の例

```xml
<segment nite:id="ES2002a.A.seg.1" transcriber="..." channel="A">
  <nite:child href="ES2002a.A.words.xml#id(ES2002a.A.words0)..id(ES2002a.A.words12)"/>
</segment>
```

### dialog-act.xml の例

```xml
<dact nite:id="ES2002a.A.dialog-act.dharshi.1">
  <nite:pointer role="da-aspect" href="da-types.xml#id(ami_da_4)"/>  <!-- inf -->
  <nite:child href="ES2002a.A.words.xml#id(ES2002a.A.words0)..id(ES2002a.A.words12)"/>
</dact>
```

ここで `ami_da_4` を `ontologies/da-types.xml` で引くと:

```xml
<da-type nite:id="ami_da_4" name="inf" gloss="Inform"/>
```

ID対応表:

| ID | name | gloss |
|---|---|---|
| ami_da_1 | bck | Backchannel |
| ami_da_2 | stl | Stall |
| ami_da_3 | fra | Fragment |
| ami_da_4 | inf | Inform |
| ami_da_5 | el.inf | Elicit-Inform |
| ami_da_6 | sug | Suggest |
| ami_da_7 | off | Offer |
| ami_da_8 | el.sug | Elicit-Offer-Or-Suggestion |
| ami_da_9 | ass | Assess |
| ami_da_11 | el.ass | Elicit-Assessment |
| ami_da_12 | und | Comment-About-Understanding |
| ami_da_13 | el.und | Elicit-Comment-Understanding |
| ami_da_14 | be.pos | Be-Positive |
| ami_da_15 | be.neg | Be-Negative |
| ami_da_16 | oth | Other |

## ITM のラベル生成

XML パーサで以下を抽出:

1. 各話者のセグメント列（時間境界 + dialog act ID）
2. 異話者間の重なり時刻
3. dialog act ID から ITM event への変換

詳細実装は [ラベル生成](../design/label-generation.md)。

## サブセット選定

Phase 1 のおすすめ 5 ミーティング:

```python
DEFAULT_MEETINGS = [
    "ES2002a", "ES2002b", "ES2002c",  # Edinburgh group 1
    "IS1000a", "IS1000b",             # Idiap group 1
]
```

これで:

- 異なる collection（Edinburgh / Idiap）でドメイン汎化を見られる
- 同じグループの連続セッション（a, b, c）で speaker overfitting を検証できる
- 合計 ~3GB（音声+映像）で扱いやすい

## ライセンス

CC BY 4.0 — クレジット表記すれば自由に使える。論文では:

> This work uses the AMI Meeting Corpus (Carletta et al., 2005), released under CC BY 4.0.

## 関連ページ

- [データセット](../research/datasets.md) — 全データセットの詳細
- [ラベル生成](../design/label-generation.md) — XML から ITM ラベルへ
- [v1 アーキテクチャ](../design/architecture.md) — どう使うか

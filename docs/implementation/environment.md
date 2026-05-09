# 環境構築

> **Status**: stable | **Last reviewed**: 2026-05-09
>
> ITM プロジェクトの開発環境セットアップ手順と Phase 0 の検証結果。

## 前提

- macOS (Apple Silicon) または Linux
- Python 3.11（3.12+ は Smart Turn 等の依存と未検証）
- uv パッケージマネージャ
- 16GB RAM 以上推奨
- ディスク 10GB 以上の空き

## 1. システム依存

### macOS

```bash
brew install portaudio  # PyAudio が要求
```

### Linux (Ubuntu/Debian)

```bash
sudo apt-get install -y portaudio19-dev python3-dev
```

## 2. Python 環境

```bash
# uv が未インストールなら
curl -LsSf https://astral.sh/uv/install.sh | sh

# プロジェクトルートで
uv venv -p 3.11 .venv
source .venv/bin/activate
uv pip install -e .
```

`pyproject.toml` の依存:

- `maai>=0.1.16`: VAP ベースライン
- `torch>=2.6`, `torchaudio>=2.6`
- `numpy`, `soundfile`, `pydantic`, `anyio`
- `huggingface-hub`, `datasets`

オプショナル (`uv pip install -e .[video]`):

- `mediapipe>=0.10`: 顔ランドマーク抽出
- `opencv-python>=4.10`

開発ツール (`uv pip install -e .[dev]`):

- `ruff`, `pytest`, `ipython`

## 3. 動作確認

### MaAI のスモークテスト

```bash
python scripts/test_maai_inference.py
```

期待される出力:

```
Loading MaAI English VAP (10Hz, 5s context)...
  loaded in 14.21s
Running inference for 12s wall clock (audio is 10s)...
Got ~100 result frames in 12.20s (rate ≈ 8.2/s)
Result type: dict
Result keys: ['t', 'x1', 'x2', 'p_now', 'p_future', 'vad']
```

このスクリプトは:

1. 10 秒の合成音声（speaker 1 のみアクティブ、speaker 2 は無音）を生成
2. MaAI の英語 VAP (10Hz, 5s context) をロード
3. 推論を回して結果を確認

`p_now=[0.66, 0.34]` のような出力が得られれば正常。

## 4. プロジェクト構造

```
itm/
├── pyproject.toml          # uv 管理、BSD-2-Clause
├── README.md
├── LICENSE
├── mkdocs.yml              # ドキュメント設定
├── src/itm/                # メインパッケージ
├── scripts/                # 一回限りのスクリプト・スモークテスト
│   ├── test_maai_inference.py
│   ├── inspect_smart_turn_data.py
│   └── download_ami_subset.py
├── configs/                # 学習・推論設定
├── data/raw/               # 元データ (gitignore)
├── data/processed/         # 前処理済み (gitignore)
├── checkpoints/            # 学習済み重み (gitignore)
├── tmp/                    # 一時ファイル (gitignore)
├── docs/                   # MkDocs ソース
├── site/                   # MkDocs ビルド結果 (gitignore)
└── .github/workflows/      # CI/CD
```

## 5. ドキュメントのローカル実行

```bash
uv pip install -e ".[docs]"  # mkdocs-material 等
mkdocs serve  # http://localhost:8000
```

編集中は `mkdocs serve` のホットリロードで確認。

## Phase 0 検証結果

著者の環境（M4 MacBook Air, 16GB）で確認した動作:

| 項目 | 結果 |
|---|---|
| `pip install maai` | 成功（Python 3.11.15、torch 2.11） |
| MaAI モデルロード | 14.2 秒 |
| MaAI 推論 (10s 音声) | 12.2 秒, 8.2 frame/s |
| 出力構造 | `t, x1, x2, p_now, p_future, vad` ✅ |

## 既知の問題

### Python 3.14 で動かない

uv のデフォルト Python は 3.14 だが、これでは `maai` の依存（pyaudio 等）に問題あり。明示的に 3.11 を指定する:

```bash
uv venv -p 3.11 .venv
```

### portaudio.h not found

macOS で:

```
src/pyaudio/device_api.c:9:10: fatal error: 'portaudio.h' file not found
```

→ `brew install portaudio` で解決

### maai の get_result() がブロックする

短い WAV ファイル（< context_len_sec）を入力すると、`maai.get_result()` が永久にブロックする。対処法:

```python
# get_result() 直接ではなく、内部キューにタイムアウト付きで取りに行く
result_q = maai.result_dict_queue
r = result_q.get(timeout=0.5)
```

## 関連ページ

- [MaAI ベースライン](maai-baseline.md) — 推論の詳細
- [AMI Corpus](ami-corpus.md) — データセット取得

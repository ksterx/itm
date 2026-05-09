# It's My Turn (ITM)

[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue)](https://ksterx.github.io/itm/)
[![License](https://img.shields.io/badge/license-BSD--2--Clause-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-Phase%200%20complete-orange)](docs/about/roadmap.md)

> Multimodal proactive turn-taking prediction for real-time conversational AI.

ITM predicts — from synchronized audio + video — **whether and how** the conversation
partner is about to speak, **before vocalization begins**. Outputs per-event hazards
for **turn-shift**, **backchannel**, and **overlap**, designed for edge-device
real-time inference.

## Why this project

Modern voice AI is bad at turn-taking: it waits 700–1000ms of silence to decide
"they're done speaking", interrupts you when you say "uh", and can't tell apart
"yeah" (continue) from "well, actually..." (yield the floor).

Humans solve this in ~200ms by reading **pre-vocal cues** — breath, lip parting,
posture, gaze. ITM learns to do the same, with three independent hazard heads
that downstream apps can mix.

📖 **[Read the docs →](https://ksterx.github.io/itm/)**

## Quickstart

```bash
git clone https://github.com/ksterx/itm.git
cd itm

# System deps (macOS)
brew install portaudio

# Python env
uv venv -p 3.11 .venv && source .venv/bin/activate
uv pip install -e .

# Smoke test (downloads MaAI English VAP weights from HF)
python scripts/test_maai_inference.py
```

For the full setup including AMI Corpus and dataset preparation, see
[Environment](https://ksterx.github.io/itm/implementation/environment/) and
[AMI Corpus](https://ksterx.github.io/itm/implementation/ami-corpus/).

## Project status

🚧 **Phase 0 complete** — environment bring-up and baseline reproduction.

| Phase | Status | Goal |
|---|---|---|
| 0. Environment | ✅ | venv, MaAI smoke test, AMI structure understood |
| 1. Baseline reproduction | 🔜 | Run MaAI VAP on AMI, reproduce hold/shift accuracy |
| 2. Multi-event extension | ⏳ | 3 hazard heads (turn / bc / overlap), AMI fine-tune |
| 3. Visual fusion | ⏳ | MediaPipe features + late fusion |
| 4. Edge optimization | ⏳ | ONNX export + int8 QAT + CPU latency benchmark |
| 5. HuggingFace release | ⏳ | Model card + Gradio Space |

See [Roadmap](https://ksterx.github.io/itm/about/roadmap/) for details.

## Architecture overview

```
                         ┌─────────────────────────────┐
   Audio 16kHz (×2 ch) ──┤ CPC Encoder (frozen)        ├─┐
                         │ + Self/Cross-Attention      │ │
                         └─────────────────────────────┘ │
                                                         ├──> Late Fusion ──> Attention Pool
   Video 30fps  ─> MediaPipe ─> Visual MLP (light)       │                         │
                                                         ┘                         ▼
                                          ┌─────────────────────────────────────────┐
                                          │  3 × Hazard Heads (40 bins × 50ms)      │
                                          │  - turn-shift hazard                    │
                                          │  - backchannel hazard                   │
                                          │  - overlap hazard                       │
                                          │  + Aux: VAD, filler                     │
                                          └─────────────────────────────────────────┘
```

Target: **~12M parameters**, CPU real-time (< 100ms / frame @ 10Hz).

📐 [Detailed design →](https://ksterx.github.io/itm/design/architecture/)

## Building blocks (we stand on)

- **[MaAI](https://github.com/maai-kyoto/maai)** — VAP / VAP-BC / VAP-Nod base implementation (29 HF models)
- **[Smart Turn v3](https://github.com/pipecat-ai/smart-turn)** — reference for edge architecture and quantization
- **[AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)** — main training data (CC BY 4.0, 100h, 4-person meetings, video+audio+annotations)
- **[MediaPipe](https://github.com/google-ai-edge/mediapipe)** — face landmarks for visual fusion

📚 [Full reference →](https://ksterx.github.io/itm/reference/resources/)

## Documentation

| Section | Contents |
|---|---|
| [About](https://ksterx.github.io/itm/about/motivation/) | Motivation, problem definition, roadmap |
| [Research](https://ksterx.github.io/itm/research/overview/) | Survey: turn-taking 101, existing models, visual cues, datasets |
| [Design](https://ksterx.github.io/itm/design/architecture/) | v1 architecture, multi-event hazard, label generation, novelty |
| [Implementation](https://ksterx.github.io/itm/implementation/environment/) | Environment, MaAI baseline, AMI, pipeline |
| [Reference](https://ksterx.github.io/itm/reference/glossary/) | Glossary, papers, resources |

## Contributing

Issues, discussions, and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

- **Code**: BSD 2-Clause (see [LICENSE](LICENSE))
- **Documentation**: CC BY 4.0
- **Pretrained weights** (when released): TBD — likely BSD 2-Clause if trained from scratch, otherwise inherits upstream license (e.g., MaAI weights are academic-only)

## Acknowledgments

This project builds on:
- VAP by Ekstedt & Skantze (Interspeech 2022)
- MaAI by Inoue, Lala, Kawahara et al. (Kyoto University)
- Smart Turn by pipecat-ai team
- AMI Meeting Corpus by Carletta et al.
- Obi & Funakoshi's pioneering work on respiratory-aware turn-taking

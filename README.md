# sc-soundmatch-agent

Iterative sound matching: an LLM agent writes SuperCollider synthesis code, renders audio, and compares it to a target — guided by FluCoMa analysis and spectral metrics.

## How it works

```text
target.wav
    │
    ├─ evaluate.py          → spectral metrics + categories
    ├─ analyze_partials.py  → FluCoMa partials + SC templates A–E
    │
    └─ OpenClaw agent loop (workspace/AGENTS.md)
           write attempt_N.scd
           pre_validate.py   → static SC checks (classes, envelopes)
           wrap_for_recording.py → NRT wrapper + SynthDef validation
           sclang              → attempt_N.wav
           compare.py          → composite_score + correction prompt
           repeat until converged or max iterations
```

Each run creates a timestamped directory under `runs/` with all attempts, comparisons, and a `final_result.scd`.

## Reference examples

The files in `workspace/reference_examples/` are derived from [SuperCollider documentation on doc.sccode.org](https://doc.sccode.org/). They were scraped from sccode.org examples, rendered and analyzed, then packaged as `.ref` files containing:

- An audio description and spectral categories
- Synthesis concepts (oscillator, filter, envelope hints)
- The original SuperCollider code

`workspace/reference_examples/index.json` maps category combinations to example filenames. The agent uses these in Step 0 to find SC patterns similar to the target before writing its first synthesis attempt.

SuperCollider example code and documentation © respective authors; sourced from [doc.sccode.org](https://doc.sccode.org/).

To rebuild the reference index from source batches, use `build_reference_index.py`.

## Requirements

| Component | Purpose |
|---|---|
| [SuperCollider](https://supercollider.github.io/) (`sclang`) | SynthDef validation and NRT audio rendering |
| [FluCoMa CLI](https://github.com/flucoma/flucoma-cli) | Sinusoidal decomposition and partial analysis |
| Python 3.10+ | Evaluation, comparison, wrapping, pre-validation |
| [OpenClaw](https://github.com/openclaw/openclaw) | Agent runtime (exec/read/write tools) |
| Ollama (or other LLM provider) | Local model for the synthesis agent |

### Python packages

```bash
pip install numpy scipy librosa soundfile
```

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/ayk-caglayan/sc-soundmatch-agent.git
   cd sc-soundmatch-agent
   ```

2. **Install external tools** — SuperCollider, FluCoMa CLI, OpenClaw, and Ollama must be on your `PATH` (or configured in the scripts below).

3. **Configure paths** — several scripts contain machine-specific paths that you may need to adjust:

   | File | Setting |
   |---|---|
   | `analyze_partials.py` | `FLUCOMA_BIN` — path to FluCoMa CLI binaries |
   | `launcher.sh` | Python interpreter path (`miniconda3/bin/python3`) |
   | `workspace/AGENTS.md` | Absolute paths to project scripts in `exec` commands |
   | `build_reference_index.py` | Source directories for reference example rebuild |

4. **Register the OpenClaw agent** — create an agent (e.g. `sc_synth_flucoma`) in `~/.openclaw/openclaw.json` pointing its workspace to this repo's `workspace/` directory. Do **not** commit `~/.openclaw/openclaw.json`; it contains API keys and tokens.

5. **Configure Ollama** — register your model in OpenClaw's provider config. For local models, set `timeoutSeconds` high enough for slow inference (e.g. 900) and choose an appropriate `num_ctx` (e.g. 65536 if the agent reads the full reference index).

## Usage

```bash
./launcher.sh --target /path/to/audio.wav \
              --max-iter 20 \
              --threshold 0.4 \
              --model ollama/qwen3.6:latest
```

| Flag | Default | Description |
|---|---|---|
| `--target` | *(required)* | Path to target WAV file |
| `--max-iter` | 85 | Maximum refinement iterations |
| `--threshold` | 0.4 | Convergence goal (`composite_score` below this) |
| `--model` | `ollama/qwen3-coder-next:latest` | OpenClaw model id |
| `--no-telegram` | off | Disable Telegram progress notifications |

Output lands in `runs/YYYYMMDD_HHMMSS_<basename>/`:

- `target_eval.txt`, `target_partials.txt` — pre-computed analysis
- `attempt_N.scd`, `attempt_N.wav` — synthesis attempts
- `comparison_N.txt` — metrics, category mismatches, correction prompt
- `final_result.scd`, `report.md` — best result and summary

For quick debugging, trim the target to a few seconds first:

```bash
ffmpeg -i long_clip.wav -t 3 -y short_clip.wav
./launcher.sh --target short_clip.wav --max-iter 5 --model ollama/qwen3.6:latest
```

## Project structure

```text
launcher.sh                 Run orchestrator
analyze_partials.py         FluCoMa analysis + SC templates
evaluate.py                 Target spectral evaluation
compare.py                  Attempt vs target comparison + correction prompt
pre_validate.py             Fast static SC code checks
wrap_for_recording.py       NRT wrapper + SynthDef validation
synthesis_evaluator_fixed.py Spectral metric engine
build_reference_index.py    Rebuild reference_examples/ from source
sc_classes.txt              SuperCollider class index (3473 classes)
sc_ugen_signatures.json     UGen parameter signature database
config.yaml                 Example OpenClaw agent config
workspace/
  AGENTS.md                 Agent protocol (read by OpenClaw)
  reference_examples/       Scraped sccode.org examples + index.json
runs/                       Runtime output (gitignored)
```

## Validation layers

Generated SuperCollider code passes through three validation layers before rendering:

1. **pre_validate.py** — `var` placement, undeclared variables, unknown classes, envelope timing rules
2. **wrap_for_recording.py Layer 1/1b** — class name and UGen parameter auto-correction
3. **wrap_for_recording.py Layer 2** — actual SynthDef build via `sclang`

## What not to commit

- `runs/` — runtime WAV and session artifacts
- `workspace/current_run` — symlink to active run
- `~/.openclaw/openclaw.json` — secrets and personal config
- Large `.wav` files (use `examples/` for small demos only)

## License

Project code: see repository license.

Reference example SuperCollider code is © respective authors from [doc.sccode.org](https://doc.sccode.org/).

# sc-soundmatch-agent

Iterative sound matching: an LLM agent writes SuperCollider synthesis code, renders audio, and compares it to a target — guided by FluCoMa analysis and spectral metrics.

- FluCoMa analysis of the target (partials, residual, templates A–E)
- LLM agent proposes synthesis *structure* each iteration; a Python optimizer tunes the numbers
- Hill-climb on the best attempt (never mutate a worse result)
- Numeric parameter optimizer (`optimize_params.py`) via `// @param` annotations
- Deterministic NRT renders (`RandSeed`) so scores are comparable across candidates
- Spectral evaluate/compare loop with score history and regression warnings
- Plateau detection and target-seeded architecture switching when stuck
- Local models via Ollama (e.g. Qwen3.6)

## How it works

```text
target.wav
    │
    ├─ evaluate.py          → spectral metrics + categories
    ├─ analyze_partials.py  → FluCoMa partials + SC templates A–E
    │
    └─ OpenClaw agent loop (workspace/AGENTS.md)
           write attempt_N.scd  (with // @param annotations)
           pre_validate.py      → static SC checks (classes, envelopes)
           wrap_for_recording.py  → NRT wrapper + SynthDef validation (+ RandSeed)
           sclang                 → attempt_N.wav
           optimize_params.py     → tune @param values, rewrite attempt_N.scd/wav
           evaluate.py + compare.py
                ├─ score history + hill-climb instruction
                ├─ BASE CODE = best attempt (not latest)
                └─ plateau → target-seeded architecture switch
           repeat until converged or max iterations
```

Each run creates a timestamped directory under `runs/` with all attempts, comparisons, and a `final_result.scd`.

## Convergent loop

Earlier versions of the pipeline tended to *random-walk*: the agent always edited its latest attempt, numeric tuning was left to verbal hints, and stochastic UGens made identical code score differently each render. Scores could improve once then drift back up.

The current loop is a two-tier hill-climb:

1. **LLM (structure)** — proposes one architectural change per iteration (oscillators, envelopes, filters). It always starts from the **best** attempt so far (`BASE CODE FOR NEXT ATTEMPT` in `comparison_N.txt`), not the latest. Regression warnings tell it when a change made things worse.
2. **Python optimizer (numbers)** — after each structural proposal, `optimize_params.py` runs coordinate descent over annotated parameters, rendering many candidates deterministically and keeping only improvements.

Additional stabilizers:

- **Continuous category penalty** — near-miss categories cost less than large mismatches, so `composite_score` does not jump when a metric crosses a threshold.
- **Deterministic renders** — `wrap_for_recording.py` injects `RandSeed` so `WhiteNoise`, `LFNoise`, `Dust`, etc. produce byte-identical output across optimization candidates.
- **Plateau handling** — if no new best score for 4 iterations, `compare.py` mandates an architecture switch using templates whose frequencies are seeded from the target's dominant partials (`target_partials.txt`). Tried architectures are tracked in `progress.json`.

Mark 3–8 tunable parameters in each attempt:

```supercollider
var cutoff = 3000;   // @param 800 8000 log
var nlev = 0.05;     // @param 0.005 0.2 log
```

The optimizer rewrites the literals and re-renders `attempt_N.wav` before evaluation. See `workspace/AGENTS.md` for the full agent protocol.

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
| Python 3.10+ | Evaluation, comparison, wrapping, pre-validation, optimization |
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
              --optimizer-budget 30 \
              --model ollama/qwen3.6:latest
```

| Flag | Default | Description |
|---|---|---|
| `--target` | *(required)* | Path to target WAV file |
| `--max-iter` | 85 | Maximum refinement iterations |
| `--threshold` | 0.4 | Convergence goal (`composite_score` below this) |
| `--optimizer-budget` | 30 | Max NRT renders per parameter-optimization step |
| `--model` | `ollama/qwen3-coder-next:latest` | OpenClaw model id |
| `--no-telegram` | off | Disable Telegram progress notifications |

Output lands in `runs/YYYYMMDD_HHMMSS_<basename>/`:

- `target_eval.txt`, `target_partials.txt` — pre-computed analysis
- `config.txt` — run settings including `optimizer_budget`
- `attempt_N.scd`, `attempt_N.wav` — synthesis attempts (post-optimization)
- `attempt_N_optlog.txt` — parameter optimization trajectory
- `comparison_N.txt` — metrics, score history, hill-climb instruction, correction prompt
- `progress.json` — score history, best attempt, plateau/architecture state
- `final_result.scd`, `report.md` — best result and summary

For quick debugging, trim the target to a few seconds first:

```bash
ffmpeg -i long_clip.wav -t 3 -y short_clip.wav
./launcher.sh --target short_clip.wav --max-iter 5 --optimizer-budget 24 --model ollama/qwen3.6:latest
```

## Project structure

```text
launcher.sh                 Run orchestrator
analyze_partials.py         FluCoMa analysis + SC templates
evaluate.py                 Target spectral evaluation
compare.py                  Attempt vs target comparison + hill-climb feedback
optimize_params.py          Numeric @param optimizer (coordinate descent)
pre_validate.py             Fast static SC code checks
wrap_for_recording.py       NRT wrapper + SynthDef validation + RandSeed
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

NRT wraps also inject `RandSeed` after the `var` block so stochastic UGens render identically across optimization candidates.

## What not to commit

- `runs/` — runtime WAV and session artifacts
- `workspace/current_run` — symlink to active run
- `~/.openclaw/openclaw.json` — secrets and personal config
- Large `.wav` files (use `examples/` for small demos only)

## License

Project code: see repository license.

Reference example SuperCollider code is © respective authors from [doc.sccode.org](https://doc.sccode.org/).

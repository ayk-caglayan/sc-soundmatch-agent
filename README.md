# sc-soundmatch-agent
Iterative sound matching: execute SuperCollider code, render audio, compare to target — guided by FluCoMa + OpenClaw.

- FluCoMa analysis of the target (partials, residual, templates A–E)
- LLM agent writes SuperCollider .scd code each iteration
- Code is executed (sclang) and rendered to attempt_N.wav
- Spectral evaluate/compare loop drives the next revision
- Plateau detection and architecture switching when stuck
- Local models via Ollama (e.g. Qwen3.6)

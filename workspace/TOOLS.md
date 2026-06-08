# Tool Usage Guide

## exec — Running Commands

### SuperCollider (sclang)

Always use these settings when running sclang:

```
QT_QPA_PLATFORM=offscreen timeout 20 sclang <file.scd>
```

- `QT_QPA_PLATFORM=offscreen` prevents GUI errors on headless systems.
- `timeout 20` kills sclang after 20 seconds. Most synthesis completes in under 15s.
- Exit code 124 means timeout — the code likely has an infinite loop or is missing a `doneAction: 2` or `0.exit`.
- Exit code 0 means success.

### Python Evaluation Scripts

All scripts are in `/home/ayk/sc_claw_flucoma/`. Use `python3`:

```
python3 /home/ayk/sc_claw_flucoma/evaluate.py <audio.wav> -o <output.txt>
python3 /home/ayk/sc_claw_flucoma/compare.py <target.wav> <attempt.wav> -o <output.txt>
python3 /home/ayk/sc_claw_flucoma/wrap_for_recording.py <input.scd>
```

## read — Inspecting Files

Use `read` to inspect:
- `config.txt` — iteration limit and convergence threshold
- `target_eval.txt` — target sound analysis
- `comparison_N.txt` — comparison reports after each iteration
- `attempt_N.scd` — your current SuperCollider code (before revising)

## write — Creating Files

Use `write` to create:
- `attempt_N.scd` — your SuperCollider synthesis code
- `final_result.scd` — copy of the best attempt
- `report.md` — summary report

## Restrictions

- Do NOT use exec to install packages, modify system files, or run anything outside the run directory and the sc_claw_flucoma scripts.
- Do NOT use exec to access the network.
- Do NOT modify `target.wav` or `target_eval.txt`.

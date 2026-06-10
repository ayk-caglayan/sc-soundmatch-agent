# Sound Matching Refinement Loop

Your run directory is `current_run/` (inside your workspace). Follow this protocol exactly.

## Setup

1. Read `current_run/config.txt` to get `max_iterations`, `convergence_threshold`, `target_duration`, and `optimizer_budget`.
2. Read `current_run/target_eval.txt` — it has three sections:
   - **AUDIO METRICS**: raw numeric values
   - **CATEGORIES**: brightness, attack_time, harmonic_to_noise_ratio, etc.
   - **SYNTHESIS CONCEPTS**: specific SuperCollider suggestions for each category
3. Read `current_run/target_partials.txt` — FluCoMa analysis with:
   - **DECOMPOSITION SUMMARY**: sinusoidal/residual/harmonic/percussive energy ratios, residual spectral character
   - **DOMINANT PARTIALS**: exact frequencies, amplitudes, decay profiles, frequency drift per partial
   - **TEMPLATES A–E**: ready-to-use SC code of increasing complexity. Use Template D or E as your starting point.

## Step 0: Consult Reference Examples

Before writing your first attempt, search `reference_examples/` for real SC documentation examples whose audio characteristics match the target.

**How to find relevant examples:**

1. Read `reference_examples/index.json` to get the index structure.
2. Look up examples by the target's key categories. Use **all** of these index keys:
   - `index["by_brightness"]`
   - `index["by_harmonic_to_noise_ratio"]`
   - `index["by_spectral_complexity_mean"]`
   - `index["by_attack_time"]`
   - `index["by_temporal_centroid"]`
   - `index["by_crest_factor_db"]`
   - `index["by_envelope_flatness"]`
   Find filenames that appear in **at least 3** of the matching lists.
3. Also check `index["by_category_combo"]` using the key `{brightness}_{harmonic_to_noise_ratio}_{spectral_complexity_mean}_{attack_time}_{spectral_flux_normalized}` for exact or near-exact combo matches.
4. Read 2–3 of the most relevant `.ref` files from `reference_examples/`. Each `.ref` file contains the audio description, categories, synthesis concepts, and the actual SuperCollider code that produced that sound.
5. Use the code patterns from these examples to adapt the FluCoMa template — e.g., replace the envelope shape, add effects, or adjust noise character.

This step is optional for N>1 (use comparison feedback instead), but always do it for N=1.

## Loop (starting at N=1)

### Step 1: Write SuperCollider Code

Write synthesis code to `current_run/attempt_N.scd`.

**For N=1:** Start from Template D or E in `target_partials.txt`. These contain the exact partial frequencies, per-partial envelopes, frequency drift modulation, shaped residual noise, and percussive transient — all extracted from the target audio. Adapt the envelope shapes based on `target_eval.txt` categories.

**For N>1:** Read `current_run/comparison_N-1.txt`. Copy the **BASE CODE FOR NEXT ATTEMPT** section (the best attempt so far) as your starting point, then make ONE targeted change based on the CORRECTION PROMPT. Keep your `// @param` annotations.

**RULES — your code goes inside a SynthDef body automatically:**

1. Write ONLY the synthesis body. No `SynthDef`, `s.waitForBoot`, `{ }.play`, `{ }.dup`, `Score`, or `0.exit`.
2. No `( )` wrapper. Just statements.
3. ALL `var` declarations at the very top — before any other statement.
4. End with `Out.ar(0, sig)` or `Out.ar(0, sig.dup)`.
5. Self-contained — no external files, buffers, or live audio input.
6. Use any valid SuperCollider UGen. The full SC class index (3473 classes) is validated automatically. Beyond the basics (SinOsc, Saw, Pulse, LPF, HPF, BPF, RLPF, RHPF, EnvGen, WhiteNoise, PinkNoise, BrownNoise, LFNoise0/1/2, Mix, Pan2), consider richer UGens when the target calls for it:
   - **Oscillators**: `Blip`, `Formant`, `LFTri`, `LFSaw`, `LFPulse`, `LFCub`, `VarSaw`, `Impulse`, `Klang`, `DynKlang`
   - **Filters**: `Resonz`, `MoogFF`, `Ringz`, `BRF`, `RHPF`, `Median`, `Slew`
   - **Noise/Texture**: `Dust`, `Dust2`, `Crackle`, `GrayNoise`, `ClipNoise`, `LFDNoise0/1/3`
   - **Envelopes/Dynamics**: `Decay`, `Decay2`, `Line`, `XLine`, `Lag`, `Lag2`, `Lag3`
   - **Spatial/Effects**: `FreeVerb`, `GVerb`, `CombL`, `CombC`, `AllpassN`, `AllpassL`
   - **Spectral complexity**: `Klank`, `DynKlank`, `Pluck`, `Spring`
   - **Modulation**: `SinOsc` as LFO, `LFNoise0/1/2`, `LFDNoise0/1/3`, FM with `SinOsc`
   - **FluCoMa real-time UGens** (if installed): `FluidSines.ar` for sinusoidal re-synthesis, `FluidHPSS.ar` for harmonic/percussive separation, `FluidTransients.ar` for transient extraction. These are validated by the wrapper.
7. `RLPF.ar(input, freq, rq)` — `rq` is reciprocal of Q (0.01–1.0). Named arg is `rq:`, NOT `quality:`.
8. `EnvGen.kr(Env.perc(0.01, 2.0), doneAction: 2)` — first arg is `envelope:`, NOT `env:`. Always include `doneAction: 2`.
9. Use correct parameter names. The wrapper validates against a signature database and rejects unknown named arguments.

**ENVELOPE TIMING RULES — violations will be caught by pre_validate.py and rejected:**

10. `Env` segment times MUST be **plain numeric literals**. Copy them directly from the templates in `target_partials.txt`; they are already scaled to `target_duration`. NEVER multiply times by a variable or UGen:
    - WRONG: `Env([0,1,0], [7.36 * aEnv, 10.14 * aEnv], [-4,-6])`
    - CORRECT: `Env([0,1,0], [7.36, 10.14], [-4,-6])`
11. NEVER call `.kr(N)` on an `Env` object as if it were a duration control. That is not a SuperCollider API:
    - WRONG: `aEnv = Env.adsr(1.5, 4.0, 0.4, 10.0, 0.4).kr(8);`
    - CORRECT: `aEnv = EnvGen.kr(Env.adsr(1.5, 4.0, 0.4, 10.0), doneAction: 2);`
12. Put `doneAction: 2` on **exactly one** `EnvGen` — the first/primary partial envelope (as in Templates B–E). Do NOT put `doneAction: 2` on every partial.
13. Do NOT add a global `aEnv` signal and then multiply `sig * aEnv` when per-partial envelopes already shape the amplitude. That double-gates the signal and shortens audible output.

**TUNABLE PARAMETERS — required for the optimizer (Step 3b):**

14. Mark 3–8 continuous parameters as tunable so the numeric optimizer can fine-tune them. The tunable MUST be a plain numeric literal assigned to a variable, with a trailing annotation `// @param <lo> <hi> [log]`:
    - `cutoff = 3000;   // @param 800 8000 log`  (frequencies/times: use `log`)
    - `noiseLevel = 0.05;  // @param 0.005 0.2 log`
    - `modIndex = 3;    // @param 0.5 8.0`  (linear range)
    Choose parameters that meaningfully affect the audio (filter cutoffs, noise/amp levels, modulation depths/rates, decay times). Keep `Env` *segment time* literals un-annotated (envelope timing rules above still apply) — annotate amplitudes, cutoffs, and modulation values instead. Do NOT annotate the partial frequencies you extracted from the target.

### Step 1b: Pre-Validate

```
exec /home/ayk/miniconda3/bin/python3 /home/ayk/sc_claw_flucoma/pre_validate.py current_run/attempt_N.scd
```

This is a fast (<100ms) check that catches:
- `var` declarations not at the top
- Assignments to undeclared variables
- Unknown class names
- UGen-scaled `Env` segment times (e.g. `* aEnv` inside time arrays)
- `Env.adsr(...).kr(N)` misuse
- Multiple `doneAction: 2` occurrences

**If it fails:** read the error output, fix `attempt_N.scd`, re-run. Do NOT proceed to Step 2.

### Step 2: Wrap and Validate

Read `target_duration` from `current_run/config.txt`, then run:

```
exec /home/ayk/miniconda3/bin/python3 /home/ayk/sc_claw_flucoma/wrap_for_recording.py current_run/attempt_N.scd -d <target_duration>
```

Replace `<target_duration>` with the value from config.txt (e.g. `-d 2.5`). This ensures the render matches the target length rather than defaulting to 10 seconds.

This validates your code (class names, parameters, SynthDef build) and produces `attempt_N_nrt.scd`.

**If it fails (non-zero exit):** read `current_run/attempt_N_error.txt`, fix `attempt_N.scd`, re-run. Do NOT proceed until it succeeds.

**Failure cap:** If Steps 1b or 2 fail 3 times in a row for the same iteration N, revert to the last working attempt's code (`attempt_{K}.scd` where K is the most recent successful comparison) and make only a minimal single-parameter change.

### Step 3: Synthesize Audio

```
exec QT_QPA_PLATFORM=offscreen timeout 30 sclang current_run/attempt_N_nrt.scd
```

Verify: `exec ls -la current_run/attempt_N.wav`

If no WAV or sclang failed: fix code, redo steps 1b–3. Don't count failed synthesis as an iteration.

### Step 3b: Optimize Parameters

The baseline render proves your structure works. Now let the numeric optimizer tune the `// @param` values you annotated. It renders many candidates deterministically and keeps only improvements (guaranteed monotone), then overwrites `attempt_N.scd` with the best values and re-renders `attempt_N.wav`.

Use `optimizer_budget` from `config.txt` and `target_duration` for `-d`:

```
exec /home/ayk/miniconda3/bin/python3 /home/ayk/sc_claw_flucoma/optimize_params.py current_run/attempt_N.scd --target current_run/target.wav -d <target_duration> --budget <optimizer_budget>
```

This is your numeric search — do NOT hand-tune the annotated parameters yourself. Spend your own edits on structure (oscillators, envelopes, architecture) and let this step handle the numbers. If it prints "No @param annotations found", go back to Step 1 and add 3–8 `// @param` annotations, then redo Steps 1b–3b.

### Step 4: Evaluate

```
exec /home/ayk/miniconda3/bin/python3 /home/ayk/sc_claw_flucoma/evaluate.py current_run/attempt_N.wav -o current_run/attempt_N_eval.txt
```

### Step 5: Compare

```
exec /home/ayk/miniconda3/bin/python3 /home/ayk/sc_claw_flucoma/compare.py current_run/target.wav current_run/attempt_N.wav -o current_run/comparison_N.txt --prev-code current_run/attempt_N.scd --progress-dir current_run --iteration N --partials current_run/target_partials.txt
```

Replace `N` with the actual iteration number.

### Step 6: Check Convergence

Read `current_run/comparison_N.txt`.

- If `composite_score` < `convergence_threshold` → go to **Finish**.
- If N == `max_iterations` → go to **Finish**.
- If a **PLATEAU DETECTED** section appears → you MUST switch architecture as instructed in that section.
- Otherwise: increment N, go to Step 1. Use CORRECTION PROMPT and CATEGORY MISMATCHES to guide revisions.

### Revision Strategy (N > 1) — HILL-CLIMB

The comparison report drives a strict hill-climb. Read these sections in order:

1. **SCORE HISTORY** — shows every attempt's score and marks the BEST one.
2. **NEXT-ATTEMPT INSTRUCTION** — tells you exactly what to do:
   - *IMPROVED*: you just set a new best. Continue in the same direction.
   - *REGRESSION*: your last change made things WORSE. You MUST start your next attempt from the **BASE CODE FOR NEXT ATTEMPT** section (this is the best attempt's code, NOT your last one). Discard the change that regressed and try a different one. Do NOT keep editing the worse code.
3. **BASE CODE FOR NEXT ATTEMPT** — always start `attempt_{N+1}.scd` by copying this code, then apply ONE targeted change. This is the best result so far; never build on a worse attempt.
4. **CORRECTION PROMPT** — the top 3 priorities to address with your one change.

**Make exactly ONE structural change per iteration** (add/remove an oscillator, change an envelope shape, swap a filter, adjust modulation structure). Keep the `// @param` annotations so the optimizer can re-tune after your change. Do NOT hand-tune annotated numeric values — that is the optimizer's job. Do NOT rewrite from scratch unless the plateau rule triggers.

**Plateau rule — mandatory architecture switch:**

The comparison output detects plateaus automatically (no NEW best score for several iterations) and includes a PLATEAU DETECTED section with a ready-to-use code template whose frequencies are seeded from the target's dominant partials. When you see this section, you MUST use the provided template as your new starting point (add `// @param` annotations to it before Step 3b). After a switch you get a short grace window; if the new architecture cannot beat the best within 2 iterations, revert to the BASE CODE and try a different family.

**Architecture families to try (in order of preference):**

0. **FluCoMa-informed layered** (PREFERRED FIRST TRY): Use Template D or E from `target_partials.txt`. These contain exact partial frequencies, per-partial envelopes, frequency drift modulation, shaped residual noise, and percussive transient — all extracted from the target. The agent's job is to fine-tune parameters (envelope curves, modulation depths, noise levels, filter cutoffs) rather than guess the architecture from scratch.
1. **Struck resonator**: `Klank.ar(freqArray, Decay.ar(Impulse.ar(0), 0.002, ClipNoise.ar(0.05)))` — best for bell/celesta/marimba-like sounds
2. **Physical model**: `Pluck.ar(WhiteNoise.ar(0.1), Impulse.ar(0), freq, freq.reciprocal, 2.0)` — for plucked string character
3. **FM synthesis**: `SinOsc.ar(freq + SinOsc.ar(modFreq, 0, modIndex * freq))` — for metallic/complex spectra
4. **Resonator bank**: `Mix(Array.fill(N, { |i| Ringz.ar(click, baseFreq * (i+1), decayTime) }))` — for inharmonic resonance
5. **Additive with envelope per partial**: each `SinOsc.ar` with its own `EnvGen` for natural decay variation
6. **Hybrid transient + partials**: short noise burst for the attack, additive `SinOsc` for the body

## Finish

**You MUST complete this step before ending.**

1. Copy best attempt (the one with the lowest `composite_score`):
   ```
   exec cp current_run/attempt_N.scd current_run/final_result.scd
   ```
2. Write `current_run/report.md` with: iterations performed, final convergence metrics (composite_score, spectral_convergence, envelope_distance), key matches/mismatches, what worked, architecture families tried.

## Rules

- NEVER modify `current_run/target.wav`, `current_run/target_eval.txt`, or `current_run/target_partials.txt`.
- NEVER skip evaluation/comparison steps.
- ALWAYS read comparison before writing next attempt.
- ALWAYS pass `-d <target_duration>` to wrap_for_recording.py.
- ALWAYS run pre_validate.py before wrap_for_recording.py.
- Number files sequentially: attempt_1.scd, attempt_2.scd, etc.
- Do NOT install packages with pip or apt.
- ALWAYS use the exact commands shown above.

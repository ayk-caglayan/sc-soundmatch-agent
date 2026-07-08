#!/usr/bin/env python3
"""
Numeric parameter optimizer for SuperCollider synthesis attempts.

The LLM agent proposes the *structure* of a synth; this script tunes the
*numbers*. It reads `// @param lo hi [log]` annotations from an attempt .scd,
then runs a (1+lambda) evolution strategy — rendering each candidate to audio
(deterministic NRT) and scoring it against the target — to find parameter
values that minimize the composite score. The ES moves all params jointly
(diagonal search with 1/5 step-size adaptation) and is monotone: the parent is
only replaced by a strictly better offspring, so the score never regresses
within a fixed architecture.

Annotation convention (the tunable is the numeric literal after `=`):

    var cutoff = 1200;   // @param 400 8000 log
    var decay = 1.5;     // @param 0.2 6.0

`log` makes the search step multiplicatively (good for frequencies/times).

Usage:
    optimize_params.py current_run/attempt_3.scd \
        --target current_run/target.wav -d 2.5 --budget 30

On success it overwrites the attempt .scd with the optimized values, regenerates
`<attempt>_nrt.scd`, and renders the optimized audio to `<attempt>.wav` so the
normal evaluate/compare steps pick up the tuned result.
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from synthesis_evaluator_fixed import SynthesisEvaluator, load_and_preprocess
from compare import compute_category_penalty
from wrap_for_recording import sanitize_code, wrap_code


_PARAM_RE = re.compile(r'@param\s+(-?[\d.]+)\s+(-?[\d.]+)(?:\s+(log))?', re.IGNORECASE)
_ASSIGN_RE = re.compile(r'(=\s*)(-?\d+\.?\d*)')


class Param:
    """A single tunable numeric literal on a specific line."""

    def __init__(self, line_idx, init, lo, hi, log):
        self.line_idx = line_idx
        self.init = init
        self.lo = lo
        self.hi = hi
        # Log stepping only valid for strictly positive ranges.
        self.log = bool(log) and lo > 0 and hi > 0

    def clamp(self, v):
        return max(self.lo, min(self.hi, v))


def parse_params(lines):
    """Extract Param objects from annotated lines. Returns [] if none."""
    params = []
    for idx, line in enumerate(lines):
        m = _PARAM_RE.search(line)
        if not m:
            continue
        lo = float(m.group(1))
        hi = float(m.group(2))
        log = m.group(3)
        if hi < lo:
            lo, hi = hi, lo
        code_part = line.split('//', 1)[0]
        a = _ASSIGN_RE.search(code_part)
        if not a:
            # No `= <number>` to tune on this line; skip silently.
            continue
        init = float(a.group(2))
        params.append(Param(idx, init, lo, hi, log))
    return params


def _fmt(v):
    """Format a value compactly, avoiding scientific notation for typical ranges."""
    if v == int(v) and abs(v) < 1e6:
        return str(int(v))
    return f'{v:.6g}'


def apply_values(base_lines, params, values):
    """Return code text with each param's literal replaced by its value."""
    lines = list(base_lines)
    for p, v in zip(params, values):
        line = lines[p.line_idx]
        code_part, sep, comment = line.partition('//')
        new_code = _ASSIGN_RE.sub(
            lambda mm: mm.group(1) + _fmt(v), code_part, count=1
        )
        lines[p.line_idx] = new_code + sep + comment
    return '\n'.join(lines)


def _perturb(p, current, sigma, rng):
    """Gaussian perturbation of one param, scaled by its (log) range."""
    if p.log:
        span = math.log(p.hi) - math.log(p.lo)
        nxt = math.exp(math.log(max(current, p.lo)) + sigma * span * rng.standard_normal())
    else:
        span = p.hi - p.lo
        nxt = current + sigma * span * rng.standard_normal()
    return p.clamp(nxt)


def evolution_strategy(base_lines, params, scorer, budget, lam=8,
                       init_sigma=0.25, log=print):
    """(1+lambda)-ES with Rechenberg 1/5 step-size adaptation.

    Replaces coordinate descent so the search moves JOINTLY across all params
    (diagonal moves), escaping the axis-aligned grooves coordinate descent
    stalls in — essential once the synth has coupled layer gains. Monotone:
    the parent is only replaced by a strictly better offspring, so the score
    never regresses within a fixed architecture.

    ponytail: ceiling is full CMA-ES (adapts the covariance matrix, not just a
    global step) — `pip install cma` and swap this function if joint directions
    matter more than joint step size. The ES here captures the main benefit
    (diagonal search + step adaptation) in ~40 lines with no new dependency.
    """
    rng = np.random.default_rng(0)
    parent = [p.clamp(p.init) for p in params]
    best_score = scorer.score(apply_values(base_lines, params, parent))
    budget -= 1
    log(f"  baseline score: {best_score:.4f} (renders left: {budget})")
    trajectory = [(list(parent), best_score)]

    sigma = init_sigma
    success_window = []
    generation = 0

    while budget >= lam and budget > 0:
        offspring = []
        for _ in range(lam):
            cand = [_perturb(p, parent[i], sigma, rng) for i, p in enumerate(params)]
            s = scorer.score(apply_values(base_lines, params, cand))
            budget -= 1
            offspring.append((s, cand))
        offspring.sort(key=lambda x: x[0])
        best_off_score, best_off = offspring[0]

        improved = best_off_score < best_score - 1e-5
        success_window.append(1 if improved else 0)
        if improved:
            parent = best_off
            best_score = best_off_score
            trajectory.append((list(parent), best_score))
            log(f"  gen {generation}: sigma {sigma:.3f} | score {best_score:.4f} "
                f"| renders left: {budget}")

        # Rechenberg 1/5 rule: adapt sigma every ~lam evaluations.
        if len(success_window) >= lam:
            success_rate = sum(success_window) / len(success_window)
            if success_rate > 0.20:
                sigma *= 1.2
            else:
                sigma *= 0.85
            sigma = max(sigma, 1e-3)
            success_window = []
        generation += 1

    return parent, best_score, trajectory


class Scorer:
    """Renders candidate code and scores it against a cached target."""

    def __init__(self, target_path, duration, sr=44100):
        self.sr = sr
        self.duration = duration
        self.ev = SynthesisEvaluator(sample_rate=sr)
        self.target_audio, _ = load_and_preprocess(
            target_path, sr=sr, normalize=True, trim_silence=True
        )
        self.target_metrics = self.ev.evaluate(self.target_audio)
        self.target_categories = self.ev.categorize_metrics(self.target_metrics)
        self.workdir = Path(tempfile.mkdtemp(prefix='opt_params_'))
        self._tag = 0
        self.renders = 0

    def cleanup(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _render(self, code):
        """Render code body to a WAV; return path or None on failure."""
        self._tag += 1
        wav_name = f'cand_{self._tag}.wav'
        nrt_path = self.workdir / f'cand_{self._tag}_nrt.scd'
        wrapped = wrap_code(sanitize_code(code), wav_name, duration=self.duration)
        nrt_path.write_text(wrapped, encoding='utf-8')
        env = os.environ.copy()
        env['QT_QPA_PLATFORM'] = 'offscreen'
        try:
            subprocess.run(
                ['sclang', str(nrt_path)],
                capture_output=True, text=True, timeout=45, env=env,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
        wav_path = self.workdir / wav_name
        return wav_path if wav_path.exists() else None

    def score(self, code):
        """Lower is better. Returns +inf when the render fails."""
        self.renders += 1
        wav_path = self._render(code)
        if wav_path is None:
            return float('inf')
        try:
            audio, _ = load_and_preprocess(
                str(wav_path), sr=self.sr, normalize=True, trim_silence=True
            )
        except Exception:
            return float('inf')
        if audio.size == 0:
            return float('inf')
        attempt_metrics = self.ev.evaluate(audio)
        attempt_categories = self.ev.categorize_metrics(attempt_metrics)
        penalty = compute_category_penalty(
            self.ev, self.target_categories, attempt_categories
        )
        conv = self.ev.compare_with_reference(
            audio, self.target_audio, category_penalty=penalty
        )
        return conv['composite_score']


def prune_layers(base_lines, params, values, best_score, scorer, budget, log=print):
    """Mute each layer-gain param in turn; flag non-contributing layers.

    A layer gain is a @param with range ~[0, 1] (the bus-skeleton convention).
    Setting it to 0 mutes that layer. If the score does not get worse, the layer
    is prunable — the agent should drop it on the next iteration to keep the
    patch idiomatic and free the optimizer's budget for the layers that matter.

    Returns a list of (param_index, muted_score) for prunable layers.
    """
    gain_idx = [i for i, p in enumerate(params)
                if p.lo < 0.01 and abs(p.hi - 1.0) < 0.01]
    if len(gain_idx) < 2 or budget <= 0:
        return []
    prunable = []
    for i in gain_idx:
        if budget <= 0:
            break
        cand = list(values)
        cand[i] = 0.0
        s = scorer.score(apply_values(base_lines, params, cand))
        budget -= 1
        verdict = 'PRUNABLE' if s <= best_score + 1e-3 else 'keeps'
        log(f"  mute param {i} -> score {s:.4f} ({verdict}, best={best_score:.4f})")
        if s <= best_score + 1e-3:
            prunable.append((i, s))
    return prunable


def detect_bound_pins(params, values, base_lines, tol=0.02):
    """Flag params pinned at their range min/max after optimization.

    A noise/amp parameter stuck at its boundary is the signature of metric
    gaming — the optimizer wants more (or less) of it than the allowed range
    permits, usually because it lowers MFCC by adding wrong-shape noise. Surfacing
    this makes the system *realize* the noise instead of silently maxing it.
    Returns a list of human-readable warning strings.
    """
    pins = []
    for i, p in enumerate(params):
        v = values[i]
        span = p.hi - p.lo
        if span <= 0:
            continue
        if v >= p.hi - tol * span:
            where = 'MAX'
        elif v <= p.lo + tol * span:
            where = 'MIN'
        else:
            continue
        # Extract the variable name from the line (token before '=').
        code_part = base_lines[p.line_idx].split('//', 1)[0]
        lhs = code_part.split('=', 1)[0].strip()
        var = lhs.split()[-1] if lhs.split() else f'param{i}'
        pins.append(
            f"  {var} = {_fmt(v)} pinned at {where} of [{_fmt(p.lo)}, {_fmt(p.hi)}] "
            f"(line {p.line_idx + 1}) — likely metric gaming; "
            f"{'too much of this source — consider reducing the range max or removing the layer'
            if where == 'MAX' else 'this source is being suppressed — consider removing the layer'}"
        )
    return pins


def main():
    parser = argparse.ArgumentParser(description='Optimize @param values in an SC attempt')
    parser.add_argument('attempt', help='Path to attempt_N.scd with @param annotations')
    parser.add_argument('--target', required=True, help='Path to target.wav')
    parser.add_argument('-d', '--duration', type=float, required=True,
                        help='Render duration in seconds (match target_duration)')
    parser.add_argument('--budget', type=int, default=30,
                        help='Max number of renders (default: 30)')
    parser.add_argument('--prune-budget', type=int, default=0,
                        help='Extra renders for the mute-and-prune pass (0 = skip). '
                             'Run after Phase B bus optimization to drop non-contributing layers.')
    parser.add_argument('--sample-rate', type=int, default=44100)
    args = parser.parse_args()

    attempt_path = Path(args.attempt)
    if not attempt_path.exists():
        print(f"Error: attempt file not found: {attempt_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.target):
        print(f"Error: target not found: {args.target}", file=sys.stderr)
        sys.exit(1)

    original = attempt_path.read_text(encoding='utf-8')
    base_lines = original.splitlines()
    params = parse_params(base_lines)

    if not params:
        print("No @param annotations found — nothing to optimize. "
              "Add `// @param lo hi [log]` comments to tunable lines.")
        sys.exit(0)

    print(f"Optimizing {len(params)} parameter(s) over budget={args.budget} renders:")
    for i, p in enumerate(params):
        scale = 'log' if p.log else 'linear'
        print(f"  [{i}] line {p.line_idx + 1}: init={_fmt(p.init)} "
              f"range=[{_fmt(p.lo)}, {_fmt(p.hi)}] ({scale})")

    scorer = Scorer(args.target, duration=args.duration, sr=args.sample_rate)
    try:
        values, best_score, trajectory = evolution_strategy(
            base_lines, params, scorer, args.budget
        )

        # Optional mute-and-prune pass: drop non-contributing bus layers.
        prune_report = []
        if args.prune_budget > 0:
            prunable = prune_layers(base_lines, params, values, best_score,
                                    scorer, args.prune_budget)
            if prunable:
                prune_report.append("Prunable layers (mute did not worsen score):")
                for i, s in prunable:
                    prune_report.append(
                        f"  param {i} (line {params[i].line_idx + 1}): "
                        f"muted score {s:.4f} -> DROP this layer next iteration"
                    )
            else:
                idxs = [i for i, p in enumerate(params)
                        if p.lo < 0.01 and abs(p.hi - 1.0) < 0.01]
                if idxs:
                    prune_report.append("All layer gains contribute — none prunable.")

        # Detect params pinned at range bounds — a metric-gaming signal.
        pin_warnings = detect_bound_pins(params, values, base_lines)

        best_code = apply_values(base_lines, params, values)

        # Persist optimized params back into the attempt file.
        attempt_path.write_text(best_code, encoding='utf-8')

        # Regenerate the NRT script and render the canonical attempt WAV so the
        # downstream evaluate/compare steps use the optimized audio.
        final_wav = attempt_path.with_suffix('.wav').name
        wrapped = wrap_code(sanitize_code(best_code), final_wav, duration=args.duration)
        nrt_path = attempt_path.with_name(attempt_path.stem + '_nrt.scd')
        nrt_path.write_text(wrapped, encoding='utf-8')
        env = os.environ.copy()
        env['QT_QPA_PLATFORM'] = 'offscreen'
        subprocess.run(['sclang', str(nrt_path)], capture_output=True,
                       text=True, timeout=45, env=env)

        # Write a short optimization log next to the attempt.
        log_path = attempt_path.with_name(attempt_path.stem + '_optlog.txt')
        log_lines = [
            f"Parameter optimization for {attempt_path.name}",
            f"renders used: {scorer.renders}",
            f"final composite_score: {best_score:.4f}",
            "",
            "Optimized values:",
        ]
        for i, (p, v) in enumerate(zip(params, values)):
            log_lines.append(f"  [{i}] line {p.line_idx + 1}: {_fmt(p.init)} -> {_fmt(v)}")
        log_lines.append("")
        log_lines.append("Improvement trajectory (score):")
        for _vals, sc in trajectory:
            log_lines.append(f"  {sc:.4f}")
        if prune_report:
            log_lines.append("")
            log_lines.extend(prune_report)
        if pin_warnings:
            log_lines.append("")
            log_lines.append("BOUND-PIN WARNINGS (params stuck at range edge — likely metric gaming):")
            log_lines.extend(pin_warnings)
        log_path.write_text('\n'.join(log_lines), encoding='utf-8')

        print(f"Optimization complete: composite_score {trajectory[0][1]:.4f} "
              f"-> {best_score:.4f} over {scorer.renders} renders.")
        print(f"Optimized code written to {attempt_path}")
        print(f"Optimized audio rendered to {attempt_path.with_suffix('.wav')}")
        if prune_report:
            print("\n".join(prune_report))
        if pin_warnings:
            print("BOUND-PIN WARNINGS (params stuck at range edge — likely metric gaming):")
            print("\n".join(pin_warnings))
    finally:
        scorer.cleanup()


if __name__ == '__main__':
    main()

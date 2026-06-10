#!/usr/bin/env python3
"""
Numeric parameter optimizer for SuperCollider synthesis attempts.

The LLM agent proposes the *structure* of a synth; this script tunes the
*numbers*. It reads `// @param lo hi [log]` annotations from an attempt .scd,
then runs coordinate descent — rendering each candidate to audio (deterministic
NRT) and scoring it against the target — to find parameter values that minimize
the composite score. This guarantees monotone (non-increasing) improvement
within a fixed architecture, which the LLM-only loop could not provide.

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


def step_value(p, current, direction, frac):
    """Move `current` by `frac` of the (log) range in `direction` (+/-1)."""
    if p.log:
        lo_l, hi_l, cur_l = math.log(p.lo), math.log(p.hi), math.log(max(current, p.lo))
        nxt = math.exp(cur_l + direction * frac * (hi_l - lo_l))
    else:
        nxt = current + direction * frac * (p.hi - p.lo)
    return p.clamp(nxt)


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


def coordinate_descent(base_lines, params, scorer, budget, start_frac=0.3,
                       min_frac=0.02, log=print):
    """Greedy coordinate descent over param values. Returns (values, score)."""
    values = [p.clamp(p.init) for p in params]
    best_score = scorer.score(apply_values(base_lines, params, values))
    budget -= 1
    log(f"  baseline score: {best_score:.4f} (renders left: {budget})")
    trajectory = [(list(values), best_score)]

    frac = start_frac
    while frac >= min_frac and budget > 0:
        improved = False
        for i, p in enumerate(params):
            if budget <= 0:
                break
            for direction in (1, -1):
                if budget <= 0:
                    break
                cand = list(values)
                cand[i] = step_value(p, values[i], direction, frac)
                if cand[i] == values[i]:
                    continue
                s = scorer.score(apply_values(base_lines, params, cand))
                budget -= 1
                if s < best_score - 1e-5:
                    best_score = s
                    values = cand
                    improved = True
                    trajectory.append((list(values), best_score))
                    log(f"  param {i} -> {_fmt(cand[i])} | score {best_score:.4f} "
                        f"| step {frac:.3f} | renders left: {budget}")
                    break  # accept and move to next param
        if not improved:
            frac *= 0.5
    return values, best_score, trajectory


def main():
    parser = argparse.ArgumentParser(description='Optimize @param values in an SC attempt')
    parser.add_argument('attempt', help='Path to attempt_N.scd with @param annotations')
    parser.add_argument('--target', required=True, help='Path to target.wav')
    parser.add_argument('-d', '--duration', type=float, required=True,
                        help='Render duration in seconds (match target_duration)')
    parser.add_argument('--budget', type=int, default=30,
                        help='Max number of renders (default: 30)')
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
        values, best_score, trajectory = coordinate_descent(
            base_lines, params, scorer, args.budget
        )

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
        log_path.write_text('\n'.join(log_lines), encoding='utf-8')

        print(f"Optimization complete: composite_score {trajectory[0][1]:.4f} "
              f"-> {best_score:.4f} over {scorer.renders} renders.")
        print(f"Optimized code written to {attempt_path}")
        print(f"Optimized audio rendered to {attempt_path.with_suffix('.wav')}")
    finally:
        scorer.cleanup()


if __name__ == '__main__':
    main()

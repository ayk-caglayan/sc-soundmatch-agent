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
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path

import numpy as np

from synthesis_evaluator_fixed import SynthesisEvaluator, load_and_preprocess
import synthesis_evaluator_fixed as sev
from compare import compute_category_penalty, resolve_loss_config, apply_loss_config
from wrap_for_recording import sanitize_code, wrap_code


# ============================================================================
# Approach #6: Parameter entropy / sensitivity tracking
# ============================================================================
# ponytail: crude variance tracking per param; no distribution model needed.
# Per Peladeau et al. 2025, the parameter→audio map is many-to-one — some
# params barely affect the score (high entropy), others are tightly coupled.
# The ES should spend its mutation budget where it matters.

class ParamSensitivity:
    """Track which params matter via pairwise score-stability across renders."""

    def __init__(self, n_params):
        self.n = n_params
        self._values = []       # list of (param_vec, score) tuples

    def record(self, values, score):
        self._values.append((list(values), score))

    def entropy_weights(self):
        """Return a length-n weight vector (0..1).  High weight = high
        sensitivity (small param change → big score change). Low weight =
        high entropy (param can vary with little effect)."""
        if len(self._values) < 3:
            return np.ones(self.n) / self.n
        arr = np.array([v for v, _ in self._values])
        scores = np.array([s for _, s in self._values])
        score_std = np.std(scores) or 1.0
        weights = np.zeros(self.n)
        for i in range(self.n):
            col = arr[:, i]
            col_std = np.std(col) or 1e-9
            # Correlation magnitude: does varying this param move the score?
            if col_std > 1e-9:
                corr = abs(np.corrcoef(col, scores)[0, 1]) if len(scores) > 2 else 0.0
            else:
                corr = 0.0
            # Weight = correlation * normalized std (params that vary AND
            # correlate with score are the ones that matter).
            weights[i] = corr * min(col_std / (np.mean(col) + 1e-9), 1.0)
        total = weights.sum() or 1.0
        return weights / total


# ============================================================================
# Approach #2: PNP Jacobian surrogate
# ============================================================================
# Han et al. 2024: precompute Jacobian of (perception o synth) around the
# current best point, build quadratic surrogate, run ES against the surrogate.
# Validates only the winner with a real render → huge speedup.
# ponytail: finite-difference Jacobian, QR-factorized for the quadratic form.

class PNPSurrogate:
    """Quadratic surrogate f(θ) ≈ score(θ) around a reference point."""

    def __init__(self, ref_params, ref_score, jacobian=None):
        self.ref = np.array(ref_params, dtype=float)
        self.ref_score = ref_score
        self._jac = jacobian  # estimated gradient vector at ref
        self._hess_diag = None  # diagonal Hessian approximation

    def estimate_jacobian(self, params, scorer, eps=0.02, budget=0):
        """Finite-difference gradient at ref point. Uses up to n_params+1
        renders. Returns remaining budget."""
        n = len(params)
        grad = np.zeros(n)
        base_score = scorer.score(
            apply_values(scorer._base_lines, params, self.ref)
        )
        budget -= 1
        for i in range(n):
            if budget <= 0:
                break
            delta = np.zeros(n)
            span = params[i].hi - params[i].lo
            h = max(eps * span, 1e-6)
            delta[i] = h
            fwd = self.ref + delta
            fwd_score = scorer.score(
                apply_values(scorer._base_lines, params,
                             [params[i].clamp(v) for i, v in enumerate(fwd)])
            )
            budget -= 1
            grad[i] = (fwd_score - base_score) / (h + 1e-12)
        self._jac = grad
        self.ref_score = base_score
        return budget

    def predict(self, theta):
        """Quadratic approximation: f(ref) + Jᵀ(θ-ref) + ½(θ-ref)ᵀH(θ-ref)."""
        d = np.array(theta, dtype=float) - self.ref
        linear = float(np.dot(self._jac, d)) if self._jac is not None else 0.0
        if self._hess_diag is not None:
            quad = 0.5 * float(np.dot(d * d, self._hess_diag))
        else:
            quad = 0.0
        return self.ref_score + linear + quad

    def build_hessian_diag(self, params, scorer, eps=0.05, budget=0):
        """Estimate diagonal Hessian entries via second finite difference.
        Returns remaining budget."""
        n = len(params)
        diag = np.zeros(n)
        for i in range(n):
            if budget <= 0:
                break
            span = params[i].hi - params[i].lo
            h = max(eps * span, 1e-6)
            fwd = self.ref.copy()
            bwd = self.ref.copy()
            fwd[i] = params[i].clamp(self.ref[i] + h)
            bwd[i] = params[i].clamp(self.ref[i] - h)
            s_fwd = scorer.score(
                apply_values(scorer._base_lines, params, fwd)
            )
            budget -= 1
            if budget <= 0:
                break
            s_bwd = scorer.score(
                apply_values(scorer._base_lines, params, bwd)
            )
            budget -= 1
            diag[i] = max(0.0, (s_fwd - 2 * self.ref_score + s_bwd) / (h * h + 1e-12))
        self._hess_diag = diag
        return budget


# ============================================================================
# Approach #4: RL reward replay buffer
# ============================================================================
# SynthRL (Shin & Lee 2025): store (params, delta, reward) tuples, bias
# subsequent mutations toward previously successful directions. ponytail:
# simple sliding-window buffer, no transformer — 20 lines, same concept.

class ReplayBuffer:
    """Remembers successful mutation directions and re-samples them."""

    def __init__(self, capacity=50):
        self._buf = deque(maxlen=capacity)

    def record(self, parent, child, parent_score, child_score):
        """Store a mutation that improved the score."""
        improvement = parent_score - child_score
        if improvement > 1e-6:
            p = np.array(parent, dtype=float)
            c = np.array(child, dtype=float)
            delta = c - p
            norm = np.linalg.norm(delta) or 1.0
            self._buf.append({
                'delta': delta / norm,  # unit direction
                'magnitude': norm,
                'improvement': improvement,
            })

    def sample_direction(self, rng):
        """Return a unit direction vector biased by past successes, or None."""
        if not self._buf:
            return None
        # Weight by improvement, sample one
        weights = [e['improvement'] + 1e-6 for e in self._buf]
        total = sum(weights)
        probs = [w / total for w in weights]
        idx = rng.choice(len(self._buf), p=probs)
        return self._buf[idx]['delta'].copy()

    def __len__(self):
        return len(self._buf)


# ============================================================================
# Approach #5: Neural proxy for black-box synth
# ============================================================================
# Combes et al. 2025: train a small NN mapping params→score, use as a fast
# approximate evaluator during ES. Real renders validate the top candidate.
# ponytail: single-hidden-layer MLP with sklearn (no torch dep).  Falls back
# gracefully when sklearn is absent.

_NEURAL_PROXY_AVAILABLE = False
try:
    from sklearn.neural_network import MLPRegressor  # noqa: F401
    from sklearn.preprocessing import StandardScaler  # noqa: F401
    _NEURAL_PROXY_AVAILABLE = True
except ImportError:
    pass


class NeuralProxy:
    """Fast approximate scorer trained on (params, score) pairs."""

    def __init__(self):
        self._model = None
        self._scaler = None

    def fit(self, param_list, score_list):
        """Train on collected (param_vec, score) pairs. Returns True if OK."""
        if not _NEURAL_PROXY_AVAILABLE or len(param_list) < 5:
            return False
        try:
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
            X = np.array(param_list, dtype=float)
            y = np.array(score_list, dtype=float)
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
            self._model = MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation='relu',
                max_iter=500,
                random_state=0,
            )
            self._model.fit(X_scaled, y)
            return True
        except Exception:
            return False

    def predict(self, theta):
        """Estimated score, or None if not fitted."""
        if self._model is None or self._scaler is None:
            return None
        try:
            X = np.array(theta, dtype=float).reshape(1, -1)
            X_scaled = self._scaler.transform(X)
            return float(self._model.predict(X_scaled)[0])
        except Exception:
            return None


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
                       init_sigma=0.25, log=print,
                       use_pnp=False, use_replay=False,
                       use_sensitivity=False):
    """(1+lambda)-ES with Rechenberg 1/5 step-size adaptation.

    Replacements for coordinate descent: joint diagonal search with step-size
    adaptation, monotone parent replacement.

    Feature flags (all default off for backward compat):

      use_pnp (approach #2):
        Finite-difference Jacobian + quadratic surrogate around the best
        point. The ES evaluates candidates against the surrogate (free)
        and only renders the best surrogate-predicted candidate for real.
        This gives ~lam× speedup per generation.

      use_replay (approach #4):
        RL-style replay buffer biases mutations toward previously successful
        directions. This makes mutations data-driven instead of isotropic.

      use_sensitivity (approach #6):
        Per-param entropy tracking weights the mutation magnitude by each
        param's measured sensitivity — spend budget where it matters.

    ponytail: ceiling is full CMA-ES (adapts the covariance matrix) via
    `pip install cma`.  The ES here captures the main benefits in ~60 lines
    with no new dependency.
    """

    # Validate base code renders before starting optimization.
    base_code = apply_values(base_lines, params,
                             [p.clamp(p.init) for p in params])
    base_score = scorer.score(base_code)
    if base_score >= float('inf') or base_score > 5.0:
        log(f"  WARNING: baseline score {base_score:.2f} is invalid or very poor. "
            "Check that the base code renders and produces audible output.")
        # Don't abort — the ES might still find a working mutation.
    budget -= 1
    log(f"  baseline score: {base_score:.4f} (renders left: {budget})")
    parent = [p.clamp(p.init) for p in params]
    best_score = base_score

    rng = np.random.default_rng(0)
    trajectory = [(list(parent), best_score)]

    # --- optional accelerators ---
    pnp = PNPSurrogate(parent, best_score) if use_pnp else None
    replay = ReplayBuffer() if use_replay else None
    sensitivity = ParamSensitivity(len(params)) if use_sensitivity else None

    # Record the baseline for sensitivity and replay.
    if sensitivity:
        sensitivity.record(parent, best_score)
    if replay:
        replay.record(parent, parent, best_score, best_score)  # seed dummy

    sigma = init_sigma
    success_window = []
    generation = 0

    # PNP: estimate Jacobian at baseline after first real eval.
    if pnp and budget > len(params) + 2:
        budget = pnp.estimate_jacobian(params, scorer, budget=budget)
        if budget > 3 * len(params) + lam * 2:
            budget = pnp.build_hessian_diag(params, scorer, budget=budget)
        log(f"  PNP surrogate built (gradient norm={np.linalg.norm(pnp._jac or np.zeros(len(params))):.4f}, "
            f"renders left: {budget})")

    while budget >= lam and budget > 0:
        offspring = []
        for _ in range(lam):
            # Build candidate, optionally biased by replay direction.
            if use_replay and replay and len(replay) > 0 and rng.random() < 0.3:
                direction = replay.sample_direction(rng)
                cand = list(parent)
                for i, p_i in enumerate(params):
                    span = p_i.hi - p_i.lo if not p_i.log else (
                        math.log(p_i.hi) - math.log(p_i.lo))
                    # 30% replay-directed, 70% random perturbation
                    cand[i] = _perturb(p_i, cand[i], sigma * 0.5, rng)
                    if direction is not None and i < len(direction):
                        cand[i] += direction[i] * span * sigma * 0.3
                        cand[i] = p_i.clamp(cand[i])
            else:
                # Standard isotropic mutation, optionally weighted by
                # per-param sensitivity (approach #6).
                if use_sensitivity and sensitivity and len(sensitivity._values) >= 3:
                    w = sensitivity.entropy_weights()
                else:
                    w = None
                cand = list(parent)
                for i, p_i in enumerate(params):
                    si = sigma * (w[i] * len(params) if w is not None else 1.0)
                    cand[i] = _perturb(p_i, cand[i], si, rng)

            # Score candidate: use PNP surrogate if available, else real render.
            if use_pnp and pnp and pnp._jac is not None:
                pred_score = pnp.predict(cand)
                offspring.append((pred_score, cand, True))  # True = surrogate
            else:
                s = scorer.score(apply_values(base_lines, params, cand))
                budget -= 1
                offspring.append((s, cand, False))

            # Track sensitivity with real scores.
            if sensitivity and not offspring[-1][2]:
                sensitivity.record(cand, offspring[-1][0])

        # Sort by predicted score (ascending = better).
        offspring.sort(key=lambda x: x[0])

        # If using surrogate, render only the top candidate for validation.
        if use_pnp and pnp and any(o[2] for o in offspring):
            best_pred_score, best_pred_cand, is_surrogate = offspring[0]
            if is_surrogate:
                real_score = scorer.score(
                    apply_values(base_lines, params, best_pred_cand))
                budget -= 1
                offspring[0] = (real_score, best_pred_cand, False)
                best_off_score, best_off = real_score, best_pred_cand
            else:
                best_off_score, best_off = offspring[0][0], offspring[0][1]
        else:
            best_off_score, best_off = offspring[0][0], offspring[0][1]

        improved = best_off_score < best_score - 1e-5
        success_window.append(1 if improved else 0)

        if improved:
            # Record successful mutation in replay buffer.
            if use_replay and replay:
                replay.record(parent, best_off, best_score, best_off_score)
            if sensitivity:
                sensitivity.record(best_off, best_off_score)

            parent = best_off
            best_score = best_off_score
            trajectory.append((list(parent), best_score))
            log(f"  gen {generation}: sigma {sigma:.3f} | score {best_score:.4f} "
                f"| renders left: {budget}")

            # PNP: recompute Jacobian at new parent when budget allows.
            if (use_pnp and pnp and budget > len(params) + lam + 5):
                pnp.ref = np.array(parent, dtype=float)
                pnp.ref_score = best_score
                pnp._jac = None
                pnp._hess_diag = None
                budget = pnp.estimate_jacobian(params, scorer, budget=budget)
                if budget > 3 * len(params) + lam * 2:
                    budget = pnp.build_hessian_diag(params, scorer, budget=budget)
                log(f"  PNP Jacobian refreshed (renders left: {budget})")

        # Rechenberg 1/5 rule.
        if len(success_window) >= lam:
            success_rate = sum(success_window) / len(success_window)
            if success_rate > 0.20:
                sigma *= 1.2
            else:
                sigma *= 0.85
            sigma = max(sigma, 1e-3)
            success_window = []
        generation += 1

    # --- sensitivity report (approach #6) ---
    if sensitivity and len(sensitivity._values) >= 3:
        w = sensitivity.entropy_weights()
        # Identify high-entropy params (weight < 0.5 * uniform) — these
        # barely affect the score and are candidates for wider ranges or
        # removal.
        uniform = 1.0 / len(params)
        low_sens = [(i, params[i], w[i]) for i in range(len(params))
                    if w[i] < 0.5 * uniform]
        if low_sens:
            log("  Param sensitivity report (low-sensitivity = barely matters):")
            for i, p, wi in low_sens:
                log(f"    param [{i}] line {p.line_idx + 1}: "
                    f"weight={wi:.4f} (uniform={uniform:.4f}) — "
                    f"consider widening range or removing @param")

    return parent, best_score, trajectory


class Scorer:
    """Renders candidate code and scores it against a cached target."""

    def __init__(self, target_path, duration, sr=44100, progress_dir=None,
                 loss_config=None):
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
        self._base_lines = []  # set by evolution_strategy before scoring
        cfg = loss_config
        if cfg is None and progress_dir:
            cfg = resolve_loss_config(progress_dir)
        apply_loss_config(cfg)

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
    # Feature flags for new approaches (all default off — backward compat).
    parser.add_argument('--use-pnp', action='store_true',
                        help='Enable PNP Jacobian surrogate (approach #2): '
                             'finite-diff gradient + quadratic model for faster ES.')
    parser.add_argument('--use-replay', action='store_true',
                        help='Enable RL replay buffer (approach #4): '
                             'bias mutations toward previously successful directions.')
    parser.add_argument('--use-sensitivity', action='store_true',
                        help='Enable per-param sensitivity tracking (approach #6): '
                             'weight mutations by measured importance.')
    parser.add_argument('--use-neural-proxy', action='store_true',
                        help='Enable neural proxy (approach #5): '
                             'train small MLP on (params,score) and use as surrogate.')
    parser.add_argument('--use-jtfs', action='store_true',
                        help='Enable JTFS perceptual distance (approach #3): '
                             'requires kymatio, adds modulation-aware metric to scorer.')
    parser.add_argument('--progress-dir', default=None,
                        help='Run directory; reads progress.json loss_config when set.')
    parser.add_argument('--loss-config', default=None,
                        choices=['default', 'spectral_heavy', 'perceptual_heavy'],
                        help='Override loss weight config for scoring.')
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

    scorer = Scorer(args.target, duration=args.duration, sr=args.sample_rate,
                    progress_dir=args.progress_dir, loss_config=args.loss_config)
    scorer._base_lines = base_lines  # exposed for PNP surrogate
    # Wire up JTFS if requested.
    if args.use_jtfs:
        scorer.ev = SynthesisEvaluator(sample_rate=args.sample_rate, use_jtfs=True)
        scorer.target_metrics = scorer.ev.evaluate(scorer.target_audio)
        scorer.target_categories = scorer.ev.categorize_metrics(scorer.target_metrics)
    try:
        values, best_score, trajectory = evolution_strategy(
            base_lines, params, scorer, args.budget,
            use_pnp=args.use_pnp,
            use_replay=args.use_replay,
            use_sensitivity=args.use_sensitivity,
        )

        # --- Neural proxy post-optimization (approach #5) ---
        # After the ES has collected real renders, train a small NN on
        # (params, score) pairs and run a few surrogate-guided mutations.
        # This echoes Combes et al. 2025: the proxy doesn't need to be
        # perfect, just good enough to rank candidates.
        if args.use_neural_proxy and scorer.renders > 10:
            proxy = NeuralProxy()
            # Collect (params, score) pairs from the Scorer's render history.
            # The ES already recorded them — we re-derive from the trajectory.
            param_list, score_list = [], []
            for vals, sc in trajectory:
                param_list.append(vals)
                score_list.append(sc)
            if proxy.fit(param_list, score_list):
                rng_proxy = np.random.default_rng(42)
                # Evaluate proxy prediction at the current best.
                proxy_pred = proxy.predict(values)
                if proxy_pred is not None:
                    print(f"  Neural proxy trained on {len(param_list)} samples. "
                          f"Best score: {best_score:.4f}, proxy estimate: {proxy_pred:.4f}")
                # Run a few proxy-guided mutations with real validation.
                proxy_budget = min(6, args.budget)
                for pi in range(proxy_budget):
                    cand = list(values)
                    for i, p in enumerate(params):
                        cand[i] = _perturb(p, cand[i], 0.08, rng_proxy)
                    proxy_est = proxy.predict(cand)
                    if proxy_est is not None and proxy_est >= best_score - 0.01:
                        continue  # proxy says no improvement; skip real render
                    real_score = scorer.score(apply_values(base_lines, params, cand))
                    if real_score < best_score - 1e-5:
                        best_score = real_score
                        values = cand
                        trajectory.append((list(values), best_score))
                        print(f"  proxy-guided {pi}: new best {best_score:.4f}")
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

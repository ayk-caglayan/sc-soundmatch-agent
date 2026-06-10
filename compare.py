#!/usr/bin/env python3
"""
Compare attempt audio against target audio.
Produces convergence metrics, category mismatches, metric deltas,
a prioritized correction prompt, and progress tracking.
"""

import sys
import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from synthesis_evaluator_fixed import SynthesisEvaluator, load_and_preprocess


CATEGORY_SUGGESTIONS = {
    'brightness': {
        ('very_dark', 'dark'): 'Raise filter cutoff frequency, use brighter oscillator (Saw, Pulse), or add high-frequency harmonics.',
        ('dark', 'very_dark'): 'Lower filter cutoff slightly or reduce high-mid content.',
        ('bright', 'very_bright'): 'Lower RLPF/LPF cutoff frequency, switch from Pulse to Saw or SinOsc, or reduce high harmonics.',
        ('very_bright', 'bright'): 'Slightly lower filter cutoff or add gentle lowpass filtering.',
        'default_higher': 'REDUCE high frequency content. Lower filter cutoff, use LPF/RLPF, or choose a darker oscillator.',
        'default_lower': 'INCREASE brightness. Raise filter cutoff, use HPF, or add harmonics with Saw/Pulse.',
    },
    'attack_time': {
        'default_higher': 'LENGTHEN attack time. Use Env.adsr with longer attack param (0.1-0.5s).',
        'default_lower': 'SHORTEN attack time. Use Env.perc with smaller attack, or reduce Env.adsr attack param.',
    },
    'harmonic_to_noise_ratio': {
        'default_higher': 'REDUCE noise. Remove WhiteNoise/PinkNoise sources, increase harmonic oscillator amplitude.',
        'default_lower': 'ADD noise or inharmonic content. Mix in WhiteNoise, use ring modulation, or add detuning.',
    },
    'spectral_flux_normalized': {
        'default_higher': 'REDUCE spectral movement. Remove random modulation, stabilize LFO rates, use static filter settings.',
        'default_lower': 'ADD spectral movement. Use LFO on filter cutoff, add frequency modulation, or use Dust-triggered changes.',
    },
    'temporal_centroid': {
        'default_higher': 'SHIFT energy later. Use longer attack, slower build, or back-loaded envelope shape.',
        'default_lower': 'SHIFT energy earlier. Use shorter decay, stronger attack, or front-loaded envelope.',
    },
    'crest_factor_db': {
        'default_higher': 'REDUCE peakiness. Compress the signal, use longer sustain, or flatten the envelope.',
        'default_lower': 'INCREASE transient character. Use Env.perc, add sharp attack, or increase dynamic range.',
    },
    'spectral_complexity_mean': {
        'default_higher': 'REDUCE spectral density. Use fewer oscillators, simpler waveforms (SinOsc), or stronger filtering.',
        'default_lower': 'INCREASE spectral richness. Add more oscillators, use FM synthesis, or widen filter bandwidth.',
    },
    'spectral_slope': {
        'default_higher': 'STEEPEN spectral rolloff. Apply stronger lowpass filter (lower cutoff or higher order).',
        'default_lower': 'FLATTEN spectral slope. Raise filter cutoff, use highpass, or boost high-frequency oscillators.',
    },
    'envelope_flatness': {
        'default_higher': 'FLATTEN the amplitude envelope. Use longer sustain, gentler attack/release, or reduce modulation depth.',
        'default_lower': 'MAKE envelope more dynamic. Use Env.perc or Env.adsr with shorter sustain.',
    },
}

METRIC_SUGGESTIONS = {
    'spectral_centroid_mean': {
        'positive': 'too dark — raise filter cutoff or use brighter oscillator',
        'negative': 'too bright — lower filter cutoff or use darker oscillator',
    },
    'band_energy_sub_bass': {
        'positive': 'needs more sub-bass — add low-frequency oscillator below 60Hz',
        'negative': 'too much sub-bass — apply highpass filter above 60Hz',
    },
    'band_energy_bass': {
        'positive': 'needs more bass — boost oscillator amplitude in 60-250Hz range',
        'negative': 'too much bass — apply highpass or reduce low-frequency oscillator level',
    },
    'band_energy_low_mid': {
        'positive': 'needs more low-mid — boost 250-500Hz content',
        'negative': 'too much low-mid — cut 250-500Hz with BPF or notch',
    },
    'band_energy_mid': {
        'positive': 'needs more mid-range — boost 500-2000Hz content',
        'negative': 'too much mid-range — attenuate 500-2000Hz',
    },
    'band_energy_high_mid': {
        'positive': 'needs more presence (2-4kHz) — raise filter cutoff or add harmonics',
        'negative': 'too much presence (2-4kHz) — lower filter cutoff',
    },
    'band_energy_highs': {
        'positive': 'needs more high frequencies — add brightness, noise, or raise cutoff above 4kHz',
        'negative': 'too much high frequency — apply LPF below 4kHz',
    },
    'attack_time': {
        'positive': 'attack too fast — increase envelope attack parameter',
        'negative': 'attack too slow — decrease envelope attack parameter',
    },
    'harmonic_to_noise_ratio': {
        'positive': 'too noisy — reduce noise sources, increase tonal content',
        'negative': 'too tonal — add noise, detuning, or inharmonic components',
    },
    'spectral_flux_normalized': {
        'positive': 'too static — add modulation (LFO on filter, FM, amplitude modulation)',
        'negative': 'too chaotic — remove or slow down modulation, stabilize parameters',
    },
    'temporal_centroid': {
        'positive': 'energy too front-heavy — lengthen sustain or add slower build',
        'negative': 'energy too back-heavy — shorten decay, use percussive envelope',
    },
    'crest_factor_db': {
        'positive': 'too compressed — increase dynamic range, use percussive envelope',
        'negative': 'too peaky — compress or sustain the signal more',
    },
    'envelope_flatness': {
        'positive': 'envelope too dynamic — flatten with longer sustain',
        'negative': 'envelope too flat — add dynamics with shorter envelope or modulation',
    },
}

_LOW_VALUE_METRICS = {
    'band_energy_sub_bass', 'band_energy_bass', 'band_energy_low_mid',
    'band_energy_high_mid', 'band_energy_highs',
    'spectral_flatness_sub_bass', 'spectral_flatness_bass',
    'spectral_flatness_low_mid', 'spectral_flatness_mid',
    'spectral_flatness_high_mid', 'spectral_flatness_highs',
    'rms_mean', 'rms_std', 'rms_max',
    'onset_mean',
}

_REL_FLOOR = 0.05

ARCHITECTURE_TEMPLATES = {
    'struck_resonator': (
        "var env, click, sig;\n"
        "env = EnvGen.kr(Env.perc(0.001, 1.5, curve: -6), doneAction: 2);\n"
        "click = Decay.ar(Impulse.ar(0), 0.002, ClipNoise.ar(0.05));\n"
        "sig = Klank.ar(`[[670, 1340, 2010, 2680, 3350], [1, 0.6, 0.4, 0.25, 0.15], "
        "[1.5, 1.0, 0.7, 0.5, 0.3]], click);\n"
        "Out.ar(0, (sig * env * 0.3).dup);"
    ),
    'physical_model': (
        "var sig;\n"
        "sig = Pluck.ar(WhiteNoise.ar(0.1), Impulse.ar(0), 440.reciprocal, 440.reciprocal, 2.0, 0.5);\n"
        "Out.ar(0, (sig * 0.3).dup);"
    ),
    'fm_synthesis': (
        "var env, sig, modFreq, modIndex;\n"
        "env = EnvGen.kr(Env.perc(0.01, 2.0), doneAction: 2);\n"
        "modFreq = 440;\n"
        "modIndex = 3;\n"
        "sig = SinOsc.ar(440 + SinOsc.ar(modFreq, 0, modIndex * 440));\n"
        "Out.ar(0, (sig * env * 0.3).dup);"
    ),
    'resonator_bank': (
        "var env, click, sig;\n"
        "env = EnvGen.kr(Env.perc(0.001, 2.0), doneAction: 2);\n"
        "click = Decay.ar(Impulse.ar(0), 0.003, WhiteNoise.ar(0.1));\n"
        "sig = Mix(Array.fill(8, { |i| Ringz.ar(click, 300 * (i+1) * (1 + (0.01 * i)), 1.5 - (0.15*i)) * (1/(i+1)) }));\n"
        "Out.ar(0, (sig * env * 0.2).dup);"
    ),
}

ARCHITECTURE_ORDER = [
    'struck_resonator', 'physical_model', 'fm_synthesis', 'resonator_bank',
]

# Convergence control:
#   PLATEAU_PATIENCE — iterations with no NEW best score before a plateau fires.
#   SWITCH_GRACE     — iterations after an architecture switch during which we do
#                      NOT re-fire a plateau (gives the new architecture room).
#   IMPROVEMENT_EPS  — minimum absolute score drop that counts as a "new best".
PLATEAU_PATIENCE = 4
SWITCH_GRACE = 2
IMPROVEMENT_EPS = 1e-4


def parse_partials(path):
    """Parse dominant partials from a target_partials.txt file.

    Returns a list of (freq_hz, amp) tuples, ordered as in the file
    (i.e. by descending average magnitude). Empty list if unparseable.
    """
    partials = []
    if not path or not os.path.exists(path):
        return partials
    line_re = re.compile(r'#\d+:\s*([\d.]+)\s*Hz,\s*amp=([\d.]+)')
    try:
        text = Path(path).read_text(encoding='utf-8')
    except OSError:
        return partials
    for line in text.splitlines():
        m = line_re.search(line)
        if m:
            partials.append((float(m.group(1)), float(m.group(2))))
    return partials


def _fmt_list(vals, fmt='{:.1f}'):
    return '[' + ', '.join(fmt.format(v) for v in vals) + ']'


def build_seeded_templates(partials):
    """Build architecture templates seeded with the target's actual partials.

    Falls back to the generic hard-coded templates when no partials are
    available so the system still behaves on targets without FluCoMa output.
    """
    if not partials:
        return dict(ARCHITECTURE_TEMPLATES)

    top = partials[:5]
    tfreqs = [f for f, _ in top]
    tamps = [a for _, a in top]
    max_amp = max(tamps) or 1.0
    namps = [round(a / max_amp, 3) for a in tamps]
    ringtimes = [round(max(0.3, 1.5 - 0.25 * i), 2) for i in range(len(tfreqs))]
    fundamental = tfreqs[0]
    mod_freq = tfreqs[1] if len(tfreqs) > 1 else fundamental

    struck = (
        "var env, click, sig;\n"
        "env = EnvGen.kr(Env.perc(0.001, 1.5, curve: -6), doneAction: 2);\n"
        "click = Decay.ar(Impulse.ar(0), 0.002, ClipNoise.ar(0.05));\n"
        f"sig = Klank.ar(`[{_fmt_list(tfreqs)}, {_fmt_list(namps, '{:.3f}')}, "
        f"{_fmt_list(ringtimes, '{:.2f}')}], click);\n"
        "Out.ar(0, (sig * env * 0.3).dup);"
    )

    physical = (
        "var sig;\n"
        f"sig = Pluck.ar(WhiteNoise.ar(0.1), Impulse.ar(0), {fundamental:.1f}.reciprocal, "
        f"{fundamental:.1f}.reciprocal, 2.0, 0.5);\n"
        "Out.ar(0, (sig * 0.3).dup);"
    )

    fm = (
        "var env, sig, modFreq, modIndex;\n"
        "env = EnvGen.kr(Env.perc(0.01, 2.0), doneAction: 2);\n"
        f"modFreq = {mod_freq:.1f};\n"
        "modIndex = 3;\n"
        f"sig = SinOsc.ar({fundamental:.1f} + SinOsc.ar(modFreq, 0, modIndex * {fundamental:.1f}));\n"
        "Out.ar(0, (sig * env * 0.3).dup);"
    )

    ringz_voices = ', '.join(
        f"Ringz.ar(click, {f:.1f}, {rt:.2f}) * {a:.3f}"
        for f, a, rt in zip(tfreqs, namps, ringtimes)
    )
    resonator = (
        "var env, click, sig;\n"
        "env = EnvGen.kr(Env.perc(0.001, 2.0), doneAction: 2);\n"
        "click = Decay.ar(Impulse.ar(0), 0.003, WhiteNoise.ar(0.1));\n"
        f"sig = Mix([{ringz_voices}]);\n"
        "Out.ar(0, (sig * env * 0.2).dup);"
    )

    return {
        'struck_resonator': struck,
        'physical_model': physical,
        'fm_synthesis': fm,
        'resonator_bank': resonator,
    }


def load_audio(path, sr=44100):
    audio, _dur = load_and_preprocess(path, sr=sr, normalize=True, trim_silence=True)
    return audio


def get_category_direction(cat_name, target_label, current_label):
    evaluator = SynthesisEvaluator()
    if cat_name not in evaluator.category_thresholds:
        return 'unknown', 0
    labels = evaluator.category_thresholds[cat_name]['labels']
    try:
        t_idx = labels.index(target_label)
        c_idx = labels.index(current_label)
    except ValueError:
        return 'unknown', 0
    distance = abs(c_idx - t_idx)
    if c_idx > t_idx:
        return 'higher', distance
    elif c_idx < t_idx:
        return 'lower', distance
    return 'match', 0


def get_suggestion(cat_name, target_label, current_label):
    direction, _ = get_category_direction(cat_name, target_label, current_label)
    if direction == 'match':
        return None

    suggestions = CATEGORY_SUGGESTIONS.get(cat_name, {})
    pair_key = (target_label, current_label)
    if pair_key in suggestions:
        return suggestions[pair_key]
    default_key = f'default_{direction}'
    if default_key in suggestions:
        return suggestions[default_key]
    return f'Adjust {cat_name}: target is {target_label}, current is {current_label}.'


def compute_category_penalty(evaluator, target_categories, attempt_categories):
    """Mean normalized label distance across all categories (0..1, lower better).

    Matched categories contribute 0; a mismatch contributes its label-bin
    distance divided by the category's bin span. Shared by compare() and the
    parameter optimizer so both score on the same continuous penalty.
    """
    total = len(target_categories) or 1
    penalty_sum = 0.0
    for cat_name, t_label in target_categories.items():
        c_label = attempt_categories.get(cat_name, 'unknown')
        if t_label == c_label:
            continue
        labels = evaluator.category_thresholds.get(cat_name, {}).get('labels', [])
        try:
            distance = abs(labels.index(c_label) - labels.index(t_label))
        except ValueError:
            distance = 1
        span = max(1, len(labels) - 1)
        penalty_sum += distance / span
    return penalty_sum / total


def _rank_score(t_val, abs_delta, key):
    if key in _LOW_VALUE_METRICS:
        return abs(abs_delta) * 0.1
    if abs(t_val) >= _REL_FLOOR:
        return abs(abs_delta) / abs(t_val)
    return abs(abs_delta)


def compare(target_path, attempt_path, sr=44100):
    target_audio = load_audio(target_path, sr)
    attempt_audio = load_audio(attempt_path, sr)

    evaluator = SynthesisEvaluator(sample_rate=sr)

    target_metrics = evaluator.evaluate(target_audio)
    attempt_metrics = evaluator.evaluate(attempt_audio)

    target_categories = evaluator.categorize_metrics(target_metrics)
    attempt_categories = evaluator.categorize_metrics(attempt_metrics)

    mismatches = []
    for cat_name in target_categories:
        t_label = target_categories[cat_name]
        c_label = attempt_categories.get(cat_name, 'unknown')
        if t_label != c_label:
            suggestion = get_suggestion(cat_name, t_label, c_label)
            _, distance = get_category_direction(cat_name, t_label, c_label)
            mismatches.append((cat_name, t_label, c_label, suggestion, distance))

    mismatches.sort(key=lambda x: x[4], reverse=True)

    # Continuous category penalty: mean normalized label distance across ALL
    # categories (matched ones contribute 0). This varies smoothly so the
    # composite score does not jump when a metric merely crosses a threshold.
    category_penalty = compute_category_penalty(
        evaluator, target_categories, attempt_categories
    )

    convergence = evaluator.compare_with_reference(
        attempt_audio, target_audio,
        category_mismatches=len(mismatches),
        category_penalty=category_penalty,
    )

    skip_metrics = {k for k in target_metrics if k.startswith('mfcc_') or k.startswith('delta')}
    deltas = []
    for key in target_metrics:
        if key in skip_metrics:
            continue
        t_val = target_metrics[key]
        c_val = attempt_metrics.get(key, 0.0)
        abs_delta = t_val - c_val
        score = _rank_score(t_val, abs_delta, key)
        deltas.append((key, t_val, c_val, abs_delta, score))

    deltas.sort(key=lambda x: x[4], reverse=True)

    return convergence, mismatches, deltas[:10]


def build_correction_prompt(mismatches, top_deltas):
    parts = []

    top_mismatches = mismatches[:3]
    if top_mismatches:
        parts.append("FIX THESE FIRST (by priority):")
        for i, (cat_name, t_label, c_label, suggestion, _dist) in enumerate(top_mismatches, 1):
            parts.append(f"  PRIORITY {i}: {cat_name} should be {t_label} but is {c_label}. {suggestion}")

    top3_deltas = []
    for key, t_val, c_val, abs_delta, _ in top_deltas[:3]:
        info = METRIC_SUGGESTIONS.get(key, {})
        direction = 'positive' if abs_delta > 0 else 'negative'
        hint = info.get(direction, '')
        if hint:
            top3_deltas.append(hint)
    if top3_deltas:
        parts.append("Metric fixes: " + "; ".join(top3_deltas) + ".")

    return "\n".join(parts) if parts else "No significant corrections needed."


def update_progress(output_dir, iteration, composite_score, seeded_templates=None):
    """Update progress.json with score history, elitism, and plateau handling.

    Plateau detection is based on lack of a NEW best score over the last
    PLATEAU_PATIENCE iterations (a hill-climb stall), not on the raw score
    delta between consecutive — possibly noisy — attempts. When a plateau
    fires, the next untried architecture is selected, recorded in
    ``architectures_tried``, and a SWITCH_GRACE window is opened so the new
    architecture is not immediately declared a plateau too.
    """
    progress_path = os.path.join(output_dir, "progress.json")

    progress = {"scores": [], "best_score": None, "best_attempt": None,
                "plateau_detected": False, "architectures_tried": [],
                "iters_since_best": 0, "last_switch_iteration": 0,
                "switch_architecture": None, "regressed": False,
                "delta_vs_best": 0.0}
    if os.path.exists(progress_path):
        try:
            with open(progress_path) as f:
                progress = json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass

    progress.setdefault('architectures_tried', [])
    progress.setdefault('last_switch_iteration', 0)

    progress['scores'].append(composite_score)
    progress['iteration'] = iteration

    is_new_best = (
        progress['best_score'] is None
        or composite_score < progress['best_score'] - IMPROVEMENT_EPS
    )
    if is_new_best:
        progress['best_score'] = composite_score
        progress['best_attempt'] = iteration

    progress['iters_since_best'] = iteration - progress['best_attempt']
    progress['delta_vs_best'] = composite_score - progress['best_score']
    progress['is_new_best'] = is_new_best
    progress['regressed'] = (
        not is_new_best
        and progress['best_attempt'] != iteration
        and progress['delta_vs_best'] > IMPROVEMENT_EPS
    )

    grace_ok = (iteration - progress['last_switch_iteration']) > SWITCH_GRACE
    plateau = (
        len(progress['scores']) >= PLATEAU_PATIENCE
        and progress['iters_since_best'] >= PLATEAU_PATIENCE
        and grace_ok
    )

    progress['switch_architecture'] = None
    if plateau:
        arch = _next_untried_architecture(progress)
        progress['switch_architecture'] = arch
        if arch not in progress['architectures_tried']:
            progress['architectures_tried'].append(arch)
        progress['last_switch_iteration'] = iteration

    progress['plateau_detected'] = plateau

    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)

    return progress


def _next_untried_architecture(progress):
    """Pick the next architecture not yet tried, cycling if all were tried."""
    tried = set(progress.get('architectures_tried', []))
    for arch in ARCHITECTURE_ORDER:
        if arch not in tried:
            return arch
    return ARCHITECTURE_ORDER[0]


def format_report(convergence, mismatches, top_deltas, prev_code=None,
                  progress=None, best_code=None, seeded_templates=None):
    lines = []

    composite = convergence.get('composite_score', convergence.get('spectral_convergence', 0))
    lines.append("=== CONVERGENCE METRICS ===")
    lines.append(f"composite_score: {composite:.4f}")
    lines.append(f"spectral_convergence: {convergence.get('spectral_convergence', 0):.4f}")
    lines.append(f"log_spectral_distance: {convergence.get('log_spectral_distance', 0):.4f}")
    lines.append(f"envelope_distance: {convergence.get('envelope_distance', 0):.4f}")
    lines.append(f"snr_db: {convergence.get('snr_db', 0):.2f}")
    lines.append(f"rmse: {convergence.get('rmse', 0):.6f}")
    lines.append("")

    if progress:
        scores = progress.get('scores', [])
        best_attempt = progress.get('best_attempt')
        best_score = progress.get('best_score')
        lines.append("=== SCORE HISTORY (lower is better) ===")
        history = []
        for i, s in enumerate(scores, 1):
            mark = ' <-- BEST' if i == best_attempt else ''
            history.append(f"  attempt {i}: {s:.4f}{mark}")
        lines.extend(history)
        lines.append("")

        lines.append("=== NEXT-ATTEMPT INSTRUCTION (HILL-CLIMB) ===")
        if progress.get('plateau_detected'):
            lines.append(
                f"No new best for {progress.get('iters_since_best', 0)} iterations. "
                "A MANDATORY ARCHITECTURE SWITCH is required (see section below)."
            )
        elif progress.get('is_new_best'):
            lines.append(
                f"IMPROVED: this is the new best (attempt {best_attempt}, "
                f"{best_score:.4f}). Continue from the BASE CODE below and make ONE "
                "more targeted change in the same direction."
            )
        elif progress.get('regressed'):
            lines.append(
                f"REGRESSION: this attempt scored {composite:.4f}, which is "
                f"+{progress.get('delta_vs_best', 0):.4f} WORSE than the best "
                f"(attempt {best_attempt}, {best_score:.4f})."
            )
            lines.append(
                f"DISCARD this attempt's direction. Start your next attempt FROM "
                f"the BASE CODE below (attempt {best_attempt}) and make ONE different "
                "targeted change. Do NOT repeat the change that caused this regression."
            )
        else:
            lines.append(
                f"NO IMPROVEMENT: this attempt did not beat the best "
                f"(attempt {best_attempt}, {best_score:.4f}). Start your next attempt "
                "FROM the BASE CODE below and try a DIFFERENT targeted change."
            )
        lines.append("")

    lines.append("=== CATEGORY MISMATCHES (ranked by severity) ===")
    if mismatches:
        for cat_name, t_label, c_label, suggestion, dist in mismatches:
            lines.append(f"{cat_name}: target={t_label}, current={c_label} (distance={dist}) --> {suggestion}")
    else:
        lines.append("(all categories match)")
    lines.append("")

    lines.append("=== METRIC DELTAS (top 10 by actionable priority) ===")
    for key, t_val, c_val, abs_delta, _score in top_deltas:
        sign = '+' if abs_delta > 0 else ''
        info = METRIC_SUGGESTIONS.get(key, {})
        direction = 'positive' if abs_delta > 0 else 'negative'
        hint = info.get(direction, '')
        hint_str = f" --> {hint}" if hint else ''
        lines.append(f"{key}: target={t_val:.4f}, current={c_val:.4f}, delta={sign}{abs_delta:.4f}{hint_str}")
    lines.append("")

    correction = build_correction_prompt(mismatches, top_deltas)
    lines.append("=== CORRECTION PROMPT ===")
    lines.append(correction)
    lines.append("")

    if progress and progress.get('plateau_detected'):
        best_attempt = progress.get('best_attempt')
        best_score = progress.get('best_score')
        arch_name = progress.get('switch_architecture') or ARCHITECTURE_ORDER[0]
        templates = seeded_templates or ARCHITECTURE_TEMPLATES
        arch_code = templates.get(arch_name, ARCHITECTURE_TEMPLATES[arch_name])
        tried = progress.get('architectures_tried', [])
        lines.append("=== PLATEAU DETECTED — MANDATORY ARCHITECTURE SWITCH ===")
        lines.append(
            f"No new best for {progress.get('iters_since_best', 0)} iterations "
            f"(best is attempt {best_attempt} at {best_score:.4f})."
        )
        lines.append(f"Architectures tried so far: {', '.join(tried) if tried else 'none'}.")
        lines.append("You MUST switch to a fundamentally different synthesis architecture.")
        lines.append("Do NOT make incremental tweaks. Rewrite from scratch using this template")
        lines.append("(its frequencies are seeded from the target's dominant partials):")
        lines.append("")
        lines.append(f"Architecture: {arch_name}")
        lines.append(arch_code)
        lines.append("")
        lines.append(
            "If this architecture cannot beat the best score within 2 iterations, "
            f"revert to the BASE CODE below (attempt {best_attempt}) and try another family."
        )
        lines.append("")

    # Elitism: always surface the BEST attempt's code as the base to mutate.
    base_code = best_code if best_code is not None else prev_code
    if base_code is not None:
        best_attempt = progress.get('best_attempt') if progress else None
        label = (f"=== BASE CODE FOR NEXT ATTEMPT (best so far: attempt {best_attempt}) ==="
                 if best_attempt is not None else "=== CURRENT ATTEMPT CODE ===")
        lines.append(label)
        lines.append(base_code)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Compare attempt audio with target')
    parser.add_argument('target', help='Path to target audio file')
    parser.add_argument('attempt', help='Path to attempt audio file')
    parser.add_argument('-o', '--output', help='Output comparison report path')
    parser.add_argument('--prev-code', help='Path to current attempt .scd file (included in report)')
    parser.add_argument('--progress-dir', help='Directory for progress.json tracking')
    parser.add_argument('--iteration', type=int, default=0, help='Current iteration number')
    parser.add_argument('--partials', help='Path to target_partials.txt (seeds architecture templates)')
    parser.add_argument('--sample-rate', type=int, default=44100)
    args = parser.parse_args()

    for path in [args.target, args.attempt]:
        if not os.path.exists(path):
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    convergence, mismatches, top_deltas = compare(
        args.target, args.attempt, sr=args.sample_rate
    )

    prev_code = None
    if args.prev_code and os.path.exists(args.prev_code):
        prev_code = Path(args.prev_code).read_text(encoding='utf-8').strip()

    # Seed architecture templates from the target's actual partials when available.
    partials_path = args.partials
    if not partials_path and args.progress_dir:
        candidate = os.path.join(args.progress_dir, 'target_partials.txt')
        if os.path.exists(candidate):
            partials_path = candidate
    seeded_templates = build_seeded_templates(parse_partials(partials_path))

    progress = None
    best_code = None
    if args.progress_dir and args.iteration > 0:
        composite = convergence.get('composite_score', 0)
        progress = update_progress(args.progress_dir, args.iteration, composite,
                                   seeded_templates=seeded_templates)
        # Elitism: surface the best attempt's code so the agent mutates that,
        # not whatever it most recently wrote.
        best_attempt = progress.get('best_attempt')
        if best_attempt is not None:
            best_path = os.path.join(args.progress_dir, f'attempt_{best_attempt}.scd')
            if os.path.exists(best_path):
                best_code = Path(best_path).read_text(encoding='utf-8').strip()
    if best_code is None:
        best_code = prev_code

    report = format_report(convergence, mismatches, top_deltas,
                           prev_code=prev_code, progress=progress,
                           best_code=best_code, seeded_templates=seeded_templates)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Comparison saved to {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()

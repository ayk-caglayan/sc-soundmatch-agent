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
from synthesis_evaluator_fixed import (
    SynthesisEvaluator, load_and_preprocess, LOSS_CONFIGS,
)
import synthesis_evaluator_fixed as sev


CATEGORY_SUGGESTIONS = {
    'brightness': {
        ('very_dark', 'dark'): 'Raise filter cutoff frequency, use brighter oscillator (Saw, Pulse), or add high-frequency harmonics.',
        ('dark', 'very_dark'): 'Lower filter cutoff slightly or reduce high-mid content.',
        ('bright', 'very_bright'): 'Lower RLPF/LPF cutoff frequency, switch from Pulse to Saw or SinOsc, or reduce high harmonics.',
        ('very_bright', 'bright'): 'Slightly lower filter cutoff or add gentle lowpass filtering.',
        'default_higher': 'REDUCE high frequency content. Lower filter cutoff, use LPF/RLPF, or choose a darker oscillator.',
        'default_lower': 'INCREASE brightness. Try subtractive (Saw/Pulse through MoogFF) or waveshaper_feedback. Raise filter cutoff, use HPF, or add harmonics with Saw/Pulse.',
    },
    'attack_time': {
        'default_higher': 'LENGTHEN attack time. Use Env.adsr with longer attack param (0.1-0.5s).',
        'default_lower': 'SHORTEN attack time. Use Env.perc with smaller attack, or reduce Env.adsr attack param.',
    },
    'harmonic_to_noise_ratio': {
        'default_higher': 'REDUCE noise. Remove WhiteNoise/PinkNoise sources, increase harmonic oscillator amplitude.',
        'default_lower': 'ADD noise or inharmonic content. Try chaos_noise (Gendy1/CuspL through Resonz) or granular (GrainSin cloud). Mix in WhiteNoise, use ring modulation, or add detuning.',
    },
    'spectral_flux_normalized': {
        'default_higher': 'REDUCE spectral movement. Remove random modulation, stabilize LFO rates, use static filter settings.',
        'default_lower': 'ADD spectral movement. Try granular (GrainSin + Dust) or chaos_noise (Gendy1/LatoocarfianL). Use LFO on filter cutoff, add frequency modulation, or use Dust-triggered changes.',
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
        'default_lower': 'INCREASE spectral richness. Try waveshaper_feedback (SinOscFB.tanh), subtractive (detuned Saw through MoogFF), or FM synthesis. Add more oscillators or widen filter bandwidth.',
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
    'granular': (
        "var env, sig, centerFreq;\n"
        "env = EnvGen.kr(Env.perc(0.05, 1.5), doneAction: 2);\n"
        "centerFreq = 440;\n"
        "sig = Mix(GrainSin.ar(2, Dust.ar(15), 0.08, centerFreq, 0, -1, 128));\n"
        "Out.ar(0, (sig * env * 0.3).dup);"
    ),
    'waveshaper_feedback': (
        "var env, sig, feedback;\n"
        "env = EnvGen.kr(Env.perc(0.01, 2.0), doneAction: 2);\n"
        "feedback = 0.5;\n"
        "sig = SinOscFB.ar(440, feedback).tanh;\n"
        "Out.ar(0, (sig * env * 0.3).dup);"
    ),
    'subtractive': (
        "var env, osc, sig, cutoff, modCutoff;\n"
        "env = EnvGen.kr(Env.perc(0.01, 2.0), doneAction: 2);\n"
        "cutoff = 2000;\n"
        "modCutoff = cutoff * EnvGen.kr(Env.perc(0.05, 1.0));\n"
        "osc = Mix(Saw.ar([440, 440 * 1.007]));\n"
        "sig = MoogFF.ar(osc, modCutoff, 2.5);\n"
        "Out.ar(0, (sig * env * 0.3).dup);"
    ),
    'chaos_noise': (
        "var env, chaos, sig, resFreq;\n"
        "env = EnvGen.kr(Env.perc(0.01, 2.0), doneAction: 2);\n"
        "resFreq = 440;\n"
        "chaos = Mix([Gendy1.ar(1, 1, 0.3, 0.3, 200, 800, 0.5, 0.5, 12), "
        "CuspL.ar(100, 3, -3, 0.1) * 0.3]);\n"
        "sig = Resonz.ar(chaos, resFreq, 0.05);\n"
        "Out.ar(0, (sig * env * 0.25).dup);"
    ),
    'formant_vocal': (
        "var env, sig, fund;\n"
        "env = EnvGen.kr(Env.perc(0.05, 1.5), doneAction: 2);\n"
        "fund = 440;\n"
        "sig = Formant.ar(fund, fund * 2.5, fund * 0.2) + Formant.ar(fund, fund * 3.5, fund * 0.25);\n"
        "Out.ar(0, (sig * env * 0.25).dup);"
    ),
    'wood_block': (
        # Inharmonic wood-bar modal model (clave, woodblock, marimba).
        # Exciter: short impulse + PinkNoise.  Resonator: 4 Ringz at
        # inharmonic ratios (~2.76×, 5.4×, 8.93× the fundamental for a
        # typical rosewood bar).  The optimizer tunes freqs, decays, and
        # balance — this seed gives it the right modal structure to start.
        "var exciter, sig, env, fund;\n"
        "fund = 2400;\n"
        "env = EnvGen.kr(Env.perc(0.0003, 0.12, curve: -4), doneAction: 2);\n"
        "exciter = Decay.ar(Impulse.ar(0), 0.0004, PinkNoise.ar(0.15));\n"
        "sig = Mix([\n"
        "  Ringz.ar(exciter, fund, 0.003),\n"
        "  Ringz.ar(exciter, fund * 2.76, 0.002),\n"
        "  Ringz.ar(exciter, fund * 5.4, 0.0015),\n"
        "  Ringz.ar(exciter, fund * 8.93, 0.001),\n"
        "]);\n"
        "sig = sig + (SinOsc.ar(fund, 0, 0.05) * env);\n"
        "Out.ar(0, (sig * env * 0.4).dup);"
    ),
}

ARCHITECTURE_ORDER = [
    'struck_resonator', 'physical_model', 'fm_synthesis', 'resonator_bank',
    'granular', 'waveshaper_feedback', 'subtractive', 'chaos_noise',
    'formant_vocal', 'wood_block',
]

# Ordered list of architecture families used during the seeding phase.
# The FluCoMa-template family is always seed 1 (written by the agent directly
# from target_partials.txt templates, not from ARCHITECTURE_TEMPLATES).
SEED_FAMILIES = ['flucoma_template', *ARCHITECTURE_ORDER]

# Convergence control:
#   PLATEAU_PATIENCE — iterations with no NEW best score before a plateau fires.
#   SWITCH_GRACE     — iterations after an architecture switch during which we do
#                      NOT re-fire a plateau (gives the new architecture room).
#   IMPROVEMENT_EPS  — minimum absolute score drop that counts as a "new best".
PLATEAU_PATIENCE = 4
SWITCH_GRACE = 2
IMPROVEMENT_EPS = 2e-3
SEED_TIEBREAK_EPS = 0.02
PREFERRED_SEED_FAMILIES = ['flucoma_template']
FORMANT_FORCE_MAX_GAP = 0.15  # max score gap to force formant_vocal (else fall through)
RACE_BUDGET = 3               # iterations per finalist in post-seed race
RACE_FINALIST_COUNT = 3       # top N seeds that enter the race


def parse_decomposition_sinusoidal_pct(partials_path):
    """Return sinusoidal_energy percent from target_partials.txt, or None."""
    if not partials_path or not os.path.exists(partials_path):
        return None
    try:
        for line in Path(partials_path).read_text(encoding='utf-8').splitlines():
            if line.startswith('sinusoidal_energy:'):
                return float(line.split(':', 1)[1].strip().rstrip('%'))
    except (OSError, ValueError):
        pass
    return None


def parse_target_field(partials_path, field):
    """Return a float field (e.g. residual_spectral_centroid, fundamental_freq) from target_partials.txt."""
    if not partials_path or not os.path.exists(partials_path):
        return None
    try:
        for line in Path(partials_path).read_text(encoding='utf-8').splitlines():
            if line.startswith(field + ':'):
                val = line.split(':', 1)[1].strip().split()[0]
                return float(val)
    except (OSError, ValueError, IndexError):
        pass
    return None


def parse_target_envelope(partials_path):
    """Read envelope params from target_partials.txt (approach #8).

    Returns a dict with keys matching estimate_envelope() output, or None.
    Used by build_seeded_templates to seed Env.perc/Env.adsr params instead
    of hardcoding ``Env.perc(0.01, 2.0)``.
    """
    if not partials_path or not os.path.exists(partials_path):
        return None
    env = {}
    try:
        lines = Path(partials_path).read_text(encoding='utf-8').splitlines()
        in_envelope = False
        for line in lines:
            if line.startswith('=== AMPLITUDE ENVELOPE'):
                in_envelope = True
                continue
            if in_envelope and line.startswith('==='):
                break
            if in_envelope and ':' in line:
                key, _, val = line.partition(':')
                key = key.strip()
                val = val.strip()
                if key == 'envelope_shape':
                    env['env_shape'] = val
                elif key == 'envelope_attack_sec':
                    env['attack_sec'] = float(val)
                elif key == 'envelope_decay_sec':
                    env['decay_sec'] = float(val)
                elif key == 'envelope_release_sec':
                    env['release_sec'] = float(val)
                elif key == 'envelope_peak_frac':
                    env['peak_frac'] = float(val)
                elif key == 'envelope_sustain_level':
                    env['sustain_level'] = float(val)
    except (OSError, ValueError):
        return None
    return env if env else None


def parse_target_field_str(partials_path, field):
    """Return a string field (e.g. recommended_primary_archetype) from target_partials.txt."""
    if not partials_path or not os.path.exists(partials_path):
        return None
    try:
        for line in Path(partials_path).read_text(encoding='utf-8').splitlines():
            if line.startswith(field + ':'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return None


def pick_seed_winner(seed_scores, partials_path=None):
    """Pick Phase B seed family; prefer the analysis-recommended archetype, then
    flucoma_template, when scores are close."""
    if not seed_scores:
        return None

    best_score = min(seed_scores.values())
    eps = SEED_TIEBREAK_EPS
    sin_pct = parse_decomposition_sinusoidal_pct(partials_path)
    if sin_pct is not None and sin_pct > 50.0:
        eps *= 1.5

    # Formant-driven promotion: if the analysis recommends formant_vocal (strong
    # formants + pitched source), promote it to Phase B base when it was seeded
    # AND its score is within FORMANT_FORCE_MAX_GAP of the best.  The seed score
    # is misleading for formant targets — additive sines match the sines-stem
    # cheaply but cannot reproduce a formant spectrum, so flucoma_template wins
    # on seed score while being architecturally wrong.  However, if the gap is
    # large the formant model is too far behind to catch up — fall through to
    # normal tiebreak so the race or plateau-restart can pick empirically.
    # ponytail: gap gate prevents wasting a full optimization budget on a
    # doomed formant candidate; FORMANT_FORCE_MAX_GAP tuned by running on
    # known formant targets.
    recommended = parse_target_field_str(partials_path, 'recommended_primary_archetype')
    if (recommended == 'formant_vocal' and 'formant_vocal' in seed_scores
            and seed_scores['formant_vocal'] <= best_score + FORMANT_FORCE_MAX_GAP):
        return 'formant_vocal'
    if recommended and recommended in seed_scores:
        eps *= 1.5

    close = [fam for fam, sc in seed_scores.items() if sc <= best_score + eps]
    preference = [recommended] + PREFERRED_SEED_FAMILIES if recommended else PREFERRED_SEED_FAMILIES
    for pref in preference:
        if pref and pref in close:
            return pref
    return min(seed_scores.items(), key=lambda x: x[1])[0]


def attempt_for_family(progress, family):
    """Return attempt number for a seed family name, or None."""
    for attempt, fam in progress.get('attempt_architectures', {}).items():
        if fam == family:
            return int(attempt)
    return None


def apply_seed_winner_tiebreak(progress, partials_path=None):
    """Re-resolve best_attempt after seeding using tiebreak rules."""
    seed_scores = progress.get('seed_scores', {})
    if not seed_scores:
        return progress

    winner_fam = pick_seed_winner(seed_scores, partials_path)
    if not winner_fam:
        return progress

    winner_attempt = attempt_for_family(progress, winner_fam)
    if winner_attempt is None:
        return progress

    raw_best = min(seed_scores.items(), key=lambda x: x[1])[0]
    progress['best_attempt'] = winner_attempt
    progress['best_score'] = seed_scores[winner_fam]
    progress['seed_winner_family'] = winner_fam
    progress['seed_winner_tiebreak'] = winner_fam != raw_best
    recommended = parse_target_field_str(partials_path, 'recommended_primary_archetype')
    progress['formant_forced'] = (
        winner_fam == 'formant_vocal' and recommended == 'formant_vocal'
        and winner_fam != raw_best
    )
    return progress


def _maybe_start_race(progress, partials_path):
    """After seeding, start a short race among the top 2-3 finalists instead
    of crowning one winner immediately.  Returns True if a race was started.

    Grounded in Shier 2021: warm-started short optimization matched full-length
    optimization at ~1/10 the cost.  Three finalists × 3 iterations = 9
    evaluator calls — roughly one plateau cycle.  If it prevents a doomed
    formant_vocal full-optimization run, it's net-negative cost.
    """
    seed_scores = progress.get('seed_scores', {})
    if len(seed_scores) < 2:
        return False

    recommended = parse_target_field_str(partials_path, 'recommended_primary_archetype')

    # Build finalist list: recommended family gets a lane only when its seed
    # score is within FORMANT_FORCE_MAX_GAP of the best — don't waste race
    # iterations on a family that seeded last (ponytail: same threshold as
    # pick_seed_winner, tuned on known formant targets).
    finalists = []
    best_seed_score = min(seed_scores.values())
    if (recommended and recommended in seed_scores
            and seed_scores[recommended] <= best_seed_score + FORMANT_FORCE_MAX_GAP):
        finalists.append(recommended)
    rest = sorted(
        [(s, f) for f, s in seed_scores.items() if f not in finalists],
        key=lambda x: x[0],
    )
    for _score, fam in rest[:RACE_FINALIST_COUNT - len(finalists)]:
        finalists.append(fam)

    finalists = finalists[:RACE_FINALIST_COUNT]
    if len(finalists) < 2:
        return False

    # Map each finalist to its seed attempt so the agent knows which template
    # to develop.
    attempt_map = {}
    for attempt_str, fam in progress.get('attempt_architectures', {}).items():
        if fam in finalists and fam not in attempt_map:
            attempt_map[fam] = int(attempt_str)

    progress['race_active'] = True
    progress['race_finalists'] = finalists
    progress['race_iterations'] = {f: 0 for f in finalists}
    progress['race_scores'] = {f: seed_scores.get(f, float('inf')) for f in finalists}
    progress['race_best_attempts'] = {f: attempt_map.get(f) for f in finalists}
    progress['race_current_idx'] = 0
    progress['race_budget'] = RACE_BUDGET
    # Don't pick a winner yet — let the race decide.
    progress['seed_winner_family'] = None
    return True


def _advance_race(progress, iteration, composite_score, arch):
    """Update race state after one iteration.  Rotates to the next finalist
    that still has budget remaining.  Returns True while the race is still
    active."""
    if not progress.get('race_active'):
        return False

    finalists = progress['race_finalists']
    race_iterations = progress.setdefault('race_iterations', {f: 0 for f in finalists})
    race_scores = progress.setdefault('race_scores', {f: float('inf') for f in finalists})
    race_attempts = progress.setdefault('race_best_attempts', {})
    budget = progress.get('race_budget', RACE_BUDGET)

    # The current iteration belongs to the current finalist.
    current = finalists[progress.get('race_current_idx', 0)]
    race_iterations[current] = race_iterations.get(current, 0) + 1

    # Track best score for this finalist.
    if composite_score < race_scores.get(current, float('inf')):
        race_scores[current] = composite_score
        race_attempts[current] = iteration

    # Rotate to next finalist with remaining budget.
    n = len(finalists)
    for _ in range(n):
        progress['race_current_idx'] = (progress.get('race_current_idx', 0) + 1) % n
        nxt = finalists[progress['race_current_idx']]
        if race_iterations.get(nxt, 0) < budget:
            break

    # Check if all finalists have exhausted their budget.
    all_done = all(race_iterations.get(f, 0) >= budget for f in finalists)
    if all_done:
        _resolve_race(progress)
        return False

    progress['race_iterations'] = race_iterations
    progress['race_scores'] = race_scores
    progress['race_best_attempts'] = race_attempts
    return True


def _resolve_race(progress):
    """Pick the race winner empirically and clean up race state."""
    race_scores = progress.get('race_scores', {})
    race_attempts = progress.get('race_best_attempts', {})
    if not race_scores:
        progress['race_active'] = False
        return

    winner_fam = min(race_scores, key=lambda f: race_scores[f])
    winner_attempt = race_attempts.get(winner_fam)
    best_score = race_scores[winner_fam]

    seed_scores = progress.get('seed_scores', {})
    raw_best = min(seed_scores.items(), key=lambda x: x[1])[0] if seed_scores else None

    progress['best_attempt'] = winner_attempt
    progress['best_score'] = best_score
    progress['seed_winner_family'] = winner_fam
    progress['seed_winner_tiebreak'] = True  # raced, not raw-picked
    progress['race_winner'] = winner_fam
    progress['race_active'] = False
    progress['_race_announced'] = False

    recommended = parse_target_field_str(
        progress.get('_partials_path'), 'recommended_primary_archetype',
    )
    progress['formant_forced'] = (
        winner_fam == 'formant_vocal' and recommended == 'formant_vocal'
        and winner_fam != raw_best
    )


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


def build_seeded_templates(partials, envelope=None):
    """Build architecture templates seeded with the target's actual partials.

    Falls back to the generic hard-coded templates when no partials are
    available so the system still behaves on targets without FluCoMa output.

    When ``envelope`` is provided (approach #8), the hardcoded
    ``Env.perc(0.01, 2.0)`` envelope is replaced with params matched to the
    target's measured amplitude envelope — attack, decay, release, sustain
    level — so the optimizer starts closer to the correct temporal shape.
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

    # --- envelope params for modulation-aware seeding (approach #8) ---
    if envelope:
        atk = max(0.001, envelope.get('attack_sec', 0.01))
        rel = max(0.3, envelope.get('release_sec', 2.0))
        sus = envelope.get('sustain_level', 0.5)
        if envelope.get('env_shape') == 'sustained':
            env_perc = f'Env([0, 1, {sus:.2f}, 0], [{atk:.3f}, {rel:.2f}, 0.1], [\\sus, \\step, -6])'
        else:
            env_perc = f'Env.perc({atk:.4f}, {rel:.2f})'
    else:
        atk, rel, sus = 0.01, 2.0, 0.5
        env_perc = f'Env.perc({atk:.2f}, {rel:.2f})'

    struck = (
        "var env, click, sig;\n"
        f"env = EnvGen.kr({env_perc}, doneAction: 2);\n"
        "click = Decay.ar(Impulse.ar(0), 0.002, ClipNoise.ar(0.05));\n"
        f"sig = Klank.ar(`[{_fmt_list(tfreqs)}, {_fmt_list(namps, '{:.3f}')}, "
        f"{_fmt_list(ringtimes, '{:.2f}')}], click);\n"
        "Out.ar(0, (sig * env * 0.3).dup);"
    )

    physical = (
        "var sig;\n"
        f"sig = Pluck.ar(WhiteNoise.ar(0.1), Impulse.ar(0), {fundamental:.1f}.reciprocal, "
        f"{fundamental:.1f}.reciprocal, 2.0, decoderCoeff: 0.5);\n"
        "Out.ar(0, (sig * 0.3).dup);"
    )

    fm = (
        "var env, sig, modFreq, modIndex;\n"
        f"env = EnvGen.kr({env_perc}, doneAction: 2);\n"
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
        f"env = EnvGen.kr({env_perc}, doneAction: 2);\n"
        "click = Decay.ar(Impulse.ar(0), 0.003, WhiteNoise.ar(0.1));\n"
        f"sig = Mix([{ringz_voices}]);\n"
        "Out.ar(0, (sig * env * 0.2).dup);"
    )

    form2 = tfreqs[1] if len(tfreqs) > 1 else fundamental * 2.5
    form3 = tfreqs[2] if len(tfreqs) > 2 else fundamental * 3.5

    granular = (
        "var env, sig, centerFreq;\n"
        f"env = EnvGen.kr({env_perc}, doneAction: 2);\n"
        f"centerFreq = {fundamental:.1f};\n"
        "sig = Mix(GrainSin.ar(2, Dust.ar(15), 0.08, centerFreq, 0, -1, 128));\n"
        "Out.ar(0, (sig * env * 0.3).dup);"
    )

    waveshaper = (
        "var env, sig, feedback;\n"
        f"env = EnvGen.kr({env_perc}, doneAction: 2);\n"
        "feedback = 0.5;\n"
        f"sig = SinOscFB.ar({fundamental:.1f}, feedback).tanh;\n"
        "Out.ar(0, (sig * env * 0.3).dup);"
    )

    subtractive = (
        "var env, osc, sig, cutoff, modCutoff;\n"
        f"env = EnvGen.kr({env_perc}, doneAction: 2);\n"
        "cutoff = 2000;\n"
        "modCutoff = cutoff * EnvGen.kr(Env.perc(0.05, 1.0));\n"
        f"osc = Mix(Saw.ar([{fundamental:.1f}, {fundamental * 1.007:.1f}]));\n"
        "sig = MoogFF.ar(osc, modCutoff, 2.5);\n"
        "Out.ar(0, (sig * env * 0.3).dup);"
    )

    chaos = (
        "var env, chaos, sig, resFreq;\n"
        f"env = EnvGen.kr({env_perc}, doneAction: 2);\n"
        f"resFreq = {fundamental:.1f};\n"
        f"chaos = Mix([Gendy1.ar(1, 1, 0.3, 0.3, {fundamental * 0.5:.0f}, "
        f"{fundamental * 2:.0f}, 0.5, 0.5, 12), "
        f"CuspL.ar({fundamental * 0.25:.1f}, 3, -3, 0.1) * 0.3]);\n"
        "sig = Resonz.ar(chaos, resFreq, 0.05);\n"
        "Out.ar(0, (sig * env * 0.25).dup);"
    )

    formant = (
        "var env, sig, fund;\n"
        f"env = EnvGen.kr({env_perc}, doneAction: 2);\n"
        f"fund = {fundamental:.1f};\n"
        f"sig = Formant.ar(fund, {form2:.1f}, fund * 0.2) + "
        f"Formant.ar(fund, {form3:.1f}, fund * 0.25);\n"
        "Out.ar(0, (sig * env * 0.25).dup);"
    )

    return {
        'struck_resonator': struck,
        'physical_model': physical,
        'fm_synthesis': fm,
        'resonator_bank': resonator,
        'granular': granular,
        'waveshaper_feedback': waveshaper,
        'subtractive': subtractive,
        'chaos_noise': chaos,
        'formant_vocal': formant,
    }


def load_audio(path, sr=44100):
    audio, _dur = load_and_preprocess(path, sr=sr, normalize=True, trim_silence=True)
    return audio


# Mapping from decomposition stem files to the hybrid-bus layer slot they inform.
STEM_SLOTS = {
    'sines.wav': 'sinusoidal',
    'harmonic.wav': 'sinusoidal',
    'residual.wav': 'residual',
    'percussive.wav': 'transient',
}


def score_components(attempt_path, stems_dir, sr=44100):
    """Score an attempt against each decomposition stem.

    Returns {slot: mfcc_distance} for slots whose stem exists, lower = the
    attempt reproduces that component better. Used to assign each hybrid-bus
    layer to the seed archetype that best matches its target component — so the
    seeds are treated as candidate layers, not tournament competitors.
    """
    if not stems_dir or not os.path.isdir(stems_dir):
        return {}
    evaluator = SynthesisEvaluator(sample_rate=sr)
    try:
        attempt_audio, _ = load_and_preprocess(
            attempt_path, sr=sr, normalize=True, trim_silence=True)
    except Exception:
        return {}
    if attempt_audio.size == 0:
        return {}

    scores = {}
    # Prefer sines.wav; fall back to harmonic.wav if sines absent.
    seen_slots = set()
    for fname in ['sines.wav', 'harmonic.wav', 'residual.wav', 'percussive.wav']:
        stem_path = os.path.join(stems_dir, fname)
        if not os.path.exists(stem_path):
            continue
        slot = STEM_SLOTS[fname]
        if slot in seen_slots:
            continue
        seen_slots.add(slot)
        try:
            stem_audio, _ = load_and_preprocess(
                stem_path, sr=sr, normalize=True, trim_silence=True)
        except Exception:
            continue
        conv = evaluator.compare_with_reference(
            attempt_audio, stem_audio, category_penalty=0.0)
        scores[slot] = float(conv.get('mfcc_distance', 1.0))
    return scores


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


def read_run_config(progress_dir):
    """Parse max_iterations, convergence_threshold, seed_count from config.txt."""
    config = {
        'max_iterations': 0,
        'convergence_threshold': 0.0,
        'seed_count': 0,
        'envelope_seed': True,
        'signal_chain_health': False,
    }
    if not progress_dir:
        return config
    config_path = os.path.join(progress_dir, 'config.txt')
    if not os.path.exists(config_path):
        return config
    try:
        for line in Path(config_path).read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, _, val = line.partition(':')
            key = key.strip()
            val = val.strip()
            if key == 'max_iterations':
                config['max_iterations'] = int(val)
            elif key == 'convergence_threshold':
                config['convergence_threshold'] = float(val)
            elif key == 'seed_count':
                config['seed_count'] = int(val)
            elif key == 'loss_config':
                val = val.strip()
                if val:
                    config['loss_config'] = val
            elif key in ('use_pnp', 'use_replay', 'use_sensitivity',
                         'use_neural_proxy', 'use_jtfs', 'envelope_seed',
                         'signal_chain_health'):
                config[key] = val.lower() in ('true', '1', 'yes', 'on')
    except (OSError, ValueError):
        pass
    return config


def update_progress(output_dir, iteration, composite_score, seeded_templates=None,
                    seed_count=0, arch=None, max_iterations=0, convergence_threshold=0.0,
                    partials_path=None, component_scores=None, spec_conv=None,
                    flatness_distance=None, noise_excess=None):
    """Update progress.json with score history, elitism, and plateau handling.

    Plateau detection is based on lack of a NEW best score over the last
    PLATEAU_PATIENCE iterations (a hill-climb stall), not on the raw score
    delta between consecutive — possibly noisy — attempts. When a plateau
    fires, the next untried architecture is selected, recorded in
    ``architectures_tried``, and a SWITCH_GRACE window is opened so the new
    architecture is not immediately declared a plateau too.

    When seed_count > 0 the first seed_count iterations form a seeding phase:
    - plateau detection and regression labels are suppressed
    - per-attempt architecture names are stored in attempt_architectures
    - at the final seed iteration the patience window and architectures_tried
      are both reset so the hill-climb phase starts clean
    """
    progress_path = os.path.join(output_dir, "progress.json")

    progress = {"scores": [], "best_score": None, "best_attempt": None,
                "plateau_detected": False, "architectures_tried": [],
                "iters_since_best": 0, "last_switch_iteration": 0,
                "switch_architecture": None, "regressed": False,
                "delta_vs_best": 0.0, "seed_count": 0,
                "attempt_architectures": {}, "seed_scores": {}}
    if os.path.exists(progress_path):
        try:
            with open(progress_path) as f:
                progress = json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass

    progress.setdefault('architectures_tried', [])
    progress.setdefault('last_switch_iteration', 0)
    progress.setdefault('attempt_architectures', {})
    progress.setdefault('seed_scores', {})
    progress.setdefault('component_scores', {})
    progress.setdefault('seed_count', seed_count)
    progress.setdefault('restarted_seeds', [])
    progress.setdefault('plateau_escapes', [])

    # Per-component stem scores (sinusoidal/residual/transient) for this seed.
    if component_scores:
        progress['component_scores'][str(iteration)] = component_scores

    # Keep seed_count up to date if caller provides it.
    if seed_count > 0:
        progress['seed_count'] = seed_count

    effective_seed_count = progress.get('seed_count', 0)
    in_seeding_phase = (effective_seed_count > 0 and iteration <= effective_seed_count)

    progress['scores'].append(composite_score)
    progress['iteration'] = iteration

    # Record per-attempt architecture if provided.
    if arch:
        progress['attempt_architectures'][str(iteration)] = arch
    if in_seeding_phase and arch:
        progress['seed_scores'][arch] = composite_score

    # During an active race, per-finalist best tracking is handled by
    # _advance_race — freeze the global best so a race iteration doesn't
    # accidentally get promoted before the race resolves.
    if progress.get('race_active'):
        is_new_best = False
    else:
        is_new_best = (
            progress['best_score'] is None
            or composite_score < progress['best_score'] - IMPROVEMENT_EPS
        )
    if is_new_best:
        progress['best_score'] = composite_score
        progress['best_attempt'] = iteration
        # Snapshot the best bus's noise excess so the plateau logic can tell
        # whether a stall is happening on an over-noisy optimum (don't add more
        # noise layers) vs a clean one (adding a layer is fine). Directional:
        # only excess (not deficit) blocks the add-a-layer escape.
        if noise_excess is not None:
            progress['best_noise_excess'] = noise_excess
        if spec_conv is not None:
            progress['best_spec_conv'] = spec_conv
        if flatness_distance is not None:
            progress['best_flatness_distance'] = flatness_distance

    progress['iters_since_best'] = iteration - progress['best_attempt']
    progress['delta_vs_best'] = composite_score - progress['best_score']
    progress['is_new_best'] = is_new_best

    # During the seeding phase, suppress regression labelling — seeds are
    # intentionally independent attempts, not mutations of each other.
    if in_seeding_phase:
        progress['regressed'] = False
    else:
        progress['regressed'] = (
            not is_new_best
            and progress['best_attempt'] != iteration
            and progress['delta_vs_best'] > IMPROVEMENT_EPS
        )

    # At the transition from seeding to hill-climb, prime the patience window
    # so a plateau cannot fire immediately, and mark all seed families as tried.
    if effective_seed_count > 0 and iteration == effective_seed_count:
        progress['last_switch_iteration'] = effective_seed_count
        seed_families_used = list(progress['attempt_architectures'].values())
        for fam in seed_families_used:
            if fam not in progress['architectures_tried'] and fam != 'flucoma_template':
                progress['architectures_tried'].append(fam)
        # ponytail: stash partials_path so _resolve_race can read the analysis
        # recommendation without threading it through every call.
        progress['_partials_path'] = partials_path
        # --- approach #1: loss multiplexing ---
        # Try different loss weightings on the seed results; pick the one
        # with the best score separation for this particular target.
        seed_scores = progress.get("seed_scores", {})
        if output_dir and seed_scores:
            try:
                ev = SynthesisEvaluator()
                tpath = os.path.join(output_dir, 'target.wav')
                taudio, _ = load_and_preprocess(tpath, sr=ev.sr,
                                                normalize=True, trim_silence=True)
                tcats = ev.categorize_metrics(ev.evaluate(taudio))
                attempts_data = []
                for att_str, _ in seed_scores.items():
                    # Find the attempt number for this family.
                    for astr, fam in progress.get('attempt_architectures', {}).items():
                        if fam == att_str:
                            apath = os.path.join(output_dir, f'attempt_{astr}.wav')
                            if os.path.exists(apath):
                                attempts_data.append((int(astr), apath, 0.0))
                            break
                if len(attempts_data) >= 2:
                    best_cfg = pick_loss_config(attempts_data, ev, taudio, tcats)
                    progress['loss_config'] = best_cfg
            except Exception:
                pass  # ponytail: non-critical, fall through to default
        # Try a short race among top finalists before crowning a winner.
        # Falls through to normal tiebreak when there aren't enough close seeds.
        if not _maybe_start_race(progress, partials_path):
            apply_seed_winner_tiebreak(progress, partials_path)
        progress['iters_since_best'] = iteration - progress['best_attempt']
        progress['delta_vs_best'] = (composite_score - progress['best_score']
                                     if progress['best_score'] is not None else 0.0)
        # Resolve the per-component hybrid layer assignment from seed stem scores.
        progress['layer_assignment'] = compute_layer_assignment(progress, partials_path)

    # Race phase management: during an active race, advance the race state and
    # suppress normal plateau detection (the race is short by design — each
    # finalist gets RACE_BUDGET iterations, so there isn't time to plateau).
    if progress.get('race_active') and not in_seeding_phase:
        _advance_race(progress, iteration, composite_score, arch)
        # After _advance_race, if race just resolved, sync the best-attempt
        # state for the next iteration's normal hill-climb tracking.
        if not progress.get('race_active'):
            progress['last_switch_iteration'] = iteration
            progress['iters_since_best'] = 0

    # Plateau detection is suppressed during seeding AND during the post-seed race.
    plateau = False
    if not in_seeding_phase and not progress.get('race_active'):
        grace_ok = (iteration - progress['last_switch_iteration']) > SWITCH_GRACE
        plateau = (
            len(progress['scores']) - effective_seed_count >= PLATEAU_PATIENCE
            and progress['iters_since_best'] >= PLATEAU_PATIENCE
            and grace_ok
        )

    progress['switch_architecture'] = None
    if plateau:
        # When all 9 architecture families have been tried as plateau escapes
        # and the bus still isn't converging (spectral_convergence > 1.0 or
        # prolonged stall), adding another layer won't help — the catalog is
        # exhausted. Instead, restart Phase B from the next-best seed family.
        if _all_architectures_exhausted(progress):
            restart = _pick_restart_seed(progress)
            if restart:
                restarted = progress.setdefault('restarted_seeds', [])
                if restart[0] not in restarted:
                    restarted.append(restart[0])
                progress['restart_seed_family'] = restart[0]
                progress['restart_seed_score'] = restart[1]
                progress['switch_architecture'] = '__restart__'
            # Fall through to normal arch switch if no restart candidates remain.
        if progress['switch_architecture'] is None:
            arch_switch = _next_untried_architecture(progress)
            progress['switch_architecture'] = arch_switch
            if arch_switch not in progress['architectures_tried']:
                progress['architectures_tried'].append(arch_switch)
            # Track plateau-escape layers separately from seed families.
            # ``architectures_tried`` is pre-populated with seed families at
            # the Phase A→B transition, but those were independent attempts,
            # not layers in the current bus.  ``plateau_escapes`` counts only
            # families added as hybrid layers during hill-climb plateaus.
            plateau_escapes = progress.setdefault('plateau_escapes', [])
            if arch_switch not in plateau_escapes:
                plateau_escapes.append(arch_switch)
        progress['last_switch_iteration'] = iteration

    progress['plateau_detected'] = plateau

    # ponytail: threshold convergence during Phase A would crown seed 1 and
    # skip the other 9 architecture basins — finish on threshold only after
    # seeding (and not mid-race).
    should_finish = (
        (max_iterations > 0 and iteration >= max_iterations)
        or (
            convergence_threshold > 0
            and composite_score < convergence_threshold
            and not in_seeding_phase
            and not progress.get('race_active')
        )
    )
    # If we're finishing mid-race, resolve it immediately so we have a valid
    # best_attempt for the finish workflow to use.
    if should_finish and progress.get('race_active'):
        _resolve_race(progress)
    progress['should_finish'] = should_finish
    progress['max_iterations'] = max_iterations
    progress['convergence_threshold'] = convergence_threshold

    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)

    return progress


def _read_attempt_core(filepath):
    """Best-effort extract the signal chain from an attempt .scd file.

    Returns (var_decls_line, signal_body, out_line) — the var declaration,
    everything between it and Out.ar, and the Out.ar line itself.
    Returns (None, None, None) when the file can't be parsed.
    """
    if not os.path.exists(filepath):
        return None, None, None
    with open(filepath) as f:
        lines = f.readlines()
    var_idx = None
    out_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('var ') and var_idx is None:
            var_idx = i
        if 'Out.ar' in line and out_idx is None:
            out_idx = i
    if var_idx is None:
        return None, None, None
    var_decls = lines[var_idx].strip()
    signal_body = ''.join(lines[var_idx + 1:out_idx]).rstrip()
    out_line = lines[out_idx].strip() if out_idx is not None else None
    return var_decls, signal_body, out_line


def _bus_skeleton(layer_assignment, next_attempt, output_dir=None):
    """Emit the Phase B decomposition-bus recipe the agent assembles.

    Each slot names the seed attempt whose signal core becomes that bus layer,
    gated by a per-layer gain @param. The agent lifts the core from its own
    validated attempt_N.scd (more reliable than re-embedding template strings).
    """
    slot_order = [('sinusoidal', 'layer1', 'g1'),
                  ('residual', 'layer2', 'g2'),
                  ('transient', 'layer3', 'g3')]
    present = [(s, l, g) for s, l, g in slot_order if s in layer_assignment]
    if not present:
        return ""
    lines = [f"=== PHASE B BUS SKELETON (build attempt_{next_attempt}.scd) ==="]
    lines.append("Assemble a decomposition bus: for each slot, lift the signal core")
    lines.append("(oscillator/resonator/noise block) from the named attempt, rename its")
    lines.append("final signal var to the layer var, STRIP its Out.ar line, gate with the")
    lines.append("gain @param. Merge ALL var declarations to the top. Keep doneAction:2")
    lines.append("on exactly ONE layer (the longest). Preserve each layer's own @param")
    lines.append("annotations so the ES can tune layer internals AND gains jointly.")
    lines.append("")
    for slot, layer, gain in present:
        info = layer_assignment[slot]
        lines.append(f"  {slot:<11} layer <- attempt_{info['attempt']} "
                     f"(family: {info['family']})  gate: {gain}  // @param 0.0 1.0")
    lines.append("")
    # Gather per-layer signal chains from the actual seed files.
    layer_cores = {}
    all_vars = set()
    if output_dir:
        for slot, layer, _ in present:
            info = layer_assignment[slot]
            fpath = os.path.join(output_dir, f"attempt_{info['attempt']}.scd")
            var_decls, signal_body, _out_line = _read_attempt_core(fpath)
            if var_decls and signal_body:
                layer_cores[slot] = (info, var_decls, signal_body)
                # Collect var names for the merged declaration.
                for name in var_decls.replace('var ', '').replace(';', '').split(','):
                    name = name.strip()
                    if name and name != 'sig':
                        all_vars.add(name)
            else:
                layer_cores[slot] = (info, None, None)

    merged_vars = ', '.join(sorted(all_vars))
    gains = ', '.join(g for _, _, g in present)
    layers = ', '.join(l for _, l, _ in present)
    if merged_vars:
        lines.append(f"var {gains}, sig, {merged_vars};"
                     if not layers else f"var {gains}, {layers}, sig, {merged_vars};")
    else:
        lines.append(f"var {gains}, {layers}, sig;")
    for _, _, g in present:
        lines.append(f"{g} = 0.4; // @param 0.0 1.0")
    for slot, layer, _ in present:
        info = layer_assignment[slot]
        core = layer_cores.get(slot)
        if core and core[1] is not None:
            _info, _var_decls, signal_body = core
            lines.append(f"// --- {slot} layer: {info['family']} "
                         f"(from attempt_{info['attempt']}) ---")
            lines.append(signal_body)
            lines.append(f"{layer} = sig; // rename signal core to bus layer")
            lines.append("")
        else:
            lines.append(f"// {slot} layer: {info['family']} "
                         f"(attempt_{info['attempt']}.scd not found — assemble manually)")
            lines.append(f"{layer} = <signal core of attempt_{info['attempt']}>;")
    terms = ' + '.join(f"({l} * {g})" for _, l, g in present)
    lines.append(f"sig = {terms};")
    lines.append("Out.ar(0, (sig * 0.4).dup);")
    lines.append("")
    lines.append(f"After writing attempt_{next_attempt}.scd, run the FULL optimizer on it")
    lines.append(f"(budget from config.txt) — the ES tunes the layer gains jointly with")
    lines.append("each layer's internal params. This is the hybrid start; the hill-climb")
    lines.append("then refines the bus (swap/fill one slot per iteration).")
    return "\n".join(lines)


def _format_layer_assignment_section(progress, output_dir, signal_chain_health,
                                     next_attempt, component_scores_iteration):
    """COMPONENT LAYER ASSIGNMENT + bus skeleton + optional signal-chain health."""
    layer_assignment = progress.get('layer_assignment', {})
    if not layer_assignment:
        return []
    lines = [
        "",
        "=== COMPONENT LAYER ASSIGNMENT (hybrid bus) ===",
        "Each decomposition slot -> the seed archetype that best",
        "matches that target component. Build the Phase B bus from these.",
    ]
    for slot in ['sinusoidal', 'residual', 'transient']:
        info = layer_assignment.get(slot)
        if not info:
            continue
        lines.append(
            f"  {slot}: attempt {info['attempt']} "
            f"(family: {info['family']}, component_score: {info['score']:.4f})"
        )
    lines.append(
        "Phase B: sum these layers with a per-layer gain @param "
        "(e.g. sig = sinusoidalLayer*g1 + residualLayer*g2 + transientLayer*g3)."
    )
    lines.append("")
    skeleton = _bus_skeleton(layer_assignment, next_attempt, output_dir)
    if skeleton:
        lines.append(skeleton)
    component_scores = progress.get('component_scores', {})
    last_comp = component_scores.get(str(component_scores_iteration), {})
    if signal_chain_health and last_comp:
        lines.append("")
        lines.append("=== SIGNAL-CHAIN LAYER HEALTH (approach #7) ===")
        lines.append("Per-stem score of THIS attempt's layers vs decomposition:")
        for slot in ['sinusoidal', 'residual', 'transient']:
            score_val = last_comp.get(slot)
            info = layer_assignment.get(slot, {})
            if score_val is not None:
                status = 'OK' if score_val < 0.5 else 'WEAK' if score_val < 1.0 else 'POOR'
                lines.append(f"  {slot}: mfcc_dist={score_val:.3f} [{status}] "
                             f"(layer: {info.get('family', '?')})")
            elif slot in layer_assignment:
                lines.append(f"  {slot}: no data (layer: {info.get('family', '?')})")
        lines.append("POOR layers are being carried by the sum — consider replacing them.")
    return lines


# ============================================================================
# Approach #1: Loss multiplexing — try different weight vectors, pick the best
# ============================================================================
# Salimi et al. 2025: no universal best loss. The optimal weighting depends on
# the synthesizer AND the target. During seeding, we compute the score spread
# (best vs worst) under 3 alternative weight vectors. The one with the widest
# normalized spread is likely the most discriminating for this particular
# target, so we use it for the hill-climb phase.

_LOSS_CONFIGS = LOSS_CONFIGS  # alias for pick_loss_config / eval_loss_config


def resolve_loss_config(progress_dir=None, cli_override=None):
    """CLI override > progress.json loss_config > None (built-in default)."""
    if cli_override:
        return cli_override if cli_override in LOSS_CONFIGS else 'default'
    if progress_dir:
        ppath = os.path.join(progress_dir, 'progress.json')
        if os.path.exists(ppath):
            try:
                progress = json.loads(Path(ppath).read_text(encoding='utf-8'))
                cfg = progress.get('loss_config')
                if cfg in LOSS_CONFIGS:
                    return cfg
            except (OSError, json.JSONDecodeError):
                pass
    return None


def apply_loss_config(name):
    """Set module-level active loss config (None = built-in default weights)."""
    sev._ACTIVE_LOSS_CONFIG = name if name in LOSS_CONFIGS else None


def eval_loss_config(config_name, evaluator, attempt_audio, target_audio,
                     target_categories, attempt_categories):
    """Compute composite score under a specific loss weight configuration."""
    prev = sev._ACTIVE_LOSS_CONFIG
    try:
        apply_loss_config(config_name)
        penalty = compute_category_penalty(
            evaluator, target_categories, attempt_categories)
        conv = evaluator.compare_with_reference(
            attempt_audio, target_audio, category_penalty=penalty)
        return float(conv['composite_score']), conv
    finally:
        sev._ACTIVE_LOSS_CONFIG = prev


def pick_loss_config(attempts_data, evaluator, target_audio, target_categories):
    """Try all loss configs on seed results; pick the one with best separation.

    ``attempts_data`` is a list of (attempt_id, audio_path, score) tuples from
    seeding. Returns the config name with the widest normalized score spread.
    """
    if len(attempts_data) < 2:
        return 'default'
    best_config = 'default'
    best_spread = -1.0
    for cfg_name in _LOSS_CONFIGS:
        scores = []
        for _aid, apath, _old_score in attempts_data:
            try:
                aaudio, _ = load_and_preprocess(apath, sr=evaluator.sr,
                                                normalize=True, trim_silence=True)
            except Exception:
                continue
            if aaudio.size == 0:
                continue
            attempt_metrics = evaluator.evaluate(aaudio)
            attempt_categories = evaluator.categorize_metrics(attempt_metrics)
            sc, _conv = eval_loss_config(
                cfg_name, evaluator, aaudio, target_audio,
                target_categories, attempt_categories)
            scores.append(sc)
        if len(scores) < 2:
            continue
        spread = (max(scores) - min(scores)) / (np.mean(scores) + 1e-9)
        if spread > best_spread:
            best_spread = spread
            best_config = cfg_name
    return best_config


def compute_layer_assignment(progress, partials_path=None):
    """Pick the best seed archetype for each hybrid-bus slot.

    For each slot (sinusoidal/residual/transient), find the seed iteration
    whose per-component score is lowest and resolve its architecture family.
    Returns {slot: {'attempt': int, 'family': str, 'score': float}} for slots
    with data. Seeds are treated as candidate layers, not competitors.

    Formant override: if the analysis recommends formant_vocal (strong formants
    + pitched source), force the formant_vocal seed into the sinusoidal/body
    slot — additive sines cannot reproduce a formant spectrum, so the per-stem
    score (which may favor sines on the sines-stem) must not strand the vocal
    model out of the bus.
    """
    component_scores = progress.get('component_scores', {})
    arch_map = progress.get('attempt_architectures', {})
    if not component_scores:
        return {}
    slots = set()
    for scores in component_scores.values():
        slots.update(scores.keys())
    assignment = {}
    for slot in slots:
        best = None
        for attempt_str, scores in component_scores.items():
            if slot not in scores:
                continue
            sc = scores[slot]
            if best is None or sc < best[2]:
                best = (int(attempt_str), arch_map.get(attempt_str, 'unknown'), sc)
        if best is not None:
            assignment[slot] = {'attempt': best[0], 'family': best[1], 'score': best[2]}

    # Formant-driven override for the body slot — only when the formant seed's
    # per-stem score is competitive with the best sinusoidal score.  Don't
    # force a last-place formant into the body slot (ponytail: same threshold
    # as pick_seed_winner / _maybe_start_race).
    recommended = parse_target_field_str(partials_path, 'recommended_primary_archetype')
    if recommended == 'formant_vocal':
        for attempt_str, fam in arch_map.items():
            if fam == 'formant_vocal':
                prev = assignment.get('sinusoidal')
                formant_sin_score = component_scores.get(attempt_str, {}).get('sinusoidal')
                if (prev is not None and formant_sin_score is not None
                        and formant_sin_score > prev['score'] + FORMANT_FORCE_MAX_GAP):
                    break  # ponytail: formant stem too far behind — trust per-stem winner
                assignment['sinusoidal'] = {
                    'attempt': int(attempt_str),
                    'family': 'formant_vocal',
                    'score': formant_sin_score if formant_sin_score is not None else (prev['score'] if prev else 0.0),
                }
                assignment['_formant_override'] = True
                break
    return assignment


def _next_untried_architecture(progress):
    """Pick the next architecture family to try on a plateau.

    When seed scores are available, prefer the best-scoring seed family that
    has not yet been developed (lowest score = closest to target).  This makes
    the plateau switch data-driven rather than relying on a fixed preference
    list.  Falls back to ARCHITECTURE_ORDER when no seed data is present.
    """
    tried = set(progress.get('architectures_tried', []))
    seed_scores = progress.get('seed_scores', {})

    if seed_scores:
        # Filter to families that are in ARCHITECTURE_ORDER (i.e. not
        # 'flucoma_template' which is the hill-climb default) and not tried.
        candidates = [
            (score, family)
            for family, score in seed_scores.items()
            if family in ARCHITECTURE_ORDER and family not in tried
        ]
        if candidates:
            candidates.sort()  # ascending score = best first
            return candidates[0][1]

    for arch in ARCHITECTURE_ORDER:
        if arch not in tried:
            return arch
    return ARCHITECTURE_ORDER[0]


def _detect_noise_contradiction(mismatches, noise_excess, noise_deficit):
    """Detect when the category-based noise label contradicts the band-wise
    flatness signal, indicating noise is in the WRONG spectral region rather
    than simply too much or too little.

    Returns a dict {category_label, direction, instruction} or None.
    """
    if not mismatches:
        return None
    for cat_name, t_label, c_label, _suggestion, _dist in mismatches:
        if cat_name != 'harmonic_to_noise_ratio':
            continue
        direction, _ = get_category_direction(cat_name, t_label, c_label)
        # Category says attempt is NOISIER than target (label index higher).
        # But noise_deficit says overall MORE TONAL — noise is concentrated in
        # the mid band while missing from low/high where it matters more.
        if direction == 'higher' and noise_deficit > 0.08:
            return {
                'type': 'noisy_category_tonal_overall',
                'cat_label': c_label,
                'target_label': t_label,
                'instruction': (
                    f"Category says '{c_label}' but band-wise analysis says more "
                    f"TONAL — noise is in the WRONG spectral region. The mid-band "
                    f"(500-2000Hz) has excess noise, but the low/high bands lack "
                    f"broadband energy. RESHAPE, don't just reduce or add: "
                    f"move noise energy from mid-band to low-band (LPF below 500Hz) "
                    f"and/or high-band (HPF above 2000Hz). Replace mid-band noise "
                    f"sources with tonal oscillators."
                ),
            }
        # Category says attempt is MORE TONAL than target.
        # But noise_excess says overall NOISIER — wrong-shape noise in low/high
        # bands while the mid-band is too tonal.
        if direction == 'lower' and noise_excess > 0.08:
            return {
                'type': 'tonal_category_noisy_overall',
                'cat_label': c_label,
                'target_label': t_label,
                'instruction': (
                    f"Category says '{c_label}' but band-wise analysis says more "
                    f"NOISY — noise is in the WRONG spectral region. The low/high "
                    f"bands have excess noise, but the mid-band lacks the target's "
                    f"harmonic density. RESHAPE: filter/attenuate the low/high noise "
                    f"and add tonal/harmonic content in the mid-band."
                ),
            }
    return None


def _pick_restart_seed(progress):
    """Pick the best-scoring seed family that is NOT the current winner and
    NOT already restarted. Returns (family_name, seed_score) or None."""
    seed_scores = progress.get('seed_scores', {})
    restarted = set(progress.get('restarted_seeds', []))
    winner = progress.get('seed_winner_family')
    if not seed_scores:
        return None
    candidates = [
        (score, fam) for fam, score in seed_scores.items()
        if fam != winner and fam not in restarted
    ]
    if not candidates:
        return None
    candidates.sort()
    return (candidates[0][1], candidates[0][0])


def _all_architectures_exhausted(progress):
    """Return True when every family in ARCHITECTURE_ORDER has been tried
    as a plateau-escape LAYER (not just as a seed), and the best score is
    still stuck (spec_conv > 1.0 or iters_since_best exceeds 2x the normal
    patience).

    Uses ``plateau_escapes`` (families explicitly added as hybrid layers
    during hill-climb plateaus), NOT ``architectures_tried`` which also
    includes seed families from Phase A — seeds were independent attempts,
    not layers in the current bus.
    """
    plateau_escapes = set(progress.get('plateau_escapes', []))
    if len(plateau_escapes) < len(ARCHITECTURE_ORDER):
        return False
    # Only declare exhaustion when the bus is genuinely not converging: either
    # the spectra are still nearly orthogonal or we've been stuck for a while
    # after the last family was tried.
    spec_conv = progress.get('best_spec_conv', 0.0)
    if spec_conv > 1.0:
        return True
    return progress.get('iters_since_best', 0) >= PLATEAU_PATIENCE + 2


def _step_phase_label(progress, iteration):
    """Human-readable phase for step N in reports."""
    if not progress:
        return 'run'
    seed_count = progress.get('seed_count', 0)
    if seed_count > 0 and iteration <= seed_count:
        return 'seeding'
    if progress.get('race_active'):
        return 'race'
    if seed_count > 0 and iteration == seed_count + 1:
        return 'bus'
    return 'hill-climb'


def _poor_layer_hints(progress, iteration):
    """Slots marked POOR in signal-chain health (for plateau hints)."""
    layer_assignment = progress.get('layer_assignment', {}) or {}
    last_comp = (progress.get('component_scores') or {}).get(str(iteration), {})
    hints = []
    for slot in ('sinusoidal', 'residual', 'transient'):
        score_val = last_comp.get(slot)
        if score_val is not None and score_val >= 1.0:
            fam = layer_assignment.get(slot, {}).get('family', '?')
            hints.append(f"{slot} ({fam}, mfcc_dist={score_val:.3f})")
    return hints


def run_self_check(progress_dir, sr=44100):
    """P3: verify loss configs discriminate and pick_loss_config is consistent."""
    progress_path = os.path.join(progress_dir, 'progress.json')
    if not os.path.exists(progress_path):
        print(f"self-check: no progress.json in {progress_dir}", file=sys.stderr)
        return 1
    progress = json.loads(Path(progress_path).read_text(encoding='utf-8'))
    seed_scores = progress.get('seed_scores', {})
    arch_map = progress.get('attempt_architectures', {})
    attempts_data = []
    for att_str, fam in arch_map.items():
        if int(att_str) > progress.get('seed_count', 0):
            continue
        apath = os.path.join(progress_dir, f'attempt_{att_str}.wav')
        if os.path.exists(apath):
            attempts_data.append((int(att_str), apath, seed_scores.get(fam, 0.0)))
    if len(attempts_data) < 2:
        print("self-check: skip (need >=2 seed attempts)")
        return 0
    tpath = os.path.join(progress_dir, 'target.wav')
    if not os.path.exists(tpath):
        print(f"self-check: missing {tpath}", file=sys.stderr)
        return 1
    ev = SynthesisEvaluator(sample_rate=sr)
    taudio, _ = load_and_preprocess(tpath, sr=sr, normalize=True, trim_silence=True)
    tcats = ev.categorize_metrics(ev.evaluate(taudio))
    spreads = {}
    for cfg_name in LOSS_CONFIGS:
        scores = []
        for _aid, apath, _ in attempts_data:
            try:
                aaudio, _ = load_and_preprocess(apath, sr=sr, normalize=True, trim_silence=True)
            except Exception:
                continue
            if aaudio.size == 0:
                continue
            am = ev.evaluate(aaudio)
            ac = ev.categorize_metrics(am)
            sc, _ = eval_loss_config(cfg_name, ev, aaudio, taudio, tcats, ac)
            scores.append(sc)
        if len(scores) >= 2:
            spreads[cfg_name] = (max(scores) - min(scores)) / (np.mean(scores) + 1e-9)
    if len(spreads) < 2:
        print("self-check: could not score enough seeds")
        return 1
    if max(spreads.values()) - min(spreads.values()) < 1e-6:
        print("self-check FAIL: all loss configs produce identical spreads", file=sys.stderr)
        return 1
    picked = pick_loss_config(attempts_data, ev, taudio, tcats)
    best_name = max(spreads, key=spreads.get)
    if picked != best_name:
        print(f"self-check FAIL: pick_loss_config={picked} != widest spread={best_name}",
              file=sys.stderr)
        return 1
    print(f"self-check OK: loss configs discriminate; selected={picked}")
    return 0


def run_self_check_race_report():
    """Verify RACE FINISHED emits once and includes layer assignment."""
    progress = {
        'race_winner': 'flucoma_template',
        'race_active': False,
        '_race_announced': False,
        'race_scores': {'flucoma_template': 0.526, 'fm_synthesis': 0.567},
        'best_score': 0.526,
        'best_attempt': 13,
        'layer_assignment': {
            'sinusoidal': {'attempt': 1, 'family': 'flucoma_template', 'score': 0.301},
            'residual': {'attempt': 4, 'family': 'fm_synthesis', 'score': 0.601},
        },
        'iteration': 19,
        'seed_count': 10,
        'scores': [0.5] * 19,
    }
    conv = {'composite_score': 0.534}
    r1 = format_report(conv, [], [], progress=progress, output_dir=None)
    assert 'RACE FINISHED' in r1, 'first call should announce race'
    assert 'COMPONENT LAYER ASSIGNMENT' in r1, 'layer block missing on race finish'
    assert 'attempt_20.scd' in r1, 'bus skeleton should target iteration+1'
    assert progress.get('_race_announced'), 'flag should flip after first emit'
    r2 = format_report(conv, [], [], progress=progress, output_dir=None)
    assert 'RACE FINISHED' not in r2, 'second call should not re-announce'
    print('self-check-race-report: ok')
    return 0


def format_report(convergence, mismatches, top_deltas, prev_code=None,
                  progress=None, best_code=None, seeded_templates=None,
                  max_iterations=0, convergence_threshold=0.0, partials_path=None,
                  signal_chain_health=False, output_dir=None):
    lines = []

    composite = convergence.get('composite_score', convergence.get('spectral_convergence', 0))
    lines.append("=== CONVERGENCE METRICS ===")
    lines.append(f"composite_score: {composite:.4f}")
    lines.append(f"mfcc_distance: {convergence.get('mfcc_distance', 0):.4f}  (primary perceptual term)")
    lines.append(f"mel_distance: {convergence.get('mel_distance', 0):.4f}")
    lines.append(f"spectral_convergence: {convergence.get('spectral_convergence', 0):.4f}")
    lines.append(f"log_spectral_distance: {convergence.get('log_spectral_distance', 0):.4f}")
    lines.append(f"envelope_distance: {convergence.get('envelope_distance', 0):.4f}")
    if 'onset_max_penalty' in convergence:
        lines.append(f"onset_max_penalty: {convergence.get('onset_max_penalty', 0):.4f}")
    lines.append(f"snr_db: {convergence.get('snr_db', 0):.2f}")
    lines.append(f"rmse: {convergence.get('rmse', 0):.6f}")
    lines.append("")

    iteration = progress.get('iteration', 0) if progress else 0
    if progress:
        max_iterations = progress.get('max_iterations', max_iterations)
        convergence_threshold = progress.get('convergence_threshold', convergence_threshold)
        phase = _step_phase_label(progress, iteration)
        if max_iterations > 0 and iteration > 0:
            lines.insert(0, f"=== STEP {iteration}/{max_iterations} ({phase}) ===")
            lines.insert(1, "")

    should_finish = progress.get('should_finish', False) if progress else (
        (max_iterations > 0 and iteration >= max_iterations)
        or (convergence_threshold > 0 and composite < convergence_threshold)
    )

    if progress and should_finish:
        # If a race is still active, resolve it immediately so we have a winner.
        if progress.get('race_active'):
            _resolve_race(progress)
        best_attempt = progress.get('best_attempt')
        lines.append("=== MANDATORY FINISH ===")
        if max_iterations > 0 and iteration >= max_iterations:
            lines.append(
                f"Step budget exhausted: N={iteration}, max_iterations={max_iterations}."
            )
            lines.append(f"Do NOT write attempt_{iteration + 1}.scd or any further attempts.")
        elif convergence_threshold > 0 and composite < convergence_threshold:
            lines.append(
                f"Convergence reached: composite_score={composite:.4f} < "
                f"threshold={convergence_threshold:.4f}."
            )
        lines.append("Go to the Finish section in AGENTS.md NOW:")
        lines.append(f"  1. cp current_run/attempt_{best_attempt}.scd current_run/final_result.scd")
        lines.append("  2. write current_run/report.md")
        lines.append("")

    if progress:
        scores = progress.get('scores', [])
        best_attempt = progress.get('best_attempt')
        best_score = progress.get('best_score')
        effective_seed_count = progress.get('seed_count', 0)
        iteration = progress.get('iteration', len(scores))
        in_seeding_phase = (effective_seed_count > 0 and iteration <= effective_seed_count)
        is_final_seed = (effective_seed_count > 0 and iteration == effective_seed_count)
        arch_map = progress.get('attempt_architectures', {})
        seed_scores = progress.get('seed_scores', {})

        lines.append("=== SCORE HISTORY (lower is better) ===")
        history = []
        for i, s in enumerate(scores, 1):
            fam = arch_map.get(str(i), '')
            fam_str = f' [{fam}]' if fam else ''
            mark = ' <-- BEST' if i == best_attempt else ''
            seed_mark = ' [SEED]' if i <= effective_seed_count else ''
            history.append(f"  attempt {i}: {s:.4f}{fam_str}{seed_mark}{mark}")
        lines.extend(history)
        lines.append("")

        if in_seeding_phase:
            lines.append("=== SEEDING PHASE STATUS ===")
            lines.append(
                f"Seed {iteration}/{effective_seed_count} evaluated "
                f"(family: {arch_map.get(str(iteration), 'unknown')}, "
                f"score: {composite:.4f})."
            )
            if (convergence_threshold > 0 and composite < convergence_threshold
                    and iteration < effective_seed_count):
                lines.append(
                    f"(Score below threshold {convergence_threshold} — seeding "
                    f"continues; {effective_seed_count - iteration} seed(s) remain.)"
                )
            if is_final_seed:
                if progress.get('race_active'):
                    # Race is active — don't crown a winner yet.
                    # Emit race instructions instead of normal Phase B.
                    finalists = progress.get('race_finalists', [])
                    race_iterations = progress.get('race_iterations', {})
                    budget = progress.get('race_budget', RACE_BUDGET)
                    current = finalists[progress.get('race_current_idx', 0)] if finalists else '?'
                    race_scores = progress.get('race_scores', {})
                    race_attempts = progress.get('race_best_attempts', {})

                    lines.append("")
                    lines.append("=== POST-SEED RACE (empirical finalist playoff) ===")
                    lines.append(
                        f"Instead of crowning one seed winner, racing top "
                        f"{len(finalists)} finalists for {budget} iterations each "
                        f"(Shier 2021: short warm-started runs predict full-run "
                        f"convergence at ~1/10 the cost)."
                    )
                    lines.append("")
                    lines.append(f"  Finalists: {', '.join(finalists)}")
                    for f in finalists:
                        done = race_iterations.get(f, 0)
                        best_s = race_scores.get(f, float('inf'))
                        att = race_attempts.get(f, '?')
                        bar = '█' * done + '░' * (budget - done)
                        lines.append(
                            f"  {f}: [{bar}] {done}/{budget}  "
                            f"(best: {best_s:.4f} @ attempt {att})"
                        )
                    lines.append("")
                    lines.append(
                        f"NEXT: Work on the CURRENT finalist [{current}] for ONE step. "
                        f"Read its seed template from current_run/seed_templates.txt "
                        f"(or reuse its last attempt), run ONE optimizer cycle, then "
                        f"re-evaluate. The race rotates through finalists round-robin."
                    )
                    # Show seed scores as reference.
                    if seed_scores:
                        ranked = sorted(seed_scores.items(), key=lambda x: x[1])
                        lines.append("")
                        lines.append("Seed ranking (reference):")
                        for rank, (fam, sc) in enumerate(ranked, 1):
                            mark = ' ← RACE' if fam in finalists else ''
                            lines.append(f"  {rank}. {fam}: {sc:.4f}{mark}")
                else:
                    # No race — normal winner announcement.
                    winner_fam = progress.get('seed_winner_family') or arch_map.get(
                        str(best_attempt), 'unknown'
                    )
                    lines.append("")
                    lines.append(
                        f"ALL SEEDS EVALUATED. WINNER: attempt {best_attempt} "
                        f"(family: {winner_fam}, score: {best_score:.4f})."
                    )
                    if progress.get('formant_forced'):
                        raw_best = min(seed_scores.items(), key=lambda x: x[1])
                        lines.append(
                            f"(FORMANT PROMOTION: {winner_fam} forced as Phase B base over raw-best "
                            f"{raw_best[0]} ({raw_best[1]:.4f}). The target has strong formants + a "
                            "pitched source, so additive sines — even if they seed-score better — "
                            "cannot reproduce it. Develop the formant_vocal body; do NOT fall back to "
                            "flucoma_template.)"
                        )
                    elif progress.get('seed_winner_tiebreak'):
                        raw_best = min(seed_scores.items(), key=lambda x: x[1])
                        lines.append(
                            f"(Tiebreak: {winner_fam} preferred over {raw_best[0]} "
                            f"({raw_best[1]:.4f}) — scores within {SEED_TIEBREAK_EPS})."
                        )
                    lines.append(
                        "NEXT STEP (Phase B): Re-run optimize_params.py on "
                        f"attempt_{best_attempt}.scd with the FULL optimizer_budget "
                        f"(from config.txt), then continue the hill-climb from that "
                        "attempt as your BASE CODE."
                    )
                    if seed_scores:
                        ranked = sorted(seed_scores.items(), key=lambda x: x[1])
                        lines.append("Seed ranking (best to worst):")
                        for rank, (fam, sc) in enumerate(ranked, 1):
                            lines.append(f"  {rank}. {fam}: {sc:.4f}")

                    # --- approach #1: loss config selected for hill-climb ---
                    loss_cfg = progress.get('loss_config')
                    if loss_cfg and loss_cfg != 'default':
                        lines.append(
                            f"Loss config selected: {loss_cfg} "
                            f"(best score separation for this target)."
                        )

                    lines.extend(_format_layer_assignment_section(
                        progress, output_dir, signal_chain_health,
                        effective_seed_count + 1, iteration,
                    ))
            else:
                next_seed_idx = iteration + 1
                next_fam = (SEED_FAMILIES[next_seed_idx - 1]
                            if next_seed_idx <= len(SEED_FAMILIES)
                            else 'any untried family')
                lines.append(
                    f"Write seed {next_seed_idx}/{effective_seed_count} using "
                    f"architecture family: {next_fam}. "
                    "Do NOT copy or mutate any previous seed. "
                    "Use the DOMINANT PARTIALS from target_partials.txt to seed "
                    "frequencies. Add 3-8 // @param annotations."
                )
            lines.append("")
        elif progress.get('race_active'):
            # Race in progress — tell the agent which finalist to work on.
            finalists = progress.get('race_finalists', [])
            race_iterations = progress.get('race_iterations', {})
            race_scores = progress.get('race_scores', {})
            budget = progress.get('race_budget', RACE_BUDGET)
            current = finalists[progress.get('race_current_idx', 0)] if finalists else '?'
            current_attempt = progress.get('race_best_attempts', {}).get(current)

            lines.append("=== RACE IN PROGRESS ===")
            for f in finalists:
                done = race_iterations.get(f, 0)
                best_s = race_scores.get(f, float('inf'))
                bar = '█' * done + '░' * (budget - done)
                lines.append(f"  {f}: [{bar}] {done}/{budget}  (best: {best_s:.4f})")
            lines.append("")
            lines.append(
                f"CURRENT: {current} ({race_iterations.get(current, 0)}/{budget} done). "
                f"Improve it for ONE step, then re-evaluate."
            )
            if current_attempt:
                lines.append(
                    f"  Base: attempt_{current_attempt}.scd — read it, apply ONE "
                    f"targeted change (tune frequencies/bandwidths/balance), "
                    f"write attempt_{iteration + 1}.scd."
                )
            else:
                lines.append(
                    f"  Base: seed template from current_run/seed_templates.txt "
                    f"(family: {current}) — write a fresh attempt_{iteration + 1}.scd "
                    f"with 3-8 // @param annotations."
                )
            lines.append(
                "  Run optimizer for ONE budget cycle, then re-evaluate. "
                "The race rotates finalists automatically."
            )
        elif (progress.get('race_winner') and not progress.get('race_active')
              and not progress.get('_race_announced')):
            # Race just resolved — announce the empirical winner (once).
            winner_fam = progress['race_winner']
            winner_score = progress.get('best_score', 0)
            winner_attempt = progress.get('best_attempt')
            race_scores = progress.get('race_scores', {})
            lines.append("=== RACE FINISHED — EMPIRICAL WINNER ===")
            for f, s in sorted(race_scores.items(), key=lambda x: x[1]):
                mark = ' ← WINNER' if f == winner_fam else ''
                lines.append(f"  {f}: {s:.4f}{mark}")
            lines.append("")
            lines.append(
                f"WINNER: {winner_fam} (attempt {winner_attempt}, score {winner_score:.4f}). "
                f"Other finalists archived — seed templates remain available for bus layer assembly."
            )
            lines.append(
                f"NEXT STEP (Phase B): Re-run optimize_params.py on "
                f"attempt_{winner_attempt}.scd with the FULL optimizer_budget "
                f"(from config.txt), then continue the hill-climb from that "
                "attempt as your BASE CODE."
            )
            loss_cfg = progress.get('loss_config')
            if loss_cfg and loss_cfg != 'default':
                lines.append(
                    f"Loss config selected: {loss_cfg} "
                    f"(best score separation for this target)."
                )
            lines.extend(_format_layer_assignment_section(
                progress, output_dir, signal_chain_health,
                iteration + 1, effective_seed_count,
            ))
            progress['_race_announced'] = True
        elif not should_finish:
            lines.append("=== NEXT-ATTEMPT INSTRUCTION (HILL-CLIMB) ===")
            if progress.get('plateau_detected'):
                lines.append(
                    f"No new best for {progress.get('iters_since_best', 0)} iterations. "
                    "A MANDATORY ADD-A-LAYER move is required (see section below)."
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

    # Directional noise warning. spectral_convergence > 1.0 only means the
    # spectra differ — it is NOT noise-specific (missing formants/brightness read
    # the same). The noise-specific, DIRECTIONAL signal is noise_excess (attempt
    # flatter/noisier than target) vs noise_deficit (attempt more tonal/thin).
    # Warn "reduce noise" only when the attempt is genuinely noisier than the
    # target; warn "too tonal" in the opposite case. This avoids false-positive
    # reduce-noise nagging on targets that are themselves noisy (vocals, breath).
    noise_excess = convergence.get('noise_excess', 0.0)
    noise_deficit = convergence.get('noise_deficit', 0.0)
    f0 = parse_target_field(partials_path, 'fundamental_freq')
    resid_cent = parse_target_field(partials_path, 'residual_spectral_centroid')

    # Detect when the harmonic_to_noise_ratio category label contradicts the
    # band-wise flatness signal. This means noise is in the WRONG spectral
    # region — e.g. concentrated in mid-band (triggering a "noisy" category)
    # but missing from low/high (triggering a "too tonal" deficit). The fix
    # is RESHAPE, not just add or reduce.
    noise_contradiction = _detect_noise_contradiction(
        mismatches, noise_excess, noise_deficit
    )

    if noise_contradiction:
        lines.append("=== RESHAPE NOISE (contradiction resolved) ===")
        lines.append(noise_contradiction['instruction'])
        lines.append("")
        lines.append(
            "IGNORE conflicting advice below: the category mismatch says "
            f"'{noise_contradiction['cat_label']}' and the band-wise warning "
            "may suggest the opposite. Follow the RESHAPE instruction above — "
            "it reconciles both signals."
        )
        lines.append("")
    elif noise_excess > 0.12:
        lines.append("=== OVER-NOISE WARNING (high priority) ===")
        lines.append(
            f"noise_excess={noise_excess:.3f}: the attempt is FLATTER (noisier) than the "
            "target per band — a noise/chaos layer is likely substituting for tonal content "
            "the synth is missing (often a weak/missing fundamental)."
        )
        parts = ["REDUCE noise/chaos amplitudes"]
        if resid_cent:
            parts.append(f"reshape the residual toward target centroid {resid_cent:.0f} Hz (HPF/BPF, not low BrownNoise)")
        if f0:
            parts.append(f"add a tonal oscillator at F0={f0:.0f} Hz so noise isn't filling for it")
        lines.append(" -> " + "; ".join(parts) + ".")
        lines.append("")
    elif noise_deficit > 0.12:
        lines.append("=== TOO-TONAL WARNING (high priority) ===")
        lines.append(
            f"noise_deficit={noise_deficit:.3f}: the attempt is MORE TONAL/thin than the "
            "target per band — the target has broadband residual energy the synth lacks. "
            "Do NOT reduce noise; instead ADD shaped noise/residual."
        )
        parts = ["add a shaped noise layer matched to the target residual"]
        if resid_cent:
            parts.append(f"target residual centroid {resid_cent:.0f} Hz (LPF/HPF the noise to that shape)")
        lines.append(" -> " + "; ".join(parts) + ".")
        lines.append("")

    correction = build_correction_prompt(mismatches, top_deltas)
    lines.append("=== CORRECTION PROMPT ===")
    lines.append(correction)
    lines.append("")

    if progress and progress.get('plateau_detected') and not should_finish:
        poor_hints = _poor_layer_hints(progress, iteration) if signal_chain_health else []
        best_attempt = progress.get('best_attempt')
        best_score = progress.get('best_score')
        arch_name = progress.get('switch_architecture') or ARCHITECTURE_ORDER[0]
        templates = seeded_templates or ARCHITECTURE_TEMPLATES
        arch_code = templates.get(arch_name, ARCHITECTURE_TEMPLATES.get(arch_name, ''))
        tried = progress.get('architectures_tried', [])
        seed_scores = progress.get('seed_scores', {})
        best_spec_conv = progress.get('best_spec_conv', 0.0)
        best_noise_excess = progress.get('best_noise_excess', 0.0)
        over_noisy = best_noise_excess > 0.12

        if over_noisy:
            # The bus is stuck on an over-noisy optimum: adding another layer
            # would compound the noise (the failure mode this guard prevents).
            # Force a reduce/reshape step before any additive escape. Directional:
            # only triggers when the attempt is genuinely noisier than the target,
            # not when it is too tonal (which wants the opposite fix).
            f0 = parse_target_field(partials_path, 'fundamental_freq')
            resid_cent = parse_target_field(partials_path, 'residual_spectral_centroid')
            lines.append("=== PLATEAU DETECTED — REDUCE NOISE FIRST (do NOT add a layer) ===")
            lines.append(
                f"No new best for {progress.get('iters_since_best', 0)} iterations "
                f"(best is attempt {best_attempt} at {best_score:.4f}), AND the best bus is "
                f"over-noisy (noise_excess={best_noise_excess:.3f} — flatter than target per band)."
            )
            lines.append("Adding another layer here would compound the noise. Instead, make ONE")
            lines.append("change that REDUCES wrong-shape noise from the current BASE CODE:")
            lines.append("  - cut noise/chaos amplitudes (or remove a noise/chaos layer entirely);")
            if resid_cent:
                lines.append(f"  - reshape the residual toward the target centroid {resid_cent:.0f} Hz (HPF/BPF, not low BrownNoise);")
            if f0:
                lines.append(f"  - add a tonal oscillator at the missing fundamental {f0:.0f} Hz so noise isn't filling for it;")
            lines.append("  - re-run the optimizer and check the BOUND-PIN warnings are gone.")
            lines.append("Only once the bus is clean (noise_excess < 0.12) may a later plateau add a layer.")
            lines.append("")
            lines.append(
                "FALLBACK: if you already tried reducing/reshaping noise within the last 2 "
                "iterations without a new best, do NOT keep nudging noise — the architecture is "
                "the wrong fit. IGNORE the reduce-noise instruction and instead develop a "
                "different seed family as a fresh base (the target's formant analysis hints which: "
                "vocal/formant targets -> formant_vocal or subtractive source/filter). This "
                "prevents a reduce-noise dead-end from ending the run early."
            )
            lines.append("")
        elif progress.get('switch_architecture') == '__restart__':
            restart_fam = progress.get('restart_seed_family', 'unknown')
            restart_score = progress.get('restart_seed_score', 0.0)
            winner_fam = progress.get('seed_winner_family', 'unknown')
            restarted = progress.get('restarted_seeds', [])
            lines.append("=== PLATEAU DETECTED — ARCHITECTURES EXHAUSTED (restart Phase B) ===")
            lines.append(
                f"No new best for {progress.get('iters_since_best', 0)} iterations "
                f"(best is attempt {best_attempt} at {best_score:.4f}). "
                f"All {len(ARCHITECTURE_ORDER)} architecture families have been tried "
                f"(tried: {', '.join(tried) if tried else 'none'})."
            )
            if best_spec_conv > 1.0:
                lines.append(
                    f"The best bus still has spectral_convergence = {best_spec_conv:.3f} "
                    f"(> 1.0 = fundamentally wrong spectral structure — the current "
                    f"architecture cannot reproduce the target's spectrum)."
                )
            lines.append(
                "Adding more layers to this bus will not help — the catalog of "
                "architectures is exhausted. Instead, RESTART Phase B from a "
                "DIFFERENT seed family as the new base:"
            )
            lines.append("")
            lines.append(f"  RESTART SEED: {restart_fam} (seed score: {restart_score:.4f})")
            lines.append(f"  Previous winner: {winner_fam}")
            lines.append(f"  Already restarted: {', '.join(restarted) if restarted else 'none'}")
            lines.append("")
            lines.append("STEPS:")
            lines.append(
                f"  1. Read the {restart_fam} block from current_run/seed_templates.txt"
            )
            lines.append(
                f"  2. Write a fresh attempt_{progress.get('iteration', 0) + 1}.scd from "
                f"that template (NOT from the current BASE CODE)"
            )
            lines.append(
                f"  3. Add 3-8 // @param annotations (use frequencies from "
                f"target_partials.txt dominant partials)"
            )
            lines.append(
                f"  4. Run the FULL optimizer budget from config.txt on this fresh seed"
            )
            lines.append(
                f"  5. This becomes the NEW Base Code for the next hill-climb iteration"
            )
            lines.append("")
            lines.append(
                "IMPORTANT: Do NOT copy or layer onto the current bus. This is a "
                "CLEAN-SLATE RESTART — the old bus is architecturally exhausted and "
                "cannot reach the target (spectral_convergence > 1.0 = wrong model). "
                "Treat this as a new Phase B starting from a different seed."
            )
            lines.append("")
            # Show the template code for the restart seed.
            restart_code = templates.get(restart_fam, ARCHITECTURE_TEMPLATES.get(restart_fam, ''))
            if restart_code:
                lines.append(f"=== {restart_fam} TEMPLATE (copy as starting point) ===")
                lines.append(restart_code)
                lines.append("")
        else:
            lines.append("=== PLATEAU DETECTED — ADD A LAYER (do NOT restart) ===")
            lines.append(
                f"No new best for {progress.get('iters_since_best', 0)} iterations "
                f"(best is attempt {best_attempt} at {best_score:.4f})."
            )
            lines.append(f"Architectures already in the bus / tried: {', '.join(tried) if tried else 'none'}.")
            if poor_hints:
                lines.append(
                    "Signal-chain health: POOR layers being carried by the sum — "
                    f"consider replacing: {', '.join(poor_hints)}."
                )
            if seed_scores and arch_name in seed_scores:
                lines.append(
                    f"Adding '{arch_name}' as a new parallel layer (measured seed score: "
                    f"{seed_scores[arch_name]:.4f} — best unexplored seed)."
                )
            lines.append("KEEP your BASE CODE (the current best bus) — do not rewrite it. Add this")
            lines.append("family's signal core as a NEW parallel layer gated by a fresh gain, then let")
            lines.append("the optimizer tune the new gain jointly with the rest:")
            lines.append("  var gNew, newLayer, ...;  // merge vars to top")
            lines.append("  gNew = 0.3; // @param 0.0 1.0")
            lines.append("  newLayer = <signal core below>;")
            lines.append("  sig = sig + (newLayer * gNew);   // parenthesize (rule 14)")
            lines.append("Escape is now ADDITIVE: you keep the partial match already built and add a")
            lines.append("strength the current bus lacks — not abandon it for a restart.")
            lines.append("")
            lines.append(f"New layer architecture: {arch_name}")
            lines.append("(frequencies seeded from the target's dominant partials):")
            lines.append("")
            if arch_code:
                lines.append(arch_code)
            lines.append("")
            lines.append(
                "If the new layer's gain collapses to ~0 after optimization (use the prune pass), "
                f"it added nothing — revert to the BASE CODE below (attempt {best_attempt}) and "
                "try the next unexplored family."
            )
            lines.append("")

    # Elitism: surface best code for next attempt, or for final_result copy on finish.
    base_code = best_code if best_code is not None else prev_code
    show_base_code = base_code is not None
    if progress:
        effective_seed_count = progress.get('seed_count', 0)
        in_seeding_phase = (effective_seed_count > 0 and iteration <= effective_seed_count)
        if in_seeding_phase and not should_finish:
            show_base_code = False

    if show_base_code:
        best_attempt = progress.get('best_attempt') if progress else None
        if should_finish:
            label = (f"=== BEST ATTEMPT CODE (copy to final_result.scd: attempt {best_attempt}) ==="
                     if best_attempt is not None else "=== CURRENT ATTEMPT CODE ===")
        else:
            label = (f"=== BASE CODE FOR NEXT ATTEMPT (best so far: attempt {best_attempt}) ==="
                     if best_attempt is not None else "=== CURRENT ATTEMPT CODE ===")
        lines.append(label)
        lines.append(base_code)

    return "\n".join(lines)


def dump_seed_templates(partials_path, output_path, envelope_seed=True):
    """Write seed templates for all architecture families to a text file.

    Each family gets a clearly delimited block that the agent can copy verbatim
    as the starting point for seeds 2–4.  Frequencies are seeded from the
    target's dominant partials when available.
    """
    partials = parse_partials(partials_path) if partials_path else []
    envelope = parse_target_envelope(partials_path) if envelope_seed else None
    templates = build_seeded_templates(partials, envelope=envelope)

    lines = [
        "# Architecture Templates — copy blocks from this file.",
        "# Seeds 2–4: use struck_resonator, fm_synthesis, resonator_bank only.",
        "# Plateau switches and hybrid layers: any block below.",
        "# Frequencies are seeded from the target's dominant partials.",
        "# After copying: add 3-8 // @param annotations, optionally add ONE",
        "# noise or modulation layer.  Do NOT invent new UGen call signatures.",
        "",
    ]
    for family in ARCHITECTURE_ORDER:
        code = templates.get(family, ARCHITECTURE_TEMPLATES.get(family, ''))
        lines.append(f"=== {family} ===")
        lines.append(code)
        lines.append("")

    Path(output_path).write_text('\n'.join(lines), encoding='utf-8')
    print(f"Seed templates written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare attempt audio with target')
    parser.add_argument('target', nargs='?', help='Path to target audio file')
    parser.add_argument('attempt', nargs='?', help='Path to attempt audio file')
    parser.add_argument('-o', '--output', help='Output comparison report path')
    parser.add_argument('--prev-code', help='Path to current attempt .scd file (included in report)')
    parser.add_argument('--progress-dir', help='Directory for progress.json tracking')
    parser.add_argument('--iteration', type=int, default=0, help='Current iteration number')
    parser.add_argument('--partials', help='Path to target_partials.txt (seeds architecture templates)')
    parser.add_argument('--sample-rate', type=int, default=44100)
    parser.add_argument('--seed-count', type=int, default=0,
                        help='Total number of seeding-phase attempts (0 = legacy single-start)')
    parser.add_argument('--arch', default=None,
                        help='Architecture family used for this attempt (recorded in progress.json)')
    parser.add_argument('--max-iter', type=int, default=0,
                        help='Maximum iterations (0 = read from progress_dir/config.txt)')
    parser.add_argument('--convergence-threshold', type=float, default=0.0,
                        help='Convergence threshold (0 = read from progress_dir/config.txt)')
    parser.add_argument('--dump-templates', metavar='OUTPUT_PATH', default=None,
                        help='Write seed templates for all families to OUTPUT_PATH and exit. '
                             'Use with --partials to seed frequencies from the target.')
    parser.add_argument('--stems-dir', default=None,
                        help='Directory of decomposition stems (sines/residual/percussive.wav). '
                             'Defaults to <progress-dir>/stems. Used for per-component seed scoring.')
    parser.add_argument('--loss-config', default=None,
                        choices=list(LOSS_CONFIGS.keys()),
                        help='Override loss weight config (default/spectral_heavy/perceptual_heavy). '
                             'CLI overrides progress.json auto-selection.')
    parser.add_argument('--signal-chain-health', action='store_true',
                        help='Emit per-layer signal-chain health block in Phase B reports.')
    parser.add_argument('--no-envelope-seed', action='store_true',
                        help='With --dump-templates: skip envelope-aware Env.perc seeding.')
    parser.add_argument('--self-check', action='store_true',
                        help='Verify loss-config selection on seed attempts in progress-dir; exit.')
    parser.add_argument('--self-check-race-report', action='store_true',
                        help='Verify RACE FINISHED one-shot + layer assignment; exit.')
    args = parser.parse_args()

    if args.self_check_race_report:
        sys.exit(run_self_check_race_report())

    if args.self_check:
        if not args.progress_dir:
            parser.error('--self-check requires --progress-dir')
        sys.exit(run_self_check(args.progress_dir, sr=args.sample_rate))

    if args.dump_templates:
        partials_path = args.partials
        if not partials_path and args.progress_dir:
            candidate = os.path.join(args.progress_dir, 'target_partials.txt')
            if os.path.exists(candidate):
                partials_path = candidate
        dump_seed_templates(partials_path, args.dump_templates,
                            envelope_seed=not args.no_envelope_seed)
        sys.exit(0)

    run_config = read_run_config(args.progress_dir)
    max_iterations = args.max_iter or run_config['max_iterations']
    convergence_threshold = (
        args.convergence_threshold or run_config['convergence_threshold']
    )
    if args.seed_count == 0 and run_config['seed_count'] > 0:
        args.seed_count = run_config['seed_count']
    signal_chain_health = (
        args.signal_chain_health or run_config.get('signal_chain_health', False)
    )
    lc = args.loss_config or run_config.get('loss_config')
    loss_cfg = resolve_loss_config(args.progress_dir, cli_override=lc or None)
    apply_loss_config(loss_cfg)

    if not args.target or not args.attempt:
        parser.error("target and attempt are required unless --dump-templates is used")

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
    use_envelope = run_config.get('envelope_seed', True)
    seeded_templates = build_seeded_templates(
        parse_partials(partials_path),
        envelope=parse_target_envelope(partials_path) if use_envelope else None,
    )

    progress = None
    best_code = None
    if args.progress_dir and args.iteration > 0:
        composite = convergence.get('composite_score', 0)
        # Per-component stem scoring during the seeding phase only.
        component_scores = None
        stems_dir = args.stems_dir
        if not stems_dir and args.progress_dir:
            stems_dir = os.path.join(args.progress_dir, 'stems')
        if args.seed_count and args.arch and stems_dir and os.path.isdir(stems_dir):
            component_scores = score_components(args.attempt, stems_dir, sr=args.sample_rate)
        progress = update_progress(
            args.progress_dir, args.iteration, composite,
            seeded_templates=seeded_templates,
            seed_count=args.seed_count,
            arch=args.arch,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            partials_path=partials_path,
            component_scores=component_scores,
            spec_conv=convergence.get('spectral_convergence'),
            flatness_distance=convergence.get('flatness_distance'),
            noise_excess=convergence.get('noise_excess'),
        )
        # During seeding, do NOT surface base code — each seed is independent.
        # After seeding (or without seeding), surface the best attempt's code.
        effective_seed_count = progress.get('seed_count', 0)
        in_seeding_phase = (effective_seed_count > 0
                            and args.iteration <= effective_seed_count)
        should_finish = progress.get('should_finish', False)
        if not in_seeding_phase or should_finish:
            best_attempt = progress.get('best_attempt')
            if best_attempt is not None:
                best_path = os.path.join(args.progress_dir,
                                         f'attempt_{best_attempt}.scd')
                if os.path.exists(best_path):
                    best_code = Path(best_path).read_text(encoding='utf-8').strip()
    if best_code is None and not (progress and progress.get('should_finish')):
        best_code = prev_code

    # format_report may flip _race_announced; capture pre-call state so we can
    # persist progress.json once when that transition happens (update_progress
    # already wrote the file before format_report ran).
    race_was_unannounced = bool(progress and not progress.get('_race_announced'))
    report = format_report(convergence, mismatches, top_deltas,
                           prev_code=prev_code, progress=progress,
                           best_code=best_code, seeded_templates=seeded_templates,
                           max_iterations=max_iterations,
                           convergence_threshold=convergence_threshold,
                           partials_path=partials_path,
                           signal_chain_health=signal_chain_health,
                           output_dir=args.progress_dir)

    if (args.progress_dir and progress is not None
            and race_was_unannounced and progress.get('_race_announced')):
        progress_path = os.path.join(args.progress_dir, 'progress.json')
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Comparison saved to {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()

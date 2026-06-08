#!/usr/bin/env python3
"""
Evaluate a single audio file and save results as both .txt and .json.
Thin wrapper around the parent project's SynthesisEvaluator.

The .txt output includes:
  - Raw numeric metrics
  - CATEGORIES section (brightness, attack_time, etc.)
  - SYNTHESIS CONCEPTS section (oscillator, filter, envelope, modulation suggestions)
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Use the fixed local version of synthesis_evaluator
from synthesis_evaluator_fixed import SynthesisEvaluator, evaluate_audio_file


# Maps category labels to synthesis concept suggestions
_SYNTHESIS_CONCEPTS = {
    'brightness': {
        'very_dark': (
            'Use low-frequency oscillators (SinOsc 50-300Hz), heavy LPF below 500Hz. '
            'Consider Klang with only low partials, or LFTri/LFSaw at sub-bass frequencies.'
        ),
        'dark': (
            'Use SinOsc or filtered Saw/LFTri, LPF around 800-1500Hz. '
            'Resonz or RLPF with low cutoff. VarSaw with low duty cycle.'
        ),
        'neutral': (
            'Use SinOsc or Saw with moderate LPF (1500-3000Hz). Mix harmonics. '
            'Formant with mid-range frequencies, or Blip with moderate harmonics.'
        ),
        'bright': (
            'Use Saw, Pulse, or Blip, LPF above 3000Hz or no filter. '
            'Add upper harmonics via additive SinOsc or Klang. LFSaw or VarSaw.'
        ),
        'very_bright': (
            'Use Pulse/Saw/Blip with wide bandwidth, HPF to remove lows. '
            'Formant with high resonance frequencies. Klang with high partials. '
            'BRF to notch out mids. Impulse through resonant filter.'
        ),
    },
    'attack_time': {
        'very_slow': (
            'Use Env.adsr with attack 0.5-2.0s for gradual onset. '
            'Lag or Lag2 on amplitude for smooth fade-in.'
        ),
        'slow': (
            'Use Env.adsr with attack 0.1-0.5s. '
            'Line.kr from 0 to 1 over 0.2-0.5s as amplitude envelope.'
        ),
        'moderate': (
            'Use Env.adsr with attack 0.02-0.1s or Env.perc with moderate attack. '
            'Decay2 for natural attack-decay shape.'
        ),
        'punchy': (
            'Use Env.perc(0.001-0.02, releaseTime) for sharp transient. '
            'Decay.ar on Impulse.ar for percussive click. '
            'Pluck or Spring for plucked string transient.'
        ),
        'instant': (
            'Use Env.perc(0.001, releaseTime) or impulse-like envelope. '
            'Impulse.ar as trigger. Dust.ar for sparse impulses. '
            'Very short Decay.ar(Impulse.ar(1), 0.001).'
        ),
    },
    'harmonic_to_noise_ratio': {
        'pure_tone': (
            'Use SinOsc only, no noise sources. '
            'Multiple SinOsc at exact harmonic ratios for additive purity. '
            'Klang with precise frequency ratios.'
        ),
        'clean': (
            'Use SinOsc with a few harmonics (additive synthesis). '
            'Blip with low harmonic count. LFTri or LFSaw for mild harmonics. '
            'Formant for clean vowel-like tones.'
        ),
        'mixed': (
            'Mix SinOsc harmonics with filtered noise or use Saw/VarSaw. '
            'BPF on WhiteNoise mixed with tonal oscillator. '
            'Ringz for resonant mixed character.'
        ),
        'gritty': (
            'Use Saw/Pulse with moderate noise, or ring modulation (SinOsc * SinOsc). '
            'Crackle.ar for gritty texture. Distortion via clipping (sig.clip2). '
            'LFDNoise1 modulating frequency for roughness.'
        ),
        'noisy': (
            'Heavy WhiteNoise, PinkNoise, GrayNoise, or ClipNoise. '
            'Crackle.ar with high chaos parameter. Dust.ar at high density. '
            'Very wide-band Pulse or Blip with many harmonics.'
        ),
    },
    'spectral_flux_normalized': {
        'static': (
            'No modulation. Use fixed frequencies and filter settings. '
            'Sustained SinOsc or Klang with constant parameters.'
        ),
        'stable': (
            'Minimal modulation. Slight vibrato (SinOsc.kr at 5-6Hz on freq). '
            'Gentle filter sweep with very slow Line.kr.'
        ),
        'evolving': (
            'Use slow LFO on filter cutoff (0.5-2Hz) or gentle FM. '
            'SinOsc.kr modulating RLPF cutoff. Lag on modulated parameters. '
            'DynKlang with slowly shifting frequencies.'
        ),
        'dynamic': (
            'Use faster LFO modulation (2-10Hz), FM synthesis, or filter sweeps. '
            'LFNoise1.kr on filter cutoff. XLine.kr sweeping frequency. '
            'CombL for resonant feedback coloring.'
        ),
        'chaotic': (
            'Use random modulation: LFNoise0/1/2 or LFDNoise0/1/3 on freq/cutoff. '
            'Fast FM with SinOsc as modulator at audio rate. '
            'Dust.ar as trigger for random amplitude bursts. '
            'Crackle.ar for unpredictable texture.'
        ),
    },
    'temporal_centroid': {
        'flat': (
            'Use sustained envelope with equal attack/release. '
            'Env.adsr with long sustain and equal attack/release. '
            'Constant amplitude (no envelope) for pure sustain.'
        ),
        'sustained': (
            'Use Env.adsr with long sustain, gentle release. '
            'Env.linen with long sustainTime. FreeVerb or GVerb for tail extension.'
        ),
        'balanced': (
            'Use Env.adsr with moderate attack and release. '
            'Env.perc with attack ≈ release. Balanced Env.linen.'
        ),
        'early': (
            'Use Env.perc — energy concentrated at the start, quick decay. '
            'Decay.ar for exponential decay from peak. Pluck for natural decay.'
        ),
        'front_heavy': (
            'Use very short Env.perc with fast decay, impulse-like. '
            'Decay.ar(Impulse.ar(1), 0.05) for sharp front-loaded burst.'
        ),
    },
    'crest_factor_db': {
        'compressed': (
            'Use sustained envelope, limit dynamic range. '
            'Env.adsr with long sustain and gentle attack. Lag on amplitude.'
        ),
        'sustained': (
            'Use Env.adsr with long sustain. '
            'Env.linen with long sustainTime. Constant amplitude oscillator.'
        ),
        'dynamic': (
            'Use Env.adsr with moderate dynamics. '
            'Env.perc with moderate attack/release ratio.'
        ),
        'percussive': (
            'Use Env.perc with sharp attack, moderate release. '
            'Decay2.ar for percussive shape. Pluck or Spring for natural decay.'
        ),
        'impulsive': (
            'Use very short Env.perc, high peak-to-average ratio. '
            'Impulse.ar through Decay.ar(0.001). Single-sample burst.'
        ),
    },
    'spectral_complexity_mean': {
        'simple': (
            'Use 1-2 SinOsc, minimal harmonics. '
            'Pure sine or single Blip with 1 harmonic. Single LFTri.'
        ),
        'sparse': (
            'Use 2-4 SinOsc at harmonic ratios. '
            'Klang with 2-3 partials. Blip with 3-5 harmonics.'
        ),
        'moderate': (
            'Use Saw or 4-6 SinOsc harmonics with varying amplitudes. '
            'Formant for vowel-like moderate complexity. '
            'Klang with 4-6 partials. VarSaw.'
        ),
        'rich': (
            'Use Mix of multiple oscillators, FM synthesis, or Klang with many partials. '
            'DynKlank for resonant richness. Formant with multiple resonances. '
            'Blip with 20-50 harmonics. FM: SinOsc.ar(freq + SinOsc.ar(modFreq) * modDepth).'
        ),
        'dense': (
            'Use many oscillators, FM, ring mod, or noise-based synthesis. '
            'DynKlank with many resonances. Mix of Saw + noise + FM. '
            'Klang with 10+ partials. Crackle + filtered noise + oscillators. '
            'Ring modulation: sig1 * sig2.'
        ),
    },
    'spectral_slope': {
        'steep_rolloff': (
            'Use heavy LPF (cutoff 200-600Hz), SinOsc-based, minimal high-frequency content. '
            'RLPF with low rq for sharp resonant rolloff. LPF cascaded twice.'
        ),
        'lowpass': (
            'Use LPF with moderate cutoff (600-2000Hz), Saw with filtering. '
            'RLPF or MoogFF for smooth lowpass character.'
        ),
        'balanced': (
            'Use moderate filtering, mix of oscillator types. '
            'BPF to shape midrange. Resonz for balanced resonant peak.'
        ),
        'bright': (
            'Use minimal filtering, Saw/Pulse/Blip oscillators. '
            'HPF to remove low mud. Formant with high resonance.'
        ),
        'highpass': (
            'Use HPF (cutoff 2000Hz+), emphasize high frequencies. '
            'RHPF for resonant high-pass character. BRF to notch lows. '
            'Impulse.ar through resonant BPF at high frequency.'
        ),
    },
    'envelope_flatness': {
        'flat': (
            'Use constant amplitude or very gentle Env.adsr with long sustain. '
            'No envelope — just a sustained oscillator. Lag for smooth constant level.'
        ),
        'sustained': (
            'Use Env.adsr with long sustain, gentle curves. '
            'Env.linen with long sustainTime. FreeVerb tail for extended decay.'
        ),
        'dynamic': (
            'Use Env.perc or Env.adsr with clear attack/decay shape. '
            'Decay2 for natural dynamic shape. Pluck for natural string decay.'
        ),
        'percussive': (
            'Use Env.perc with short decay (0.1-0.5s). '
            'Decay.ar for exponential percussive decay. '
            'AllpassN for short resonant tail after percussive hit.'
        ),
        'impulsive': (
            'Use very short Env.perc (< 0.1s total). '
            'Impulse.ar through Decay.ar(0.02). CombL for pitched resonance after impulse.'
        ),
    },
}


def save_result_to_txt(result, categories, output_path):
    """Save evaluation result with categories and synthesis concepts to a text file."""
    with open(output_path, 'w') as f:
        f.write("=== AUDIO METRICS ===\n\n")
        for key, value in sorted(result.items()):
            f.write(f"{key}: {value:.6f}\n")

        f.write("\n=== CATEGORIES ===\n\n")
        for cat_name, label in sorted(categories.items()):
            f.write(f"{cat_name}: {label}\n")

        f.write("\n=== SYNTHESIS CONCEPTS ===\n\n")
        for cat_name, label in sorted(categories.items()):
            concepts = _SYNTHESIS_CONCEPTS.get(cat_name, {})
            suggestion = concepts.get(label, '')
            if suggestion:
                f.write(f"{cat_name} ({label}): {suggestion}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate audio file for sound matching')
    parser.add_argument('audio_file', help='Path to audio file (.wav)')
    parser.add_argument('-o', '--output', help='Output .txt file path (also saves .json alongside)')
    parser.add_argument('--sample-rate', type=int, default=44100)
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: audio file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    result = evaluate_audio_file(args.audio_file, sample_rate=args.sample_rate)

    # Categorize the metrics
    evaluator = SynthesisEvaluator(sample_rate=args.sample_rate)
    categories = evaluator.categorize_metrics(result)

    if args.output:
        save_result_to_txt(result, categories, args.output)

        json_path = str(Path(args.output).with_suffix('.json'))
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"Evaluation saved to {args.output} and {json_path}")
    else:
        save_result_to_txt(result, categories, '/dev/stdout')


if __name__ == '__main__':
    main()

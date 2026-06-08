#!/usr/bin/env python3
"""
FluCoMa-based target audio analysis for sc_claw_flucoma.

Runs four FluCoMa CLI tools on the target WAV, performs layered spectral
analysis in Python, and outputs target_partials.txt with:
  - Decomposition summary (sinusoidal/residual/harmonic/percussive ratios)
  - Dominant partial data (frequency, amplitude, decay profile, freq drift)
  - 5 ready-to-use SC templates of increasing complexity (A through E)
"""

import sys
import os
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

FLUCOMA_BIN = "/home/ayk/flucoma_cli/bin"


def run_flucoma(tool, args):
    cmd = [os.path.join(FLUCOMA_BIN, tool)] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"WARNING: {tool} returned {result.returncode}: {result.stderr}", file=sys.stderr)
    return result.returncode == 0


def analyze_partials(source_path, n_peaks=10, sr=44100):
    """Run fluid-sinefeature and extract per-partial data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        freq_path = os.path.join(tmpdir, "freqs.wav")
        mag_path = os.path.join(tmpdir, "mags.wav")

        ok = run_flucoma("fluid-sinefeature", [
            "-source", source_path,
            "-frequency", freq_path,
            "-magnitude", mag_path,
            "-order", "1",
            "-detectionthreshold", "-60.0",
        ])
        if not ok or not os.path.exists(freq_path):
            return None

        freqs, _ = sf.read(freq_path, always_2d=True)
        mags, _ = sf.read(mag_path, always_2d=True)

    n_channels = freqs.shape[1]
    if n_channels > n_peaks:
        half = n_channels // 2
        freqs = (freqs[:, :half] + freqs[:, half:]) / 2
        mags = (mags[:, :half] + mags[:, half:]) / 2

    actual_peaks = min(n_peaks, freqs.shape[1])
    hop_sec = 1024 / sr
    n_frames = freqs.shape[0]

    partials = []
    for pk in range(actual_peaks):
        f_col = freqs[:, pk]
        m_col = mags[:, pk]
        active = f_col > 0

        if active.sum() < 5:
            continue

        active_idx = np.where(active)[0]
        onset_frame = active_idx[0]
        offset_frame = active_idx[-1]
        peak_idx_local = np.argmax(m_col[active_idx])
        peak_frame = active_idx[peak_idx_local]
        peak_mag = float(m_col[peak_frame])

        q_mags = []
        for q in [0.25, 0.5, 0.75]:
            qi = active_idx[int(len(active_idx) * q)]
            q_mags.append(float(m_col[qi]))

        decay_ratio = q_mags[2] / peak_mag if peak_mag > 0 else 0.0

        f_active = f_col[active]
        freq_mean = float(f_active.mean())
        freq_std = float(f_active.std())

        mod_rate = 0.0
        if len(f_active) > 20:
            kernel = min(10, len(f_active) // 2)
            if kernel > 0:
                smoothed = np.convolve(f_active, np.ones(kernel) / kernel, mode='same')
                detrended = f_active - smoothed
                zc = np.sum(np.diff(np.sign(detrended)) != 0)
                duration = len(f_active) * hop_sec
                mod_rate = float(zc / duration) if duration > 0 else 0.0

        presence = float(active.sum() / n_frames * 100)

        partials.append({
            'rank': pk,
            'freq_mean': freq_mean,
            'freq_std': freq_std,
            'amp_peak': peak_mag,
            'amp_mean': float(m_col[active].mean()),
            'presence': presence,
            'onset_sec': float(onset_frame * hop_sec),
            'peak_sec': float(peak_frame * hop_sec),
            'offset_sec': float(offset_frame * hop_sec),
            'decay_ratio': decay_ratio,
            'mod_rate': mod_rate,
        })

    return partials


def analyze_decomposition(source_path, sr=44100):
    """Run fluid-sines and fluid-hpss, return energy ratios."""
    info = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        sines_path = os.path.join(tmpdir, "sines.wav")
        resid_path = os.path.join(tmpdir, "residual.wav")

        ok = run_flucoma("fluid-sines", [
            "-source", source_path,
            "-sines", sines_path,
            "-residual", resid_path,
            "-detectionthreshold", "-60.0",
        ])

        src, _ = sf.read(source_path)
        if src.ndim > 1:
            src = src.mean(axis=1)
        src_rms = float(np.sqrt(np.mean(src ** 2)))

        if ok and os.path.exists(sines_path):
            sines, _ = sf.read(sines_path)
            resid, _ = sf.read(resid_path)
            if sines.ndim > 1:
                sines = sines.mean(axis=1)
            if resid.ndim > 1:
                resid = resid.mean(axis=1)

            info['sine_rms'] = float(np.sqrt(np.mean(sines ** 2)))
            info['resid_rms'] = float(np.sqrt(np.mean(resid ** 2)))
            info['sine_pct'] = info['sine_rms'] / src_rms * 100 if src_rms > 0 else 0
            info['resid_pct'] = info['resid_rms'] / src_rms * 100 if src_rms > 0 else 0

            resid_spec = np.abs(librosa.stft(resid, n_fft=4096))
            freqs_axis = librosa.fft_frequencies(sr=sr, n_fft=4096)
            avg_spec = resid_spec.mean(axis=1)

            centroid = float(np.sum(freqs_axis * avg_spec) / (np.sum(avg_spec) + 1e-12))
            info['resid_centroid'] = centroid

            log_f = np.log10(freqs_axis[1:] + 1e-12)
            log_s = np.log10(avg_spec[1:] + 1e-12)
            slope, _ = np.polyfit(log_f, log_s, 1)
            info['resid_slope'] = float(slope)

            if slope > -0.3:
                info['noise_type'] = 'WhiteNoise'
            elif slope > -0.7:
                info['noise_type'] = 'PinkNoise'
            else:
                info['noise_type'] = 'BrownNoise'

            resid_rms_frames = librosa.feature.rms(y=resid, frame_length=2048, hop_length=512)[0]
            n_fr = len(resid_rms_frames)
            third = max(1, n_fr // 3)
            early = float(resid_rms_frames[:third].mean())
            mid = float(resid_rms_frames[third:2 * third].mean())
            late = float(resid_rms_frames[2 * third:].mean())
            info['resid_env'] = {'early': early, 'mid': mid, 'late': late}

            if mid >= early and mid >= late:
                info['resid_env_shape'] = 'sustained/humped'
            elif early >= mid >= late:
                info['resid_env_shape'] = 'decaying'
            else:
                info['resid_env_shape'] = 'building'

        harm_path = os.path.join(tmpdir, "harmonic.wav")
        perc_path = os.path.join(tmpdir, "percussive.wav")

        ok = run_flucoma("fluid-hpss", [
            "-source", source_path,
            "-harmonic", harm_path,
            "-percussive", perc_path,
        ])

        if ok and os.path.exists(harm_path):
            harm, _ = sf.read(harm_path)
            perc, _ = sf.read(perc_path)
            if harm.ndim > 1:
                harm = harm.mean(axis=1)
            if perc.ndim > 1:
                perc = perc.mean(axis=1)
            info['harm_pct'] = float(np.sqrt(np.mean(harm ** 2)) / src_rms * 100) if src_rms > 0 else 0
            info['perc_pct'] = float(np.sqrt(np.mean(perc ** 2)) / src_rms * 100) if src_rms > 0 else 0

    return info


def generate_templates(partials, decomp, target_duration):
    """Generate 5 SC code templates of increasing complexity."""
    if not partials:
        return ""

    top5 = partials[:5]
    freqs = [p['freq_mean'] for p in top5]
    amps = [p['amp_peak'] for p in top5]
    decay_ratios = [p['decay_ratio'] for p in top5]
    drift_depths = [min(p['freq_std'] * 0.12, 80) for p in top5]
    mod_rates = [max(2.0, min(p['mod_rate'] * 0.3, 8.0)) for p in top5]

    peak_frac = min(top5[0]['peak_sec'] / target_duration, 0.8) if target_duration > 0 else 0.5
    attack_time = round(max(0.01, peak_frac * target_duration), 2)
    release_time = round(max(0.5, (1 - peak_frac) * target_duration), 2)

    noise_type = decomp.get('noise_type', 'PinkNoise')
    resid_cutoff = int(decomp.get('resid_centroid', 2000))
    resid_pct = decomp.get('resid_pct', 30)
    perc_pct = decomp.get('perc_pct', 10)
    noise_amp = round(max(0.005, min(resid_pct / 100 * 0.08, 0.06)), 4)
    transient_amp = round(max(0.01, min(perc_pct / 100 * 0.3, 0.2)), 3)

    env_shape = decomp.get('resid_env_shape', 'sustained/humped')
    re = decomp.get('resid_env', {'early': 1, 'mid': 1, 'late': 0.5})
    peak_re = max(re['early'], re['mid'], re['late'])
    if peak_re > 0:
        ne_early = round(re['early'] / peak_re, 2)
        ne_mid = round(re['mid'] / peak_re, 2)
        ne_late = round(re['late'] / peak_re, 2)
    else:
        ne_early, ne_mid, ne_late = 0.5, 1.0, 0.3

    third_dur = round(target_duration / 3, 2)

    freq_strs = ", ".join(f"{f:.1f}" for f in freqs)
    amp_strs = ", ".join(f"{a:.3f}" for a in amps)

    lines = []

    # Template A
    lines.append("=== TEMPLATE A — Klang foundation (exact partials, single envelope) ===")
    lines.append("var env, sig;")
    lines.append(f"env = EnvGen.kr(Env.perc({attack_time}, {release_time}), doneAction: 2);")
    lines.append(f"sig = Klang.ar(`[[{freq_strs}], [{amp_strs}], nil]);")
    lines.append("Out.ar(0, (sig * env * 0.4).dup);")
    lines.append("")

    # Template B
    lines.append("=== TEMPLATE B — Per-partial envelopes (temporal realism) ===")
    var_names = [f"p{i}" for i in range(len(top5))]
    lines.append(f"var sig, {', '.join(var_names)};")
    for i, p in enumerate(top5):
        dr = round(p['decay_ratio'], 2)
        rel = round(release_time - i * 0.5, 2)
        if rel < 1.0:
            rel = 1.0
        da = ", doneAction: 2" if i == 0 else ""
        lines.append(
            f"{var_names[i]} = SinOsc.ar({p['freq_mean']:.1f}) * "
            f"EnvGen.kr(Env([0,1,{dr}],[{attack_time},{rel}],[-4,-6]){da}) * {p['amp_peak']:.3f};"
        )
    lines.append(f"sig = {' + '.join(var_names)};")
    lines.append("Out.ar(0, (sig * 0.4).dup);")
    lines.append("")

    # Template C
    lines.append("=== TEMPLATE C — Drifting partials (spectral movement) ===")
    lines.append(f"var sig, {', '.join(var_names)};")
    for i, p in enumerate(top5):
        dr = round(p['decay_ratio'], 2)
        rel = round(release_time - i * 0.5, 2)
        if rel < 1.0:
            rel = 1.0
        dd = round(drift_depths[i], 1)
        mr = round(mod_rates[i], 1)
        da = ", doneAction: 2" if i == 0 else ""
        lines.append(
            f"{var_names[i]} = SinOsc.ar({p['freq_mean']:.1f} + LFNoise1.kr({mr}, {dd})) * "
            f"EnvGen.kr(Env([0,1,{dr}],[{attack_time},{rel}],[-4,-6]){da}) * {p['amp_peak']:.3f};"
        )
    lines.append(f"sig = {' + '.join(var_names)};")
    lines.append("Out.ar(0, (sig * 0.4).dup);")
    lines.append("")

    # Template D
    lines.append("=== TEMPLATE D — Layered: partials + shaped residual ===")
    lines.append(f"var sig, partials, noise, noiseEnv, {', '.join(var_names)};")
    for i, p in enumerate(top5):
        dr = round(p['decay_ratio'], 2)
        rel = round(release_time - i * 0.5, 2)
        if rel < 1.0:
            rel = 1.0
        dd = round(drift_depths[i], 1)
        mr = round(mod_rates[i], 1)
        da = ", doneAction: 2" if i == 0 else ""
        lines.append(
            f"{var_names[i]} = SinOsc.ar({p['freq_mean']:.1f} + LFNoise1.kr({mr}, {dd})) * "
            f"EnvGen.kr(Env([0,1,{dr}],[{attack_time},{rel}],[-4,-6]){da}) * {p['amp_peak']:.3f};"
        )
    lines.append(f"partials = {' + '.join(var_names)};")
    lines.append(
        f"noiseEnv = EnvGen.kr(Env([0, {ne_early}, {ne_mid}, {ne_late}, 0], "
        f"[{third_dur}, {third_dur}, {third_dur}, 0.1], [-2, 0, -4, -4]));"
    )
    lines.append(f"noise = LPF.ar({noise_type}.ar({noise_amp}), {resid_cutoff}) * noiseEnv;")
    lines.append("sig = partials + noise;")
    lines.append("Out.ar(0, (sig * 0.4).dup);")
    lines.append("")

    # Template E
    lines.append("=== TEMPLATE E — Full layered: partials + shaped residual + percussive transient ===")
    lines.append(f"var sig, partials, noise, noiseEnv, transient, {', '.join(var_names)};")
    for i, p in enumerate(top5):
        dr = round(p['decay_ratio'], 2)
        rel = round(release_time - i * 0.5, 2)
        if rel < 1.0:
            rel = 1.0
        dd = round(drift_depths[i], 1)
        mr = round(mod_rates[i], 1)
        da = ", doneAction: 2" if i == 0 else ""
        lines.append(
            f"{var_names[i]} = SinOsc.ar({p['freq_mean']:.1f} + LFNoise1.kr({mr}, {dd})) * "
            f"EnvGen.kr(Env([0,1,{dr}],[{attack_time},{rel}],[-4,-6]){da}) * {p['amp_peak']:.3f};"
        )
    lines.append(f"partials = {' + '.join(var_names)};")
    lines.append(
        f"noiseEnv = EnvGen.kr(Env([0, {ne_early}, {ne_mid}, {ne_late}, 0], "
        f"[{third_dur}, {third_dur}, {third_dur}, 0.1], [-2, 0, -4, -4]));"
    )
    lines.append(f"noise = LPF.ar({noise_type}.ar({noise_amp}), {resid_cutoff}) * noiseEnv;")
    lines.append(
        f"transient = Decay2.ar(Impulse.ar(0), 0.005, 0.08) * "
        f"BPF.ar(WhiteNoise.ar({transient_amp}), {min(resid_cutoff, 1200)}, 0.4);"
    )
    lines.append("sig = partials + noise + transient;")
    lines.append("Out.ar(0, (sig * 0.4).dup);")

    return "\n".join(lines)


def format_output(partials, decomp, templates):
    lines = []

    lines.append("=== DECOMPOSITION SUMMARY ===")
    lines.append(f"sinusoidal_energy: {decomp.get('sine_pct', 0):.1f}%")
    lines.append(f"residual_energy: {decomp.get('resid_pct', 0):.1f}%")
    lines.append(f"harmonic_energy: {decomp.get('harm_pct', 0):.1f}%")
    lines.append(f"percussive_energy: {decomp.get('perc_pct', 0):.1f}%")
    cent = decomp.get('resid_centroid', 0)
    lines.append(f"residual_spectral_centroid: {cent:.0f} Hz")
    slope = decomp.get('resid_slope', 0)
    noise = decomp.get('noise_type', 'PinkNoise')
    lines.append(f"residual_spectral_slope: {slope:.2f} (use LPF on {noise})")
    lines.append(f"residual_envelope: {decomp.get('resid_env_shape', 'unknown')}")

    if partials:
        avg_drift = np.mean([p['freq_std'] for p in partials[:5]])
        lines.append(f"partial_freq_drift: {'high' if avg_drift > 100 else 'moderate' if avg_drift > 30 else 'low'} (avg std {avg_drift:.0f} Hz)")
        dr_range = [p['decay_ratio'] for p in partials[:5]]
        if len(dr_range) >= 2:
            lines.append(f"per_partial_decay: higher partials decay faster (ratio {dr_range[0]:.2f} -> {dr_range[-1]:.2f})")

    lines.append("")
    lines.append("=== DOMINANT PARTIALS (top 10, by average magnitude) ===")
    for i, p in enumerate(partials[:10]):
        lines.append(
            f"  #{i + 1}: {p['freq_mean']:.1f} Hz, amp={p['amp_peak']:.4f}, "
            f"presence={p['presence']:.0f}%, decay_ratio={p['decay_ratio']:.2f}, "
            f"freq_drift_hz={p['freq_std']:.0f}"
        )

    lines.append("")
    lines.append(templates)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='FluCoMa target audio analysis')
    parser.add_argument('audio_file', help='Path to target audio file (.wav)')
    parser.add_argument('-o', '--output', help='Output .txt file path')
    parser.add_argument('--sample-rate', type=int, default=44100)
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: audio file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    info = sf.info(args.audio_file)
    target_duration = info.duration

    print(f"Analyzing partials: {args.audio_file} ({target_duration:.1f}s)")
    partials = analyze_partials(args.audio_file, sr=args.sample_rate)
    if not partials:
        print("WARNING: fluid-sinefeature produced no usable partials", file=sys.stderr)
        partials = []

    print("Analyzing decomposition (sines/residual/harmonic/percussive)...")
    decomp = analyze_decomposition(args.audio_file, sr=args.sample_rate)

    print("Generating SC templates...")
    templates = generate_templates(partials, decomp, target_duration)

    output = format_output(partials, decomp, templates)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Analysis saved to {args.output}")
    else:
        print(output)


if __name__ == '__main__':
    main()

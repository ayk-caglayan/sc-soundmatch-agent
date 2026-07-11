#!/usr/bin/env python3
"""
Fixed Synthesis Evaluator - Comprehensive Audio Quality Metrics

Evaluates synthesis algorithms based on:
- Spectral Balance: Energy distribution across frequency bands
- Temporal Dynamics: Attack, decay, transient characteristics
- Richness: Spectral complexity, inharmonicity, modulation
"""

import numpy as np
import librosa
from scipy import signal, stats
from typing import Dict, Optional, Tuple

# ponytail: JTFS is a better perceptual metric for nonstationary sounds
# (captures AM/FM modulation that MSS misses), but kymatio is a heavy dep.
# Only import when the evaluator is constructed with use_jtfs=True.
_JTFS_AVAILABLE = False
try:
    import kymatio  # noqa: F401
    _JTFS_AVAILABLE = True
except ImportError:
    pass


# Loss multiplexing (approach #1): set before scoring; read by compare_with_reference.
# compare.py / optimize_params.py assign this from progress['loss_config'] or CLI.
LOSS_CONFIGS = {
    'default': {
        'mfcc': 0.22, 'mel': 0.08, 'spec': 0.18, 'centroid': 0.10,
        'flatness': 0.07, 'lsd': 0.05, 'envelope': 0.10, 'onset': 0.05,
        'category': 0.15,
    },
    'spectral_heavy': {
        'mfcc': 0.15, 'mel': 0.10, 'spec': 0.28, 'centroid': 0.12,
        'flatness': 0.05, 'lsd': 0.07, 'envelope': 0.08, 'onset': 0.03,
        'category': 0.12,
    },
    'perceptual_heavy': {
        'mfcc': 0.30, 'mel': 0.10, 'spec': 0.10, 'centroid': 0.08,
        'flatness': 0.06, 'lsd': 0.04, 'envelope': 0.12, 'onset': 0.05,
        'category': 0.15,
    },
}
_ACTIVE_LOSS_CONFIG = None  # name key into LOSS_CONFIGS, or None for built-in default


def _composite_from_metrics(metrics, spec_term, cat_penalty, cfg=None):
    """Blend metric dict into composite_score using cfg weights (or defaults)."""
    if cfg is None:
        return float(
            0.22 * min(metrics['mfcc_distance'], 2.0)
            + 0.08 * min(metrics['mel_distance'], 2.0)
            + 0.18 * min(spec_term, 2.5)
            + 0.10 * metrics['centroid_distance']
            + 0.07 * metrics['flatness_distance']
            + 0.05 * min(metrics['log_spectral_distance'] / 10.0, 2.0)
            + 0.10 * min(metrics['envelope_distance'], 2.0)
            + 0.05 * metrics['onset_max_penalty']
            + 0.15 * cat_penalty
        )
    return float(
        cfg['mfcc'] * min(metrics['mfcc_distance'], 2.0)
        + cfg['mel'] * min(metrics['mel_distance'], 2.0)
        + cfg['spec'] * min(spec_term, 2.5)
        + cfg['centroid'] * metrics['centroid_distance']
        + cfg['flatness'] * metrics['flatness_distance']
        + cfg['lsd'] * min(metrics['log_spectral_distance'] / 10.0, 2.0)
        + cfg['envelope'] * min(metrics['envelope_distance'], 2.0)
        + cfg['onset'] * metrics['onset_max_penalty']
        + cfg['category'] * cat_penalty
    )


# ---------------------------------------------------------------------------
# Shared audio preprocessing
# ---------------------------------------------------------------------------

def preprocess_audio(
    audio: np.ndarray,
    sr: int = 44100,
    normalize: bool = True,
    trim_silence: bool = True,
    top_db: float = 40.0,
) -> np.ndarray:
    """
    Normalize and optionally trim silence from an audio array.

    Args:
        audio:          1-D float audio array (already mono, already at target sr)
        sr:             Sample rate (used for onset-based trim fallback)
        normalize:      If True, RMS-normalize to a fixed target level.
        trim_silence:   If True, strip leading/trailing silence using librosa.effects.trim.
        top_db:         Silence threshold in dB below peak for trimming.

    Returns:
        Preprocessed 1-D float array.
    """
    if audio.size == 0:
        return audio

    if trim_silence:
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        if trimmed.size > 0:
            audio = trimmed

    if normalize:
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms > 1e-9:
            target_rms = 0.1
            audio = audio * (target_rms / rms)

    return audio


def load_and_preprocess(
    path: str,
    sr: int = 44100,
    normalize: bool = True,
    trim_silence: bool = True,
    top_db: float = 40.0,
) -> Tuple[np.ndarray, float]:
    """
    Load an audio file, convert to mono, resample, trim silence, and normalize.

    Returns:
        (audio, original_duration_seconds)
    """
    import soundfile as sf

    audio, file_sr = sf.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    original_duration = len(audio) / file_sr

    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

    audio = preprocess_audio(audio, sr=sr, normalize=normalize,
                             trim_silence=trim_silence, top_db=top_db)

    return audio, original_duration


class SynthesisEvaluator:
    """Compute objective metrics for synthesis quality evaluation."""
    
    def __init__(self, sample_rate=16000, use_jtfs=False):
        self.sr = sample_rate
        self._use_jtfs = use_jtfs and _JTFS_AVAILABLE
        self._jtfs_transform = None
        self._jtfs_shape = None
        
        # Frequency band definitions (Hz)
        self.bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'highs': (4000, 8000)
        }
        
        # Category thresholds for categorize_metrics
        self.category_thresholds = {
            'brightness': {
                'metric': 'spectral_centroid_mean',
                'thresholds': [200, 500, 1500, 4000],
                'labels': ['very_dark', 'dark', 'neutral', 'bright', 'very_bright'],
            },
            'attack_time': {
                'metric': 'onset_max',
                'thresholds': [2, 5, 10, 20],
                'labels': ['very_slow', 'slow', 'moderate', 'punchy', 'instant'],
            },
            'harmonic_to_noise_ratio': {
                'metric': 'spectral_flatness_mid',
                'thresholds': [0.001, 0.01, 0.1, 0.3],
                'labels': ['pure_tone', 'clean', 'mixed', 'gritty', 'noisy'],
            },
            'spectral_flux_normalized': {
                'metric': 'spectral_entropy_std',
                'thresholds': [0.05, 0.15, 0.3, 0.5],
                'labels': ['static', 'stable', 'evolving', 'dynamic', 'chaotic'],
            },
            'temporal_centroid': {
                'metric': 'rms_std',
                'thresholds': [0.01, 0.05, 0.1, 0.2],
                'labels': ['flat', 'sustained', 'balanced', 'early', 'front_heavy'],
            },
            'crest_factor_db': {
                'metric': 'rms_max',
                'thresholds': [0.05, 0.15, 0.3, 0.5],
                'labels': ['compressed', 'sustained', 'dynamic', 'percussive', 'impulsive'],
            },
            'spectral_complexity_mean': {
                'metric': 'spectral_entropy_mean',
                'thresholds': [0.3, 0.5, 0.7, 0.9],
                'labels': ['simple', 'sparse', 'moderate', 'rich', 'dense'],
            },
            'spectral_slope': {
                'metric': 'spectral_spread_mean',
                'thresholds': [50, 150, 400, 1000],
                'labels': ['steep_rolloff', 'lowpass', 'balanced', 'bright', 'highpass'],
            },
            'envelope_flatness': {
                'metric': 'rms_std',
                'thresholds': [0.005, 0.02, 0.08, 0.15],
                'labels': ['flat', 'sustained', 'moderate', 'dynamic', 'very_dynamic'],
            },
        }
    
    def evaluate(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Compute all metrics for an audio signal.
        
        Args:
            audio: 1D audio array
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Spectral balance metrics
        metrics.update(self.spectral_balance_metrics(audio))
        
        # Temporal dynamics metrics
        metrics.update(self.temporal_dynamics_metrics(audio))
        
        # Richness metrics
        metrics.update(self.richness_metrics(audio))
        
        return metrics
    
    def spectral_balance_metrics(self, audio: np.ndarray) -> Dict[str, float]:
        """Compute spectral balance metrics."""
        # Compute STFT
        D = librosa.stft(audio, n_fft=2048)
        S_mag = np.abs(D)
        S_power = S_mag ** 2
        
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)
        
        metrics = {}
        
        # Energy per band - FIX: use proper 2D indexing
        total_energy = np.sum(S_power)
        for band_name, (f_low, f_high) in self.bands.items():
            band_mask = (freqs >= f_low) & (freqs < f_high)
            # Fixed: use band_mask with proper axis indexing for 2D array
            band_energy = np.sum(S_power[band_mask, :], axis=None)
            metrics[f'band_energy_{band_name}'] = band_energy / (total_energy + 1e-12)
        
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr, n_fft=2048)[0]
        metrics['spectral_centroid_mean'] = float(np.mean(centroid))
        metrics['spectral_centroid_std'] = float(np.std(centroid))
        
        # Spectral spread (width)
        spread = np.sqrt(np.sum(
            ((freqs[:, np.newaxis] - centroid) ** 2) * S_power, axis=0
        ) / (np.sum(S_power, axis=0) + 1e-12))
        metrics['spectral_spread_mean'] = float(np.mean(spread))
        
        # Spectral flatness (per band)
        for band_name, (f_low, f_high) in self.bands.items():
            band_mask = (freqs >= f_low) & (freqs < f_high)
            if np.any(band_mask):
                band_spec = S_mag[band_mask, :]
                # Spectral flatness = geometric mean / arithmetic mean
                geometric_mean = stats.gmean(band_spec.flatten() + 1e-12)
                arithmetic_mean = np.mean(band_spec)
                flatness = geometric_mean / (arithmetic_mean + 1e-12)
                metrics[f'spectral_flatness_{band_name}'] = flatness
        
        return metrics
    
    def temporal_dynamics_metrics(self, audio: np.ndarray) -> Dict[str, float]:
        """Compute temporal dynamics metrics."""
        metrics = {}
        
        # RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        metrics['rms_mean'] = float(np.mean(rms))
        metrics['rms_std'] = float(np.std(rms))
        metrics['rms_max'] = float(np.max(rms))
        
        # Energy envelope - attack and decay characteristics
        envelope = librosa.onset.onset_strength(y=audio, sr=self.sr)
        metrics['onset_mean'] = float(np.mean(envelope))
        metrics['onset_max'] = float(np.max(envelope))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=512)[0]
        metrics['zero_crossing_rate_mean'] = float(np.mean(zcr))
        metrics['zero_crossing_rate_std'] = float(np.std(zcr))
        
        return metrics
    
    def richness_metrics(self, audio: np.ndarray) -> Dict[str, float]:
        """Compute richness/complexity metrics."""
        metrics = {}
        
        # Spectral entropy (complexity)
        D = librosa.stft(audio, n_fft=2048)
        S_mag = np.abs(D)
        S_power = (S_mag ** 2)
        
        # Normalize to probability distribution
        S_normalized = S_power / (np.sum(S_power, axis=0, keepdims=True) + 1e-12)
        
        # Compute entropy
        entropy = -np.sum(S_normalized * np.log(S_normalized + 1e-12), axis=0)
        metrics['spectral_entropy_mean'] = float(np.mean(entropy))
        metrics['spectral_entropy_std'] = float(np.std(entropy))
        
        # Spectral contrast (perceptual richness)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr, n_fft=2048)
        metrics['spectral_contrast_mean'] = float(np.mean(contrast))
        metrics['spectral_contrast_std'] = float(np.std(contrast))
        
        # Chroma features (pitch content richness)
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr, n_fft=2048)
        chroma_energy = np.mean(chroma, axis=1)
        metrics['chroma_energy_mean'] = float(np.mean(chroma_energy))
        metrics['chroma_energy_std'] = float(np.std(chroma_energy))
        
        return metrics
    
    def _jtfs_distance(self, test_audio, ref_audio):
        """JTFS-based perceptual distance (approach #3: Han et al. 2024).

        Joint Time-Frequency Scattering captures spectrotemporal modulations
        that standard MSS misses — critical for FM synthesis, arpeggios, and
        transient sounds. Returns a float distance or None if kymatio is
        absent or computation fails.

        ponytail: uses minimal JTFS config (J=8, Q=8, J_fr=3) — enough to
        capture first-order modulation.  Full config is overkill for the
        typical 2-second SC render.
        """
        if not self._use_jtfs:
            return None
        try:
            from kymatio.scattering1d import TimeFrequencyScattering1D
            import torch
            min_len = min(len(test_audio), len(ref_audio))
            if min_len < 2048:
                return None
            t = torch.from_numpy(test_audio[:min_len].astype(np.float32))
            r = torch.from_numpy(ref_audio[:min_len].astype(np.float32))
            J, Q, J_fr = 8, 8, 3
            if self._jtfs_transform is None or self._jtfs_shape != min_len:
                self._jtfs_transform = TimeFrequencyScattering1D(
                    shape=min_len, J=J, Q=Q, J_fr=J_fr, Q_fr=1,
                    T=2 ** J, F=2 ** J_fr, out_type='array',
                )
                self._jtfs_shape = min_len
            jtfs = self._jtfs_transform
            Sx_test = jtfs(t)
            Sx_ref = jtfs(r)
            # Log-magnitude L2 distance, normalized
            log_test = np.log1p(np.abs(Sx_test))
            log_ref = np.log1p(np.abs(Sx_ref))
            num = np.linalg.norm(log_test - log_ref)
            den = np.linalg.norm(log_ref) + 1e-12
            return float(num / den)
        except Exception:
            return None

    def compare_with_reference(self, test_audio: np.ndarray,
                               ref_audio: np.ndarray,
                               category_mismatches: int = 0,
                               category_penalty: Optional[float] = None) -> Dict[str, float]:
        """
        Compare test audio with reference audio.

        Both arrays should already be preprocessed (trimmed, normalized) before
        calling this method.  The comparison is done on the active region of each
        signal independently, then on a common-length window for time-domain
        metrics.

        Args:
            test_audio:           Synthesized audio (preprocessed)
            ref_audio:            Reference audio (preprocessed)
            category_mismatches:  Count of mismatched categories (0-9). Used only
                                  as a fallback when ``category_penalty`` is None.
            category_penalty:     Continuous penalty in [0, 1] (mean normalized
                                  label distance across categories). Preferred over
                                  ``category_mismatches`` because it varies smoothly
                                  — a near-miss costs less than a 3-bin miss, so the
                                  composite score does not jump when a metric merely
                                  crosses a category threshold.

        Returns:
            Comparison metrics including a composite_score (lower is better).
        """
        metrics = {}

        # --- active-region spectral metrics (each signal on its own trimmed length) ---
        S_test = np.abs(librosa.stft(test_audio))
        S_ref = np.abs(librosa.stft(ref_audio))

        # Spectral convergence on active regions (pad shorter one to match)
        t_frames = S_test.shape[1]
        r_frames = S_ref.shape[1]
        if t_frames < r_frames:
            S_test_pad = np.pad(S_test, ((0, 0), (0, r_frames - t_frames)))
            S_ref_pad = S_ref
        else:
            S_test_pad = S_test
            S_ref_pad = np.pad(S_ref, ((0, 0), (0, t_frames - r_frames)))

        sc_num = np.linalg.norm(S_test_pad - S_ref_pad)
        sc_den = np.linalg.norm(S_ref_pad)
        metrics['spectral_convergence'] = float(sc_num / (sc_den + 1e-12))

        # Log-spectral distance (on padded frames)
        lsd = np.mean(np.sqrt(np.mean(
            (np.log10(S_test_pad + 1e-12) - np.log10(S_ref_pad + 1e-12)) ** 2,
            axis=0
        )))
        metrics['log_spectral_distance'] = float(lsd)

        # --- perceptual terms (primary): MFCC + mel-spectrogram distance ---
        # MFCC distance is robust to sub-bin pitch drift and aligns with human
        # timbre perception, unlike spectral_convergence which a few-Hz pitch
        # error can dominate. Both are normalized by the reference's own norm so
        # they sit on the same ~[0, 1+] scale as spectral_convergence.
        mfcc_test = librosa.feature.mfcc(y=test_audio, sr=self.sr, n_mfcc=20)
        mfcc_ref = librosa.feature.mfcc(y=ref_audio, sr=self.sr, n_mfcc=20)
        n_mfcc_frames = min(mfcc_test.shape[1], mfcc_ref.shape[1])
        if n_mfcc_frames > 0:
            mfcc_diff = mfcc_test[:, :n_mfcc_frames] - mfcc_ref[:, :n_mfcc_frames]
            metrics['mfcc_distance'] = float(
                np.linalg.norm(mfcc_diff) / (np.linalg.norm(mfcc_ref[:, :n_mfcc_frames]) + 1e-12)
            )
        else:
            metrics['mfcc_distance'] = 1.0

        mel_test = librosa.feature.melspectrogram(y=test_audio, sr=self.sr, n_mels=64)
        mel_ref = librosa.feature.melspectrogram(y=ref_audio, sr=self.sr, n_mels=64)
        mel_test_db = librosa.power_to_db(mel_test + 1e-12)
        mel_ref_db = librosa.power_to_db(mel_ref + 1e-12)
        n_mel_frames = min(mel_test_db.shape[1], mel_ref_db.shape[1])
        if n_mel_frames > 0:
            mel_diff = mel_test_db[:, :n_mel_frames] - mel_ref_db[:, :n_mel_frames]
            metrics['mel_distance'] = float(
                np.linalg.norm(mel_diff) / (np.linalg.norm(mel_ref_db[:, :n_mel_frames]) + 1e-12)
            )
        else:
            metrics['mel_distance'] = 1.0

        # --- spectral-shape terms: centroid + flatness distance ---
        # These give the optimizer counter-levers against noise-gaming: a
        # building low-frequency noise tail lowers the attempt centroid away
        # from a bright target, and changes its flatness. MFCC (time-integrated,
        # low-order) can reward such a tail for matching the target's energy
        # envelope; centroid/flatness distance penalize the wrong spectral shape.
        c_test = float(np.mean(librosa.feature.spectral_centroid(y=test_audio, sr=self.sr)[0]))
        c_ref = float(np.mean(librosa.feature.spectral_centroid(y=ref_audio, sr=self.sr)[0]))
        metrics['centroid_distance'] = float(min(abs(c_test - c_ref) / (c_ref + 1e-9), 2.0))

        # Band-wise flatness distance: whole-signal flatness is masked by strong
        # sine partials, so it misses a low-frequency noise layer substituting
        # for a missing fundamental. Comparing flatness per band (low vs high)
        # isolates that substitution — the attempt's low band goes flat (noisy)
        # where the target's is tonal, and this term penalizes it.
        from scipy import stats as _stats
        freqs_full = librosa.fft_frequencies(sr=self.sr, n_fft=2048)
        band_defs = [('low', 20, 250), ('high', 2000, 8000)]
        flat_dists = []
        excess = []   # attempt NOISIER (flatter) than target, per band
        deficit = []  # attempt MORE TONAL than target, per band
        for _name, flo, fhi in band_defs:
            mask = (freqs_full >= flo) & (freqs_full < fhi)
            if not np.any(mask):
                continue
            bt = S_test_pad[mask, :].flatten()
            br = S_ref_pad[mask, :].flatten()
            ft = float(_stats.gmean(bt + 1e-12) / (np.mean(bt) + 1e-12))
            fr = float(_stats.gmean(br + 1e-12) / (np.mean(br) + 1e-12))
            d = (np.log10(ft + 1e-12) - np.log10(fr + 1e-12)) / 3.0
            flat_dists.append(abs(d))
            excess.append(max(0.0, d))
            deficit.append(max(0.0, -d))
        metrics['flatness_distance'] = float(min(np.mean(flat_dists) if flat_dists else 0.0, 2.0))
        # Directional noise terms — drive the over-noise / too-tonal warnings.
        # noise_excess > 0: attempt has MORE broadband noise than target (reduce it).
        # noise_deficit > 0: attempt is MORE tonal/thin than target (add shaped noise).
        metrics['noise_excess'] = float(min(np.mean(excess) if excess else 0.0, 2.0))
        metrics['noise_deficit'] = float(min(np.mean(deficit) if deficit else 0.0, 2.0))

        # --- JTFS perceptual distance (approach #3) ---
        jtfs_d = self._jtfs_distance(test_audio, ref_audio)
        if jtfs_d is not None:
            metrics['jtfs_distance'] = float(min(jtfs_d, 2.0))

        # --- time-domain metrics on common-length window ---
        min_len = min(len(test_audio), len(ref_audio))
        t_td = test_audio[:min_len]
        r_td = ref_audio[:min_len]

        diff = t_td - r_td
        metrics['mse'] = float(np.mean(diff ** 2))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))

        signal_power = np.sum(r_td ** 2)
        noise_power = np.sum(diff ** 2)
        metrics['snr_db'] = float(10 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12)))

        # --- onset/envelope match ---
        # Compare onset strength envelopes (trimmed to common frame count)
        env_test = librosa.onset.onset_strength(y=test_audio, sr=self.sr)
        env_ref = librosa.onset.onset_strength(y=ref_audio, sr=self.sr)
        min_env = min(len(env_test), len(env_ref))
        env_diff = env_test[:min_env] - env_ref[:min_env]
        env_ref_norm = np.linalg.norm(env_ref[:min_env])
        metrics['envelope_distance'] = float(
            np.linalg.norm(env_diff) / (env_ref_norm + 1e-12)
        )

        # Peak onset strength — proxy for punchy attack (category attack_time).
        onset_max_test = float(np.max(env_test))
        onset_max_ref = float(np.max(env_ref))
        metrics['onset_max_test'] = onset_max_test
        metrics['onset_max_ref'] = onset_max_ref
        metrics['onset_max_penalty'] = float(
            min(abs(onset_max_test - onset_max_ref) / (onset_max_ref + 1e-12), 2.0)
        )

        # --- composite score (lower = better match) ---
        # MFCC is perceptually aligned and robust to sub-bin pitch drift, but it
        # can be GAMED by broadband/wrong-shape noise that fakes the target's
        # energy envelope while diverging in fine spectral structure. That fine-
        # structure mismatch shows up as spectral_convergence > 1.0 (the attempt
        # spectrum is nearly orthogonal to the target). So spectral_convergence
        # is made super-linear above 1.0 — an over-shape/wrong-noise penalty that
        # forces the optimizer to pull noise back down rather than pin noise
        # params at their maxima to lower MFCC.
        sc = metrics['spectral_convergence']
        # ponytail: 3.0 slope — seed selection uses sc > 1.0 as a
        # structural disqualifier (wrong-shape architecture). A steeper
        # slope makes the composite score reflect structural failure so
        # pick_seed_winner / _resolve_race naturally rank correct-spectrum
        # seeds above wrong-spectrum ones even under perceptual_heavy.
        spec_term = sc if sc <= 1.0 else 1.0 + 3.0 * (sc - 1.0)
        metrics['over_shape_penalty'] = float(max(0.0, sc - 1.0))

        if category_penalty is not None:
            cat_penalty = float(min(max(category_penalty, 0.0), 1.0))
        else:
            cat_penalty = min(category_mismatches / 9.0, 1.0)
        # JTFS reweight overrides loss multiplex config when JTFS is active.
        if 'jtfs_distance' in metrics:
            metrics['composite_score'] = float(
                0.20 * min(metrics['mfcc_distance'], 2.0)
                + 0.07 * min(metrics['mel_distance'], 2.0)
                + 0.15 * min(spec_term, 2.5)
                + 0.09 * metrics['centroid_distance']
                + 0.06 * metrics['flatness_distance']
                + 0.04 * min(metrics['log_spectral_distance'] / 10.0, 2.0)
                + 0.08 * min(metrics['envelope_distance'], 2.0)
                + 0.04 * metrics['onset_max_penalty']
                + 0.12 * cat_penalty
                + 0.15 * metrics['jtfs_distance']
            )
        elif _ACTIVE_LOSS_CONFIG and _ACTIVE_LOSS_CONFIG in LOSS_CONFIGS:
            metrics['composite_score'] = _composite_from_metrics(
                metrics, spec_term, cat_penalty, LOSS_CONFIGS[_ACTIVE_LOSS_CONFIG])
        else:
            metrics['composite_score'] = _composite_from_metrics(
                metrics, spec_term, cat_penalty)

        return metrics
    
    def categorize_metrics(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Categorize numeric metrics into human-readable labels.
        
        Args:
            metrics: Dictionary of metric names and values
            
        Returns:
            Dictionary of category names and labels
        """
        categories = {}
        for cat_name, cat_info in self.category_thresholds.items():
            metric_key = cat_info['metric']
            thresholds = cat_info['thresholds']
            labels = cat_info['labels']
            
            value = metrics.get(metric_key, 0.0)
            
            # Find which bin the value falls into
            label_idx = 0
            for t in thresholds:
                if value > t:
                    label_idx += 1
                else:
                    break
            
            categories[cat_name] = labels[min(label_idx, len(labels) - 1)]
        
        return categories


def evaluate_audio_file(audio_path: str, sample_rate=44100) -> Dict[str, float]:
    """
    Convenience function to evaluate audio from file.

    Applies shared preprocessing (silence trim + RMS normalization) before
    computing metrics so that evaluation is not skewed by trailing silence or
    level differences between target and attempt files.

    Args:
        audio_path:   Path to audio file
        sample_rate:  Target sample rate

    Returns:
        Dictionary of metrics
    """
    audio, _orig_dur = load_and_preprocess(audio_path, sr=sample_rate,
                                           normalize=True, trim_silence=True)
    evaluator = SynthesisEvaluator(sample_rate=sample_rate)
    return evaluator.evaluate(audio)


if __name__ == '__main__':
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python synthesis_evaluator_fixed.py <audio_file>")
        sys.exit(1)
    
    metrics = evaluate_audio_file(sys.argv[1])
    print(json.dumps(metrics, indent=2))

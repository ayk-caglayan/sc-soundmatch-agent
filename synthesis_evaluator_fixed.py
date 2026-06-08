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
    
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        
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
    
    def compare_with_reference(self, test_audio: np.ndarray,
                               ref_audio: np.ndarray,
                               category_mismatches: int = 0) -> Dict[str, float]:
        """
        Compare test audio with reference audio.

        Both arrays should already be preprocessed (trimmed, normalized) before
        calling this method.  The comparison is done on the active region of each
        signal independently, then on a common-length window for time-domain
        metrics.

        Args:
            test_audio:           Synthesized audio (preprocessed)
            ref_audio:            Reference audio (preprocessed)
            category_mismatches:  Optional count of mismatched categories (0-9)

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

        # --- composite score (lower = better match) ---
        # Weighted combination: spectral convergence is the primary term,
        # log_spectral_distance, envelope_distance, and category accuracy
        # add perceptual context.
        cat_penalty = min(category_mismatches / 9.0, 1.0)
        metrics['composite_score'] = float(
            0.4 * metrics['spectral_convergence']
            + 0.25 * min(metrics['log_spectral_distance'] / 10.0, 2.0)
            + 0.2 * min(metrics['envelope_distance'], 2.0)
            + 0.15 * cat_penalty
        )

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

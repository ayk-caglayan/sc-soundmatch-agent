#!/usr/bin/env python3
"""
Build the reference examples database for sc_claw_flucoma.

Reads paired audio_desc + sc_code files from sc_code2sound/scdoc_batch/,
creates combined reference files in sc_claw_flucoma/workspace/reference_examples/,
and builds an index.json mapping category combinations to example filenames.

Usage:
    python3 build_reference_index.py [--audio-desc DIR] [--sc-code DIR] [--output DIR]

Defaults:
    --audio-desc  /home/ayk/sc_code2sound/scdoc_batch/audio_desc/
    --sc-code     /home/ayk/sc_code2sound/scdoc_batch/sc_code/
    --output      /home/ayk/sc_claw_flucoma/workspace/reference_examples/
"""

import argparse
import json
import re
from pathlib import Path


AUDIO_DESC_DIR = Path('/home/ayk/sc_code2sound/scdoc_batch/audio_desc')
SC_CODE_DIR    = Path('/home/ayk/sc_code2sound/scdoc_batch/sc_code')
OUTPUT_DIR     = Path('/home/ayk/sc_claw_flucoma/workspace/reference_examples')

CATEGORY_KEYS = [
    'brightness',
    'attack_time',
    'harmonic_to_noise_ratio',
    'spectral_flux_normalized',
    'temporal_centroid',
    'crest_factor_db',
    'spectral_complexity_mean',
    'spectral_slope',
    'envelope_flatness',
]


def parse_categories(desc_text: str) -> dict:
    """Extract CATEGORIES section from an audio description file."""
    cats = {}
    in_cats = False
    for line in desc_text.splitlines():
        if line.strip() == '=== CATEGORIES ===':
            in_cats = True
            continue
        if in_cats:
            if line.strip().startswith('==='):
                break
            m = re.match(r'^(\w+):\s*(.+)$', line.strip())
            if m:
                cats[m.group(1)] = m.group(2).strip()
    return cats


def parse_description_line(desc_text: str) -> str:
    """Extract the one-line description from the DESCRIPTION section."""
    in_desc = False
    for line in desc_text.splitlines():
        if line.strip() == '=== DESCRIPTION ===':
            in_desc = True
            continue
        if in_desc:
            stripped = line.strip()
            if stripped.startswith('==='):
                break
            if stripped:
                return stripped
    return ''


def parse_synthesis_concepts(desc_text: str) -> str:
    """Extract the SYNTHESIS CONCEPTS section (compact form)."""
    lines = []
    in_section = False
    for line in desc_text.splitlines():
        if line.strip() == '=== SYNTHESIS CONCEPTS ===':
            in_section = True
            continue
        if in_section:
            if line.strip().startswith('==='):
                break
            lines.append(line)
    # Return only the first 10 lines to keep combined files compact
    return '\n'.join(lines[:10]).strip()


def category_key(cats: dict) -> str:
    """Create a short string key from all perceptually important categories."""
    parts = []
    for k in ['brightness', 'harmonic_to_noise_ratio', 'spectral_complexity_mean',
              'attack_time', 'spectral_flux_normalized',
              'temporal_centroid', 'crest_factor_db', 'envelope_flatness']:
        if k in cats:
            parts.append(cats[k])
    return '_'.join(parts)


def category_key_short(cats: dict) -> str:
    """5-term combo key (original narrow key, kept for backward compatibility)."""
    parts = []
    for k in ['brightness', 'harmonic_to_noise_ratio', 'spectral_complexity_mean',
              'attack_time', 'spectral_flux_normalized']:
        if k in cats:
            parts.append(cats[k])
    return '_'.join(parts)


def build_reference_examples(audio_desc_dir: Path, sc_code_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    desc_files = sorted(audio_desc_dir.glob('*.txt'))
    print(f"Found {len(desc_files)} audio description files in {audio_desc_dir}")

    index = {
        'examples': [],
        # Individual category buckets (all 9 categories now indexed)
        'by_brightness': {},
        'by_harmonic_to_noise_ratio': {},
        'by_spectral_complexity_mean': {},
        'by_attack_time': {},
        'by_spectral_flux_normalized': {},
        'by_temporal_centroid': {},
        'by_crest_factor_db': {},
        'by_envelope_flatness': {},
        'by_spectral_slope': {},
        # Combo keys
        'by_category_combo': {},        # full 8-term key
        'by_category_combo_short': {},  # original 5-term key (backward compat)
    }

    written = 0
    skipped = 0

    for desc_path in desc_files:
        stem = desc_path.stem
        code_path = sc_code_dir / (stem + '.scd')

        if not code_path.exists():
            skipped += 1
            continue

        desc_text = desc_path.read_text(encoding='utf-8')
        code_text = code_path.read_text(encoding='utf-8')

        cats = parse_categories(desc_text)
        if not cats:
            skipped += 1
            continue

        description_line = parse_description_line(desc_text)
        synth_concepts = parse_synthesis_concepts(desc_text)

        # Write combined reference file
        out_filename = stem + '.ref'
        out_path = output_dir / out_filename

        combined = (
            f"=== EXAMPLE: {stem} ===\n\n"
            f"DESCRIPTION: {description_line}\n\n"
            f"CATEGORIES:\n"
            + '\n'.join(f"  {k}: {v}" for k, v in sorted(cats.items()))
            + f"\n\nSYNTHESIS CONCEPTS (key values):\n{synth_concepts}\n\n"
            f"SUPERCOLLIDER CODE:\n{code_text.strip()}\n"
        )
        out_path.write_text(combined, encoding='utf-8')
        written += 1

        # Build index entries
        entry = {
            'file': out_filename,
            'stem': stem,
            'description': description_line,
            'categories': cats,
        }
        index['examples'].append(entry)

        # Index by individual category values (all categories)
        for cat_key_name, cat_val in cats.items():
            idx_key = f'by_{cat_key_name}'
            if idx_key in index:
                index[idx_key].setdefault(cat_val, []).append(out_filename)

        # Index by full 8-term combo key
        combo = category_key(cats)
        index['by_category_combo'].setdefault(combo, []).append(out_filename)

        # Index by original 5-term combo key (backward compatibility)
        combo_short = category_key_short(cats)
        index['by_category_combo_short'].setdefault(combo_short, []).append(out_filename)

    # Write index
    index_path = output_dir / 'index.json'
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2)

    print(f"Written {written} reference files to {output_dir}")
    print(f"Skipped {skipped} (no matching code file or no categories)")
    print(f"Index written to {index_path}")
    print(f"\nIndex summary:")
    print(f"  Total examples: {len(index['examples'])}")
    for k in ['by_brightness', 'by_harmonic_to_noise_ratio', 'by_spectral_complexity_mean',
              'by_attack_time', 'by_spectral_flux_normalized',
              'by_temporal_centroid', 'by_crest_factor_db',
              'by_envelope_flatness', 'by_spectral_slope']:
        vals = index[k]
        print(f"  {k}: {sorted(vals.keys())}")


def main():
    parser = argparse.ArgumentParser(description='Build sc_claw_flucoma reference examples database')
    parser.add_argument('--audio-desc', type=Path, default=AUDIO_DESC_DIR)
    parser.add_argument('--sc-code', type=Path, default=SC_CODE_DIR)
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    build_reference_examples(args.audio_desc, args.sc_code, args.output)


if __name__ == '__main__':
    main()

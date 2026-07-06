#!/usr/bin/env python3
"""Validate architecture templates pass pre_validate (and optionally render)."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from compare import (
    ARCHITECTURE_ORDER, ARCHITECTURE_TEMPLATES, SEED_FAMILIES, build_seeded_templates,
    dump_seed_templates, pick_seed_winner,
)
from pre_validate import validate

SCRIPT_DIR = Path(__file__).parent
DUMMY_PARTIALS = [(440.0, 0.8), (880.0, 0.5), (1320.0, 0.3), (1760.0, 0.2), (2200.0, 0.1)]


def collect_templates():
    """Generic + partial-seeded template blocks."""
    seeded = build_seeded_templates(DUMMY_PARTIALS)
    blocks = {}
    for name, code in ARCHITECTURE_TEMPLATES.items():
        blocks[f"{name}/generic"] = code
    for name, code in seeded.items():
        blocks[f"{name}/seeded"] = code
    return blocks


def check_pre_validate(blocks):
    failed = []
    for label, code in blocks.items():
        errors = validate(code)
        if errors:
            failed.append((label, errors))
    return failed


def check_seed_tiebreak():
    """flucoma_template wins when within SEED_TIEBREAK_EPS of best."""
    scores = {
        'flucoma_template': 0.7373,
        'chaos_noise': 0.7173,
        'subtractive': 0.8131,
    }
    if pick_seed_winner(scores) != 'flucoma_template':
        return ['expected flucoma_template tiebreak over chaos_noise']
    scores2 = {'flucoma_template': 0.80, 'chaos_noise': 0.7173}
    if pick_seed_winner(scores2) != 'chaos_noise':
        return ['expected chaos_noise when flucoma is not close']
    return []


def check_seed_families():
    """SEED_FAMILIES must be flucoma_template + ARCHITECTURE_ORDER."""
    if SEED_FAMILIES[0] != 'flucoma_template':
        return ['SEED_FAMILIES[0] must be flucoma_template']
    if SEED_FAMILIES[1:] != ARCHITECTURE_ORDER:
        return [f'SEED_FAMILIES[1:] drifted from ARCHITECTURE_ORDER: {SEED_FAMILIES[1:]}']
    return []


def check_dump_coverage():
    """dump_seed_templates must include every ARCHITECTURE_ORDER family."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'seed_templates.txt'
        dump_seed_templates(None, str(out))
        text = out.read_text(encoding='utf-8')
    missing = [f for f in ARCHITECTURE_ORDER if f"=== {f} ===" not in text]
    return missing


PRECEDENCE_BUGGY = """\
var sig, filtered, env;
env = EnvGen.kr(Env.perc(0.01, 1.5), doneAction: 2);
filtered = LPF.ar(SinOsc.ar(440), 2000);
filtered = filtered + WhiteNoise.ar(0.01) * EnvGen.kr(Env.perc(0.01, 1.5));
Out.ar(0, (filtered * env).dup);
"""

PRECEDENCE_OK = """\
var sig, filtered, env;
env = EnvGen.kr(Env.perc(0.01, 1.5), doneAction: 2);
filtered = LPF.ar(SinOsc.ar(440), 2000);
filtered = filtered + (WhiteNoise.ar(0.01) * EnvGen.kr(Env.perc(0.01, 1.5)));
Out.ar(0, (filtered * env).dup);
"""


def check_precedence_fixtures():
    """Precedence check rejects buggy layering, accepts parenthesized form."""
    failed = []
    if not validate(PRECEDENCE_BUGGY):
        failed.append(('precedence/buggy', 'expected errors, got none'))
    if validate(PRECEDENCE_OK):
        failed.append(('precedence/ok', validate(PRECEDENCE_OK)))
    return failed


def check_render(blocks, duration=1.0):
    """Wrap + sclang render; returns list of (label, error_msg)."""
    if not shutil.which('sclang'):
        print("sclang not found — skipping render check", file=sys.stderr)
        return []

    wrap = SCRIPT_DIR / 'wrap_for_recording.py'
    failed = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for label, code in blocks.items():
            scd = tmpdir / f"{label.replace('/', '_')}.scd"
            scd.write_text(code, encoding='utf-8')
            wrap_cmd = [
                sys.executable, str(wrap), str(scd), '-d', str(duration),
            ]
            r = subprocess.run(wrap_cmd, capture_output=True, text=True)
            if r.returncode != 0:
                err = (scd.with_name(scd.stem + '_error.txt').read_text(encoding='utf-8')
                       if scd.with_name(scd.stem + '_error.txt').exists()
                       else r.stderr or r.stdout)
                failed.append((label, err.strip()[:500]))
                continue

            nrt = scd.with_name(scd.stem + '_nrt.scd')
            env = {**os.environ, 'QT_QPA_PLATFORM': 'offscreen'}
            r = subprocess.run(
                ['timeout', '30', 'sclang', str(nrt)],
                capture_output=True, text=True, env=env, cwd=tmpdir,
            )
            wav = scd.with_suffix('.wav')
            if r.returncode != 0 or not wav.exists() or wav.stat().st_size < 1000:
                failed.append((label, (r.stderr or r.stdout or 'no wav').strip()[:500]))
    return failed


def main():
    parser = argparse.ArgumentParser(description='Self-check architecture templates')
    parser.add_argument('--render', action='store_true',
                        help='Also wrap and render each template with sclang')
    args = parser.parse_args()

    blocks = collect_templates()
    print(f"Checking {len(blocks)} template blocks…")

    seed_drift = check_seed_families()
    if seed_drift:
        print("SEED_FAMILIES FAILURES:", file=sys.stderr)
        for msg in seed_drift:
            print(f"  {msg}", file=sys.stderr)
        sys.exit(1)
    print(f"seed families: OK ({len(SEED_FAMILIES)} families)")

    tiebreak_fail = check_seed_tiebreak()
    if tiebreak_fail:
        print("SEED TIEBREAK FAILURES:", file=sys.stderr)
        for msg in tiebreak_fail:
            print(f"  {msg}", file=sys.stderr)
        sys.exit(1)
    print("seed tiebreak: OK")

    missing = check_dump_coverage()
    if missing:
        print("DUMP COVERAGE FAILURES:", file=sys.stderr)
        for f in missing:
            print(f"  missing family: {f}", file=sys.stderr)
        sys.exit(1)
    print(f"dump coverage: OK ({len(ARCHITECTURE_ORDER)} families)")

    prec_fail = check_precedence_fixtures()
    if prec_fail:
        print("PRECEDENCE FIXTURE FAILURES:", file=sys.stderr)
        for label, err in prec_fail:
            print(f"  {label}: {err}", file=sys.stderr)
        sys.exit(1)
    print("precedence fixtures: OK")

    pre_fail = check_pre_validate(blocks)
    if pre_fail:
        print("PRE-VALIDATE FAILURES:", file=sys.stderr)
        for label, errors in pre_fail:
            print(f"  {label}:", file=sys.stderr)
            for e in errors:
                print(f"    - {e}", file=sys.stderr)
        sys.exit(1)

    print("pre_validate: OK")

    if args.render:
        render_fail = check_render(blocks)
        if render_fail:
            print("RENDER FAILURES:", file=sys.stderr)
            for label, err in render_fail:
                print(f"  {label}: {err}", file=sys.stderr)
            sys.exit(1)
        print("render: OK")

    print("All templates passed.")


if __name__ == '__main__':
    main()

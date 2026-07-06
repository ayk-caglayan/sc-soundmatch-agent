#!/usr/bin/env python3
"""Write final_result.scd and report.md when the agent skips Finish."""

import argparse
import json
import re
from pathlib import Path


def best_from_comparisons(run_dir):
    best_attempt, best_score = None, None
    for comp in sorted(run_dir.glob('comparison_*.txt'), key=lambda p: int(p.stem.split('_')[1])):
        m = re.search(r'^composite_score:\s*([\d.]+)', comp.read_text(encoding='utf-8'), re.M)
        if not m:
            continue
        score = float(m.group(1))
        attempt = int(comp.stem.split('_')[1])
        if best_score is None or score < best_score:
            best_score, best_attempt = score, attempt
    return best_attempt, best_score


def write_report(run_dir, progress, best_attempt, best_score):
    report_path = run_dir / 'report.md'
    if report_path.exists():
        return

    config = {}
    cfg = run_dir / 'config.txt'
    if cfg.exists():
        for line in cfg.read_text(encoding='utf-8').splitlines():
            if ':' in line:
                k, v = line.split(':', 1)
                config[k.strip()] = v.strip()

    scores = progress.get('scores', []) if progress else []
    seed_scores = progress.get('seed_scores', {}) if progress else {}
    comp_count = len(list(run_dir.glob('comparison_*.txt')))

    lines = [
        '# Run Report (auto-generated)',
        '',
        f'- Iterations completed: {comp_count}',
        f'- Best attempt: {best_attempt} (composite_score: {best_score:.4f})' if best_attempt else '- Best attempt: unknown',
        f"- Convergence threshold: {config.get('convergence_threshold', 'N/A')}",
        f"- Max iterations: {config.get('max_iterations', 'N/A')}",
        '',
    ]

    if seed_scores:
        lines.append('## Seeding phase')
        for fam, sc in sorted(seed_scores.items(), key=lambda x: x[1]):
            mark = ' **WINNER**' if progress and progress.get('seed_winner_family') == fam else ''
            tie = ' (tiebreak)' if progress and progress.get('seed_winner_tiebreak') and mark else ''
            lines.append(f'- {fam}: {sc:.4f}{mark}{tie}')
        lines.append('')

    if best_attempt:
        comp = run_dir / f'comparison_{best_attempt}.txt'
        if comp.exists():
            text = comp.read_text(encoding='utf-8')
            lines.append('## Final metrics')
            for key in ('composite_score', 'spectral_convergence', 'envelope_distance',
                        'onset_max_penalty', 'snr_db'):
                m = re.search(rf'^{key}:\s*([\d.\-]+)', text, re.M)
                if m:
                    lines.append(f'- {key}: {m.group(1)}')
            lines.append('')

    lines.append('## Note')
    lines.append('Agent did not complete the Finish step; launcher/finish_run.py wrote this report.')
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {report_path}')


def main():
    parser = argparse.ArgumentParser(description='Finalize run artifacts')
    parser.add_argument('run_dir', help='Path to run directory')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f'Not a directory: {run_dir}')

    progress = None
    prog_path = run_dir / 'progress.json'
    if prog_path.exists():
        progress = json.loads(prog_path.read_text(encoding='utf-8'))

    best_attempt = progress.get('best_attempt') if progress else None
    best_score = progress.get('best_score') if progress else None
    if best_attempt is None:
        best_attempt, best_score = best_from_comparisons(run_dir)

    final = run_dir / 'final_result.scd'
    if not final.exists() and best_attempt is not None:
        src = run_dir / f'attempt_{best_attempt}.scd'
        if src.exists():
            final.write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
            print(f'Copied attempt_{best_attempt}.scd -> final_result.scd (score: {best_score})')

    if best_score is None and best_attempt is not None:
        _, best_score = best_from_comparisons(run_dir)

    write_report(run_dir, progress, best_attempt, best_score)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Fast Python-only pre-validation of SuperCollider synthesis code.

Catches the most common agent mistakes instantly (<100ms) without needing
sclang. Designed to run before wrap_for_recording.py to save time.

Checks:
  1. All `var` declarations are at the top (before any non-var statement)
  2. No assignments to undeclared variables
  3. All class names exist in sc_classes.txt
  4. Env segment times are plain numeric literals (not UGen-scaled)
  5. Env.adsr/perc/linen(...).kr(N) misuse is rejected
  6. At most one doneAction: 2 per synth body
"""

import sys
import os
import re
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
SC_CLASSES_PATH = SCRIPT_DIR / "sc_classes.txt"

_SC_CLASSES = None


def load_sc_classes():
    global _SC_CLASSES
    if _SC_CLASSES is None:
        if SC_CLASSES_PATH.exists():
            _SC_CLASSES = set(SC_CLASSES_PATH.read_text().strip().splitlines())
        else:
            _SC_CLASSES = set()
    return _SC_CLASSES


def check_var_declarations(code):
    """Ensure all var declarations are at the top, before any non-var statement."""
    errors = []
    lines = code.strip().splitlines()
    past_vars = False

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue

        if stripped.startswith("var "):
            if past_vars:
                errors.append(
                    f"Line {i}: `var` declaration after non-var statement. "
                    f"All var declarations must be at the top of the code."
                )
        else:
            past_vars = True

    return errors


def check_undeclared_variables(code):
    """Check for assignments to variables not declared with `var`."""
    errors = []

    declared = set()
    for match in re.finditer(r'\bvar\s+([^;]+);', code):
        var_list = match.group(1)
        for name in re.findall(r'(\w+)', var_list):
            declared.add(name)

    sc_keywords = {
        'nil', 'true', 'false', 'inf', 'pi', 'thisProcess', 'this',
        'SinOsc', 'Saw', 'Pulse', 'LPF', 'HPF', 'BPF', 'RLPF', 'RHPF',
        'EnvGen', 'Env', 'Out', 'Mix', 'Array', 'Pan2', 'Klang', 'DynKlang',
        'Klank', 'DynKlank', 'WhiteNoise', 'PinkNoise', 'BrownNoise',
        'GrayNoise', 'ClipNoise', 'Dust', 'Dust2', 'Crackle',
        'LFNoise0', 'LFNoise1', 'LFNoise2', 'LFDNoise0', 'LFDNoise1', 'LFDNoise3',
        'LFTri', 'LFSaw', 'LFPulse', 'LFCub', 'VarSaw', 'Impulse',
        'Blip', 'Formant', 'Ringz', 'Resonz', 'MoogFF',
        'FreeVerb', 'GVerb', 'CombL', 'CombC', 'CombN',
        'AllpassN', 'AllpassL', 'AllpassC',
        'Decay', 'Decay2', 'Line', 'XLine', 'Lag', 'Lag2', 'Lag3',
        'Pluck', 'Spring', 'BRF', 'Median', 'Slew',
        'FluidSines', 'FluidHPSS', 'FluidTransients', 'FluidSineFeature',
        'Select', 'LinLin', 'LinExp', 'Clip',
    }

    for match in re.finditer(r'^[ \t]*(\w+)\s*=\s*', code, re.MULTILINE):
        name = match.group(1)
        if name not in declared and name not in sc_keywords:
            line_num = code[:match.start()].count('\n') + 1
            errors.append(
                f"Line {line_num}: Assignment to undeclared variable `{name}`. "
                f"Add it to a `var` declaration at the top."
            )

    return errors


def _strip_line_comments(code):
    """Remove // line comments from code."""
    lines = []
    for line in code.splitlines():
        pos = line.find('//')
        lines.append(line[:pos] if pos >= 0 else line)
    return '\n'.join(lines)


def _split_top_level_args(s):
    """
    Split string s on commas that are not inside (), [], or {}.
    Returns a list of argument strings.
    """
    parts = []
    current = []
    depth = 0
    for ch in s:
        if ch in '([{':
            depth += 1
            current.append(ch)
        elif ch in ')]}':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current))
    return parts


def _find_balanced_call_args(code, call_start):
    """
    Given code and the position of the opening '(' of a call, return the
    contents between the balanced parens, or None if unbalanced.
    """
    if call_start >= len(code) or code[call_start] != '(':
        return None
    depth = 0
    for i in range(call_start, len(code)):
        if code[i] == '(':
            depth += 1
        elif code[i] == ')':
            depth -= 1
            if depth == 0:
                return code[call_start + 1:i]
    return None


# Patterns that indicate a time value is not a plain numeric literal.
_UGEN_TIME_PATTERNS = re.compile(
    r'\*\s*[a-zA-Z_]\w*'        # * someVar
    r'|[a-zA-Z_]\w*\s*\*'       # someVar *
    r'|\.kr\s*\('                # .kr(
    r'|\.ar\s*\('                # .ar(
    r'|\bLFNoise'                # LFNoise...
    r'|\bEnvGen\b'               # EnvGen inside times
    r'|\bLine\b'                 # Line UGen
    r'|\bXLine\b'                # XLine UGen
)


def check_envelope_times(code):
    """
    Check 3 envelope-related constraints:

    1. Env(levels, times, curves) — the 'times' argument must contain only
       numeric literals, not UGen expressions like `7.36 * aEnv`.
    2. Env.adsr/perc/linen/asr(...).kr(N) — calling .kr() directly on an
       Env object is not a valid SC API and typically indicates the agent
       tried to use it as a duration control.
    3. At most one `doneAction: 2` per synth body.
    """
    errors = []
    clean = _strip_line_comments(code)

    # --- Check 1: UGen-scaled Env segment times ---
    # Find all `Env(` call sites (not `Env.perc(` etc., those use positional args)
    for m in re.finditer(r'\bEnv\s*\(', clean):
        paren_start = m.end() - 1
        args_str = _find_balanced_call_args(clean, paren_start)
        if args_str is None:
            continue
        top_args = _split_top_level_args(args_str)
        if len(top_args) < 2:
            continue
        times_arg = top_args[1].strip()
        # times_arg should be a literal array like [7.36, 10.14]
        if _UGEN_TIME_PATTERNS.search(times_arg):
            line_num = clean[:m.start()].count('\n') + 1
            errors.append(
                f"Line {line_num}: Env segment times must be fixed numeric values "
                f"copied from target_partials.txt — do not multiply times by variables "
                f"or UGens (found: {times_arg.strip()[:60]}). "
                f"WRONG: [7.36 * aEnv, ...] CORRECT: [7.36, 10.14]"
            )

    # --- Check 2: Env.xxx(...).kr(N) misuse ---
    # Matches e.g. Env.adsr(...).kr(8) or Env.perc(...).kr(2)
    env_kr_pattern = re.compile(
        r'\bEnv\s*\.\s*(?:adsr|perc|linen|asr|cutoff|sine|triangle|pairs|circle)'
        r'\s*\([^)]*\)\s*\.\s*kr\s*\('
    )
    for m in env_kr_pattern.finditer(clean):
        line_num = clean[:m.start()].count('\n') + 1
        errors.append(
            f"Line {line_num}: Do not call .kr(N) on an Env object — that is not "
            f"a SuperCollider API. Use EnvGen.kr(Env.adsr(...), doneAction: 2) "
            f"instead, or keep the template's EnvGen.kr(Env([...],[...]), doneAction: 2)."
        )

    # --- Check 3: More than one doneAction: 2 ---
    done_action_count = len(re.findall(r'doneAction\s*:\s*2', clean))
    if done_action_count > 1:
        errors.append(
            f"Found {done_action_count} occurrences of `doneAction: 2` — only one "
            f"EnvGen should free the synth. Keep doneAction: 2 on the first/primary "
            f"partial envelope only and remove it from all others."
        )

    return errors


def check_class_names(code):
    """Check that all class names (capitalized identifiers used as UGens) exist."""
    errors = []
    sc_classes = load_sc_classes()
    if not sc_classes:
        return errors

    for match in re.finditer(r'\b([A-Z][a-zA-Z0-9]+)\b', code):
        name = match.group(1)
        if name in {'SynthDef', 'Score', 'ServerOptions', 'PathName', 'Date',
                     'Env', 'Array', 'Signal', 'Buffer', 'Bus', 'Server',
                     'Mix', 'Select', 'Out', 'In', 'Clip',
                     'LinLin', 'LinExp', 'NamedControl'}:
            continue
        if name not in sc_classes:
            line_num = code[:match.start()].count('\n') + 1
            errors.append(
                f"Line {line_num}: Unknown class `{name}`. "
                f"Not found in SuperCollider class index."
            )

    return errors


def validate(code):
    """Run all checks, return list of error strings."""
    errors = []
    errors.extend(check_var_declarations(code))
    errors.extend(check_undeclared_variables(code))
    errors.extend(check_class_names(code))
    errors.extend(check_envelope_times(code))
    return errors


def main():
    parser = argparse.ArgumentParser(description='Fast pre-validate SC synthesis code')
    parser.add_argument('input', help='Input .scd file with raw synthesis code')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    code = input_path.read_text(encoding='utf-8')
    errors = validate(code)

    if errors:
        error_path = input_path.with_name(input_path.stem + '_error.txt')
        report = (
            "SC CODE PRE-VALIDATION FAILED\n"
            "==============================\n\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\n\nFix these issues before running wrap_for_recording.py.\n"
        )
        error_path.write_text(report, encoding='utf-8')
        print(report, file=sys.stderr)
        sys.exit(1)

    print("Pre-validation passed.")


if __name__ == '__main__':
    main()

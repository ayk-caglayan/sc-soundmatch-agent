#!/usr/bin/env python3
"""
Wrap a SuperCollider synthesis script for Non-Real-Time (NRT) rendering.
Takes raw synthesis code and produces an NRT script that renders audio
to a WAV file without requiring any audio hardware.

The wrapper is tolerant of common agent mistakes:
- Strips { ... }.play, { ... }.dup, { ... }.dup(N) wrappers
- Adds Out.ar(0, ...) if missing
- Removes outer ( ) block wrappers

Three-layer validation:
- Layer 1:  Static class-name check against SC class index (instant, Python-side)
            Auto-corrects known typos and fuzzy-matches unknown class names.
- Layer 1b: Static parameter check against UGen signature database (instant, Python-side)
            Validates named arguments, checks positional arg counts, auto-fixes known
            parameter mistakes.
- Layer 2:  SynthDef build check via sclang subprocess (~0.3s)
            Actually builds a SynthDef with the code, catching runtime errors like
            non-existent methods, wrong keyword args (WARNING), and type errors.
"""

import sys
import re
import os
import json
import argparse
import subprocess
import tempfile
import difflib
from pathlib import Path


# ---------------------------------------------------------------------------
# SC class index (Layer 1)
# ---------------------------------------------------------------------------

_SC_CLASSES = None  # lazy-loaded set of all SC class names


def _load_sc_classes():
    """Load the SC class index from sc_classes.txt (next to this script)."""
    global _SC_CLASSES
    if _SC_CLASSES is not None:
        return _SC_CLASSES

    classes_file = Path(__file__).resolve().parent / 'sc_classes.txt'
    if not classes_file.exists():
        print(f"Warning: SC class index not found at {classes_file}", file=sys.stderr)
        _SC_CLASSES = set()
        return _SC_CLASSES

    _SC_CLASSES = set()
    for line in classes_file.read_text(encoding='utf-8').splitlines():
        name = line.strip()
        if name:
            _SC_CLASSES.add(name)
    return _SC_CLASSES


def _extract_class_references(code):
    """
    Extract class-name references from SC code.
    Returns a list of (class_name, line_number, column) tuples.
    """
    pattern = re.compile(r'\b([A-Z][A-Za-z0-9_]*)\s*\.')
    standalone = re.compile(r'\b([A-Z][A-Za-z0-9_]*)\s*\(')

    refs = []
    for lineno, line in enumerate(code.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith('//'):
            continue
        comment_pos = line.find('//')
        if comment_pos >= 0:
            line = line[:comment_pos]

        for m in pattern.finditer(line):
            refs.append((m.group(1), lineno, m.start()))
        for m in standalone.finditer(line):
            name = m.group(1)
            if (name, lineno, m.start()) not in refs:
                refs.append((name, lineno, m.start()))

    return refs


# Known typo -> correct class name mapping
_KNOWN_CLASS_TYPOS = {
    'SawOsc': 'Saw',
    'PulseOsc': 'Pulse',
    'SineOsc': 'SinOsc',
    'SinOscillator': 'SinOsc',
    'RndXExp': 'ExpRand',
    'RandXExp': 'ExpRand',
    'Rrand': 'Rand',
    'RandExp': 'ExpRand',
    'NoiseWhite': 'WhiteNoise',
    'NoisePink': 'PinkNoise',
    'NoiseBrown': 'BrownNoise',
    'LowPassFilter': 'LPF',
    'HighPassFilter': 'HPF',
    'BandPassFilter': 'BPF',
    'MoogFilter': 'MoogFF',
    'Sawtooth': 'Saw',
    'SquareWave': 'Pulse',
    'Triangle': 'LFTri',
    'Sine': 'SinOsc',
    'Noise': 'WhiteNoise',
    'PinkNois': 'PinkNoise',
    'Envelop': 'Env',
    'Envelope': 'Env',
    'EnvelopeGen': 'EnvGen',
    'Panner': 'Pan2',
    'Mixer': 'Mix',
    'FreeVerb2': 'FreeVerb',
    'Klang2': 'Klang',
}


def static_class_check(code):
    """
    Layer 1: Check all class references in the code against the SC class index.

    Returns:
        (fixed_code, warnings)
        - fixed_code: code with known typos auto-corrected
        - warnings: list of warning strings for unknown classes that couldn't be auto-fixed
    """
    sc_classes = _load_sc_classes()
    if not sc_classes:
        return code, []

    refs = _extract_class_references(code)
    warnings = []
    replacements = {}

    for class_name, lineno, col in refs:
        if class_name in sc_classes:
            continue

        if class_name in _KNOWN_CLASS_TYPOS:
            correct = _KNOWN_CLASS_TYPOS[class_name]
            replacements[class_name] = correct
            continue

        close = difflib.get_close_matches(class_name, sc_classes, n=1, cutoff=0.75)
        if close:
            replacements[class_name] = close[0]
        else:
            warnings.append(
                f"Line {lineno}: Unknown class '{class_name}' — not found in SuperCollider class index"
            )

    fixed_code = code
    for old_name, new_name in replacements.items():
        fixed_code = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, fixed_code)
        print(f"  Auto-fixed class: {old_name} -> {new_name}")

    return fixed_code, warnings


# ---------------------------------------------------------------------------
# UGen signature database (Layer 1b)
# ---------------------------------------------------------------------------

_UGEN_SIGNATURES = None  # lazy-loaded dict: "ClassName.method" -> [arg_names]


def _load_ugen_signatures():
    """Load the UGen signature database from sc_ugen_signatures.json."""
    global _UGEN_SIGNATURES
    if _UGEN_SIGNATURES is not None:
        return _UGEN_SIGNATURES

    sig_file = Path(__file__).resolve().parent / 'sc_ugen_signatures.json'
    if not sig_file.exists():
        print(f"Warning: UGen signature database not found at {sig_file}", file=sys.stderr)
        _UGEN_SIGNATURES = {}
        return _UGEN_SIGNATURES

    with open(sig_file, 'r', encoding='utf-8') as f:
        _UGEN_SIGNATURES = json.load(f)
    return _UGEN_SIGNATURES


# Known parameter name mistakes: (ClassName.method, wrong_param) -> correct_param
_KNOWN_PARAM_FIXES = {
    ('RLPF.ar', 'quality'): 'rq',
    ('RLPF.kr', 'quality'): 'rq',
    ('RLPF.ar', 'q'): 'rq',
    ('RLPF.kr', 'q'): 'rq',
    ('RHPF.ar', 'quality'): 'rq',
    ('RHPF.kr', 'quality'): 'rq',
    ('RHPF.ar', 'q'): 'rq',
    ('RHPF.kr', 'q'): 'rq',
    ('BPF.ar', 'quality'): 'rq',
    ('BPF.kr', 'quality'): 'rq',
    ('BPF.ar', 'q'): 'rq',
    ('BPF.kr', 'q'): 'rq',
    ('Resonz.ar', 'quality'): 'bwr',
    ('Resonz.kr', 'quality'): 'bwr',
    ('EnvGen.ar', 'done'): 'doneAction',
    ('EnvGen.kr', 'done'): 'doneAction',
    ('EnvGen.ar', 'doneaction'): 'doneAction',
    ('EnvGen.kr', 'doneaction'): 'doneAction',
    ('EnvGen.ar', 'env'): 'envelope',
    ('EnvGen.kr', 'env'): 'envelope',
    ('Env.perc', 'release'): 'releaseTime',
    ('Env.perc', 'attack'): 'attackTime',
    ('Env.linen', 'release'): 'releaseTime',
    ('Env.linen', 'attack'): 'attackTime',
    ('Env.linen', 'sustain'): 'sustainTime',
    ('Env.adsr', 'release'): 'releaseTime',
    ('Env.adsr', 'attack'): 'attackTime',
    ('Env.adsr', 'decay'): 'decayTime',
    ('Env.adsr', 'sustain'): 'sustainLevel',
}


def _extract_method_calls(code):
    """
    Extract method calls from SC code for parameter validation.

    Returns a list of dicts:
    {
        'class': 'SinOsc',
        'method': 'ar',
        'key': 'SinOsc.ar',
        'named_args': {'freq': '440', ...},  # named keyword args found
        'positional_count': 2,  # number of positional args
        'line': 5,
        'raw': 'SinOsc.ar(440, 0, 0.3)',
    }
    """
    # Match: ClassName.method( ... )
    # We need to handle nested parens to find the matching closing paren.
    call_pattern = re.compile(r'\b([A-Z][A-Za-z0-9_]*)\.([a-z][A-Za-z0-9_]*)\s*\(')

    calls = []
    lines = code.splitlines()

    for lineno, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if stripped.startswith('//'):
            continue
        comment_pos = line.find('//')
        check_line = line[:comment_pos] if comment_pos >= 0 else line

        for m in call_pattern.finditer(check_line):
            cls_name = m.group(1)
            method_name = m.group(2)
            paren_start = m.end() - 1  # position of '('

            # Find matching closing paren
            args_str = _extract_balanced_parens(check_line, paren_start)
            if args_str is None:
                # Multi-line call — accumulate lines until the opening paren
                # is balanced, then use _extract_balanced_parens so that only
                # the content *inside* this call's parens is captured.  This
                # prevents text after the closing paren (e.g. "* ampArray[i]")
                # from leaking into the argument string and inflating arg counts.
                remaining = check_line[paren_start:]
                for next_line in lines[lineno:]:
                    cp = next_line.find('//')
                    remaining += '\n' + (next_line[:cp] if cp >= 0 else next_line)
                    if _count_parens(remaining) == 0:
                        break
                args_str = _extract_balanced_parens(remaining, 0)
                if args_str is None:
                    continue  # genuinely unbalanced — let Layer 2 catch it

            # Parse the args string to find named args and count positional args
            named_args, positional_count = _parse_args(args_str)

            calls.append({
                'class': cls_name,
                'method': method_name,
                'key': f'{cls_name}.{method_name}',
                'named_args': named_args,
                'positional_count': positional_count,
                'line': lineno,
                'raw': f'{cls_name}.{method_name}({args_str.strip()[:60]}...)'
                       if len(args_str.strip()) > 60
                       else f'{cls_name}.{method_name}({args_str.strip()})',
            })

    return calls


def _extract_balanced_parens(text, start):
    """Extract content between balanced parens starting at position start."""
    if start >= len(text) or text[start] != '(':
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    return None  # unbalanced


def _count_parens(text):
    """Count net paren depth in text."""
    depth = 0
    for ch in text:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
    return depth


def _parse_args(args_str):
    """
    Parse a SC argument string to extract named args and count positional args.

    SC named args look like: name: value
    Positional args are everything else.

    Returns (named_args_dict, positional_count)
    """
    if not args_str or not args_str.strip():
        return {}, 0

    named_args = {}
    positional_count = 0

    # Split by commas at depth 0 (not inside nested parens/brackets)
    parts = _split_args(args_str)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Check for named arg pattern: identifier: value
        # But NOT inside strings or after operators like ::
        named_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)$', part, re.DOTALL)
        if named_match:
            arg_name = named_match.group(1)
            named_args[arg_name] = named_match.group(2).strip()
        else:
            positional_count += 1

    return named_args, positional_count


def _split_args(args_str):
    """Split argument string by commas at depth 0.

    Depth is clamped to >= 0 on closing characters so that a stray ')' or ']'
    (which should never appear in a correctly extracted arg string, but can
    occur after imperfect multi-line extraction) does not re-enable top-level
    comma splitting and inflate positional arg counts.
    """
    parts = []
    current = []
    depth = 0
    bracket_depth = 0

    for ch in args_str:
        if ch == '(' or ch == '[':
            depth += 1
            current.append(ch)
        elif ch == ')' or ch == ']':
            depth = max(depth - 1, 0)
            current.append(ch)
        elif ch == '{':
            bracket_depth += 1
            current.append(ch)
        elif ch == '}':
            bracket_depth = max(bracket_depth - 1, 0)
            current.append(ch)
        elif ch == ',' and depth == 0 and bracket_depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)

    if current:
        parts.append(''.join(current))

    return parts


def static_param_check(code):
    """
    Layer 1b: Check method call parameters against the UGen signature database.

    Validates:
    - Named arguments exist in the method signature
    - Positional argument count doesn't exceed signature length
    - Auto-fixes known parameter name mistakes

    Returns:
        (fixed_code, warnings)
        - fixed_code: code with known param mistakes auto-corrected
        - warnings: list of warning strings for parameter issues
    """
    signatures = _load_ugen_signatures()
    if not signatures:
        return code, []

    calls = _extract_method_calls(code)
    warnings = []
    text_replacements = []  # (old_text, new_text) pairs for auto-fix

    for call in calls:
        key = call['key']
        sig = signatures.get(key)
        if sig is None:
            # Method not in database — might be a non-UGen method, skip
            continue

        # Check named arguments
        for arg_name in call['named_args']:
            if arg_name in sig:
                continue  # valid named arg

            # Check known param fixes
            fix_key = (key, arg_name)
            if fix_key in _KNOWN_PARAM_FIXES:
                correct = _KNOWN_PARAM_FIXES[fix_key]
                text_replacements.append((
                    f'{arg_name}:',
                    f'{correct}:',
                    call['line'],
                    key,
                ))
                print(f"  Auto-fixed param: {key}({arg_name}: ...) -> {key}({correct}: ...)")
                continue

            # Fuzzy match against signature
            close = difflib.get_close_matches(arg_name, sig, n=1, cutoff=0.6)
            if close:
                text_replacements.append((
                    f'{arg_name}:',
                    f'{close[0]}:',
                    call['line'],
                    key,
                ))
                print(f"  Auto-fixed param (fuzzy): {key}({arg_name}: ...) -> {key}({close[0]}: ...)")
            else:
                warnings.append(
                    f"Line {call['line']}: Unknown parameter '{arg_name}' in {key}(). "
                    f"Valid parameters: {', '.join(sig)}"
                )

        # Check positional argument count
        # sig includes optional args like mul/add, so we use the full length
        max_args = len(sig)
        if call['positional_count'] > max_args:
            warnings.append(
                f"Line {call['line']}: Too many positional arguments in {key}() — "
                f"got {call['positional_count']}, max is {max_args}. "
                f"Parameters: {', '.join(sig)}"
            )

    # Apply text replacements
    fixed_code = code
    for old_text, new_text, lineno, method_key in text_replacements:
        # Replace only on the specific line to avoid false matches
        code_lines = fixed_code.splitlines()
        if 0 < lineno <= len(code_lines):
            code_lines[lineno - 1] = code_lines[lineno - 1].replace(old_text, new_text, 1)
            fixed_code = '\n'.join(code_lines)

    return fixed_code, warnings


# ---------------------------------------------------------------------------
# sclang SynthDef build validation (Layer 2)
# ---------------------------------------------------------------------------

# SynthDef build validation script template
_SCLANG_SYNTHDEF_VALIDATE_SCRIPT = '''\
var codeFile, code, testCode, result;
codeFile = File("{code_path}", "r");
code = codeFile.readAllString;
codeFile.close;
try {{
    testCode = "SynthDef(\\\\test, {{ " ++ code ++ " }})";
    result = testCode.interpret;
    if(result.notNil) {{
        "SYNTHDEF_BUILD_OK".postln;
    }} {{
        "SYNTHDEF_BUILD_FAILED".postln;
    }};
}} {{ |error|
    ("SYNTHDEF_BUILD_ERROR: " ++ error.errorString).postln;
}};
0.exit;
'''


def sclang_validate(code, error_file=None):
    """
    Layer 2: Validate SC code by building a SynthDef with it in sclang.

    This catches:
    - Undefined classes (ERROR)
    - Syntax errors (ERROR)
    - Non-existent methods (ERROR: Message 'foo' not understood)
    - Wrong keyword arguments (WARNING: keyword arg 'x' not found)
    - Any other runtime error during SynthDef construction

    Args:
        code: The sanitized SC code (SynthDef body)
        error_file: Optional path to write error details to

    Returns:
        (ok, error_message)
        - ok: True if validation passed with no errors or warnings
        - error_message: None if ok, otherwise the error/warning text from sclang
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.scd', delete=False,
                                     prefix='sc_validate_code_') as code_f:
        code_f.write(code)
        code_path = code_f.name

    escaped_path = code_path.replace('\\', '\\\\')
    validate_script = _SCLANG_SYNTHDEF_VALIDATE_SCRIPT.format(code_path=escaped_path)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.scd', delete=False,
                                     prefix='sc_validate_script_') as script_f:
        script_f.write(validate_script)
        script_path = script_f.name

    try:
        env = os.environ.copy()
        env['QT_QPA_PLATFORM'] = 'offscreen'

        result = subprocess.run(
            ['sclang', script_path],
            capture_output=True, text=True, timeout=15, env=env
        )

        stdout = result.stdout
        stderr = result.stderr
        combined = stdout + '\n' + stderr

        # Collect errors
        error_lines = []
        capture = False
        for line in combined.splitlines():
            if 'ERROR' in line:
                capture = True
            if capture:
                error_lines.append(line)
            if line.strip().startswith('---'):
                if capture:
                    error_lines.append(line)
                    capture = False

        # Collect warnings (e.g., "WARNING: keyword arg 'quality' not found")
        warning_lines = []
        for line in combined.splitlines():
            if line.strip().startswith('WARNING:'):
                warning_lines.append(line.strip())

        # Check for SYNTHDEF_BUILD_ERROR
        build_error_lines = []
        for line in combined.splitlines():
            if 'SYNTHDEF_BUILD_ERROR:' in line:
                build_error_lines.append(line.strip())

        # Determine result
        has_errors = bool(error_lines) or bool(build_error_lines) or 'SYNTHDEF_BUILD_FAILED' in combined
        has_warnings = bool(warning_lines)

        if not has_errors and not has_warnings and 'SYNTHDEF_BUILD_OK' in combined:
            return True, None

        # Build error message
        parts = []
        if error_lines:
            parts.append("ERRORS:\n" + '\n'.join(error_lines))
        if build_error_lines:
            parts.append("BUILD ERRORS:\n" + '\n'.join(build_error_lines))
        if warning_lines:
            parts.append("WARNINGS (treated as errors):\n" + '\n'.join(warning_lines))

        error_msg = '\n\n'.join(parts) if parts else "sclang validation failed (no specific error captured)"

        if error_file:
            Path(error_file).write_text(
                f"SC CODE VALIDATION FAILED (Layer 2: SynthDef build check)\n"
                f"=========================================================\n\n"
                f"sclang reported the following issue(s) when building a SynthDef:\n\n"
                f"{error_msg}\n\n"
                f"Fix the code in your attempt_N.scd and re-run the wrap step.\n"
                f"Common fixes:\n"
                f"  - Wrong keyword arg name -> check the UGen's documentation for correct param names\n"
                f"  - Message not understood -> you used a method that doesn't exist on that class\n"
                f"  - Class not defined -> use a standard SuperCollider UGen\n",
                encoding='utf-8'
            )

        return False, error_msg

    except subprocess.TimeoutExpired:
        error_msg = "sclang validation timed out (>15s) — code may contain an infinite loop"
        if error_file:
            Path(error_file).write_text(
                f"SC CODE VALIDATION FAILED\n"
                f"========================\n\n"
                f"{error_msg}\n",
                encoding='utf-8'
            )
        return False, error_msg

    except FileNotFoundError:
        print("Warning: sclang not found, skipping Layer 2 validation", file=sys.stderr)
        return True, None

    finally:
        try:
            os.unlink(code_path)
        except OSError:
            pass
        try:
            os.unlink(script_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Code sanitization
# ---------------------------------------------------------------------------

def strip_outer_parens(code):
    """Remove outer ( ) block wrapper, handling nested parens correctly."""
    code = code.strip()
    if not (code.startswith('(') and code.endswith(')')):
        return code
    depth = 0
    for i, ch in enumerate(code):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if depth == 0:
            if i == len(code) - 1:
                return code[1:-1].strip()
            else:
                return code
    return code


def sanitize_code(code):
    """
    Clean up agent-written SC code so it works inside a SynthDef for NRT rendering.

    Handles common patterns:
    - ( ... ) outer block wrappers (stripped repeatedly)
    - { ... }.play  -> extract body
    - { ... }.dup   -> extract body
    - { ... }.dup(N) -> extract body
    - Bare { ... }  -> extract body
    - Leading comments before { are preserved
    - Adds Out.ar(0, ...) if not present
    """
    code = code.strip()

    # Remove outer ( ) block wrapper — may be nested
    prev = None
    while prev != code:
        prev = code
        code = strip_outer_parens(code)

    # Separate leading comments from the code body
    lines = code.splitlines()
    leading_comments = []
    rest_lines = []
    in_comments = True
    for line in lines:
        stripped = line.strip()
        if in_comments and (stripped.startswith('//') or stripped == ''):
            leading_comments.append(line)
        else:
            in_comments = False
            rest_lines.append(line)

    rest_code = '\n'.join(rest_lines).strip()

    # Strip trailing .play, .dup, .dup(N) from a function block
    func_pattern = re.compile(
        r'^\{\s*(.*?)\s*\}\s*\.\s*(?:play|dup(?:\s*\(\s*\d*\s*\))?)\s*;?\s*$',
        re.DOTALL
    )
    m = func_pattern.match(rest_code)
    if m:
        rest_code = m.group(1).strip()

    # Also handle bare { ... } without .play/.dup
    bare_func = re.compile(r'^\{\s*(.*?)\s*\}\s*;?\s*$', re.DOTALL)
    m = bare_func.match(rest_code)
    if m:
        body = m.group(1).strip()
        if 'var ' in body or 'SinOsc' in body or 'Out.ar' in body or 'EnvGen' in body:
            rest_code = body

    # Strip ( ) again after unwrapping { }
    prev = None
    while prev != rest_code:
        prev = rest_code
        rest_code = strip_outer_parens(rest_code)

    # Reassemble with leading comments
    if leading_comments:
        code = '\n'.join(leading_comments) + '\n' + rest_code
    else:
        code = rest_code

    # Check if Out.ar is already present
    has_out = bool(re.search(r'Out\.ar\s*\(', code))

    if not has_out:
        code_lines = code.splitlines()
        last_stmt_idx = -1
        for i in range(len(code_lines) - 1, -1, -1):
            stripped = code_lines[i].strip()
            if stripped and not stripped.startswith('//'):
                last_stmt_idx = i
                break

        if last_stmt_idx >= 0:
            last_line = code_lines[last_stmt_idx].rstrip()
            if last_line.endswith(';'):
                last_line = last_line[:-1]
            code_lines[last_stmt_idx] = f"Out.ar(0, ({last_line}).dup);"
            code = '\n'.join(code_lines)
        else:
            code = f"Out.ar(0, ({code}).dup);"

    return code


# ---------------------------------------------------------------------------
# NRT wrapping
# ---------------------------------------------------------------------------

# Deterministic-render seed. Injected at the top of the SynthDef body (after the
# var declarations, which SC requires to come first) so that stochastic UGens
# (WhiteNoise, LFNoise*, Dust, ClipNoise, ...) produce byte-identical output on
# every render. Without this, the hill-climb and the parameter optimizer would be
# comparing render noise rather than real changes.
_DETERMINISM_SEED = "RandID.ir(0); RandSeed.ir(1, 56789);"


def inject_determinism(code, seed_stmt=_DETERMINISM_SEED):
    """Insert a one-shot RNG reseed after the leading `var` declarations.

    SuperCollider requires all `var` declarations at the top of a function body,
    so the seed statement is placed immediately after the last top-of-body `var`
    line (or before the first real statement if there are none).
    """
    if 'RandSeed' in code:
        return code  # already seeded

    lines = code.splitlines()
    insert_at = 0
    seen_code = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('//'):
            if not seen_code:
                insert_at = i + 1
            continue
        if stripped.startswith('var ') or stripped.startswith('var\t'):
            insert_at = i + 1
            seen_code = True
        else:
            if not seen_code:
                insert_at = i
            break

    lines.insert(insert_at, seed_stmt)
    return '\n'.join(lines)


def wrap_code(code, wav_filename, duration=10.0):
    """Generate the full NRT rendering script from sanitized SynthDef body code."""
    code = inject_determinism(code)
    indented_code = '\n'.join('    ' + line for line in code.splitlines())

    return f'''(
var score, dur = {duration}, outPath, oscPath;
outPath = thisProcess.nowExecutingPath.dirname +/+ "{wav_filename}";
oscPath = PathName.tmp +/+ "sc_nrt_" ++ Date.getDate.stamp ++ ".osc";

SynthDef(\\nrt_synth, {{
{indented_code}
}}).writeDefFile(SynthDef.synthDefDir);

score = Score([
    [0.0, [\\s_new, \\nrt_synth, 1000, 0, 0]],
    [dur, [\\c_set, 0, 0]]
]);

score.recordNRT(
    oscPath,
    outPath,
    sampleRate: 44100,
    headerFormat: "WAV",
    sampleFormat: "int24",
    options: ServerOptions.new
        .numOutputBusChannels_(2)
        .sampleRate_(44100),
    duration: dur,
    action: {{ "NRT render complete: %".format(outPath).postln; 0.exit }}
);
)
'''


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Wrap SC code for NRT rendering')
    parser.add_argument('input', help='Input .scd file with raw synthesis code')
    parser.add_argument('-o', '--output', help='Output .scd file (default: <input>_nrt.scd)')
    parser.add_argument('-d', '--duration', type=float, default=10.0,
                        help='Render duration in seconds (default: 10)')
    parser.add_argument('--skip-validate', action='store_true',
                        help='Skip sclang validation (Layer 2)')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    code = input_path.read_text(encoding='utf-8')
    wav_filename = input_path.stem + '.wav'

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + '_nrt.scd')

    error_path = input_path.with_name(input_path.stem + '_error.txt')

    # --- Step 1: Sanitize ---
    print(f"Sanitizing: {input_path}")
    code = sanitize_code(code)

    # --- Step 2: Layer 1 — static class-name check + auto-fix ---
    print("Validating (Layer 1): static class-name check...")
    code, class_warnings = static_class_check(code)

    if class_warnings:
        warning_text = '\n'.join(f"  WARNING: {w}" for w in class_warnings)
        print(warning_text)
        error_path.write_text(
            f"SC CODE VALIDATION FAILED (Layer 1: static class check)\n"
            f"======================================================\n\n"
            f"The following class names were not found in the SuperCollider class index:\n\n"
            + '\n'.join(f"  - {w}" for w in class_warnings) +
            f"\n\nFix the class names in your attempt_N.scd and re-run the wrap step.\n"
            f"Use valid SuperCollider UGen class names. The full SC class index (3473 classes)\n"
            f"is used for validation. Check sc_classes.txt for the complete list, or consult\n"
            f"the SuperCollider documentation at https://doc.sccode.org/\n",
            encoding='utf-8'
        )
        print(f"Error details written to: {error_path}", file=sys.stderr)
        sys.exit(1)

    # --- Step 3: Layer 1b — static parameter check + auto-fix ---
    print("Validating (Layer 1b): static parameter check...")
    code, param_warnings = static_param_check(code)

    if param_warnings:
        warning_text = '\n'.join(f"  WARNING: {w}" for w in param_warnings)
        print(warning_text)
        error_path.write_text(
            f"SC CODE VALIDATION FAILED (Layer 1b: parameter check)\n"
            f"=====================================================\n\n"
            f"The following parameter issues were found:\n\n"
            + '\n'.join(f"  - {w}" for w in param_warnings) +
            f"\n\nFix the parameters in your attempt_N.scd and re-run the wrap step.\n"
            f"Use only valid parameter names for each UGen. Check the SuperCollider\n"
            f"documentation for the correct argument names and order.\n",
            encoding='utf-8'
        )
        print(f"Error details written to: {error_path}", file=sys.stderr)
        sys.exit(1)

    # --- Step 4: Layer 2 — sclang SynthDef build validation ---
    if not args.skip_validate:
        print("Validating (Layer 2): sclang SynthDef build check...")
        ok, error_msg = sclang_validate(code, error_file=str(error_path))
        if not ok:
            print(f"VALIDATION FAILED: {error_msg}", file=sys.stderr)
            print(f"Error details written to: {error_path}", file=sys.stderr)
            sys.exit(1)
        print("  Validation passed.")
    else:
        print("  Skipping sclang validation (--skip-validate).")

    # --- Step 5: Generate NRT script ---
    wrapped = wrap_code(code, wav_filename, duration=args.duration)
    output_path.write_text(wrapped, encoding='utf-8')
    print(f"Wrapped (NRT): {output_path}")

    # Clean up any previous error file on success
    if error_path.exists():
        error_path.unlink()


if __name__ == '__main__':
    main()

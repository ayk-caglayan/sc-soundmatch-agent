#!/bin/bash
#
# sc_claw_flucoma launcher — sets up a run directory, runs FluCoMa analysis,
# and invokes the OpenClaw agent to iteratively match a target sound with
# SuperCollider synthesis.
#
# Usage:
#   ./launcher.sh --target /path/to/audio.wav [--max-iter 91] [--threshold 0.4] [--model MODEL]
#   ./launcher.sh --resume runs/YYYYMMDD_HHMMSS_basename [--max-iter N] [--model MODEL]
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
MAX_ITER=91
THRESHOLD=0.4
TIMEOUT_SEC=28800   # 8 hours
TARGET=""
TELEGRAM_NOTIFY=true
MODEL_ID="ollama/qwen3-coder-next:latest"
OPTIMIZER_BUDGET=30
SEED_COUNT=10
RESUME_DIR=""
RESUME_MODE=false
MAX_ITER_CLI_PASSED=false
CLI_MAX_ITER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --resume)
            RESUME_DIR="$2"
            RESUME_MODE=true
            shift 2
            ;;
        --max-iter)
            CLI_MAX_ITER="$2"
            MAX_ITER="$2"
            MAX_ITER_CLI_PASSED=true
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --model)
            MODEL_ID="$2"
            shift 2
            ;;
        --optimizer-budget)
            OPTIMIZER_BUDGET="$2"
            shift 2
            ;;
        --seed-count)
            SEED_COUNT="$2"
            shift 2
            ;;
        --no-telegram)
            TELEGRAM_NOTIFY=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --target <audio.wav> [options]"
            echo "       $0 --resume <run-dir> [--max-iter N] [options]"
            echo ""
            echo "Arguments:"
            echo "  --target       Path to target audio file (required for new runs)"
            echo "  --resume       Path to existing run dir (name, runs/..., or absolute)"
            echo "  --max-iter     Max scored steps (default: 91; on resume, extends budget if N > current step)"
            echo "  --threshold    Spectral convergence threshold (default: 0.4)"
            echo "  --model        Model id to use (default: ollama/qwen3-coder-next:latest)"
            echo "  --optimizer-budget  Renders per parameter-optimization step (default: 30)"
            echo "  --seed-count   Number of diverse architecture seeds before hill-climb (default: 10, set 0 to disable)"
            echo "  --no-telegram  Disable Telegram progress notifications"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

latest_scored_step() {
    python3 - "$1" <<'PYEOF'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
prog = run / 'progress.json'
if prog.exists():
    it = json.loads(prog.read_text(encoding='utf-8')).get('iteration')
    if it:
        print(it)
        sys.exit(0)
nums = [int(p.stem.split('_')[1]) for p in run.glob('comparison_*.txt')]
print(max(nums) if nums else 0)
PYEOF
}

resolve_run_dir() {
    local path="$1"
    if [ -d "$path" ]; then
        (cd "$path" && pwd)
    elif [ -d "${SCRIPT_DIR}/runs/${path}" ]; then
        (cd "${SCRIPT_DIR}/runs/${path}" && pwd)
    elif [ -d "${SCRIPT_DIR}/${path}" ]; then
        (cd "${SCRIPT_DIR}/${path}" && pwd)
    else
        echo "Error: run directory not found: $path" >&2
        exit 1
    fi
}

load_run_config() {
    python3 - "$1" <<'PYEOF'
import sys
from pathlib import Path
cfg = {}
for line in Path(sys.argv[1]).read_text(encoding='utf-8').splitlines():
    if ':' in line:
        k, v = line.split(':', 1)
        cfg[k.strip()] = v.strip()
for key in ('max_iterations', 'convergence_threshold', 'optimizer_budget',
            'seed_count', 'seed_optimizer_budget', 'target_duration'):
    print(cfg.get(key, ''))
PYEOF
}

patch_max_iterations() {
    python3 - "$1" "$2" <<'PYEOF'
import json, re, sys
from pathlib import Path
run_dir, new_max = Path(sys.argv[1]), int(sys.argv[2])
cfg_path = run_dir / 'config.txt'
text = cfg_path.read_text(encoding='utf-8')
if re.search(r'^max_iterations:', text, re.M):
    text = re.sub(r'^max_iterations:.*$', f'max_iterations: {new_max}', text, flags=re.M)
else:
    text = f'max_iterations: {new_max}\n' + text
cfg_path.write_text(text, encoding='utf-8')
prog_path = run_dir / 'progress.json'
if prog_path.exists():
    prog = json.loads(prog_path.read_text(encoding='utf-8'))
    prog['max_iterations'] = new_max
    prog['should_finish'] = False
    prog_path.write_text(json.dumps(prog, indent=2) + '\n', encoding='utf-8')
PYEOF
}

clear_stale_finish() {
    rm -f "$RUN_DIR/final_result.scd" "$RUN_DIR/report.md" "$RUN_DIR/ITERATION_LIMIT_REACHED"
}

tg_send() {
    $TELEGRAM_NOTIFY || return 0
    openclaw message send --channel telegram --target 876543184 --message "$1" 2>/dev/null || true
}

TG_MEDIA_STAGING="${HOME}/.openclaw/media/sc_claw_flucoma"

tg_send_media() {
    $TELEGRAM_NOTIFY || return 0
    local file="$1" caption="$2" as_document="${3:-false}"
    [ -f "$file" ] || return 0
    mkdir -p "$TG_MEDIA_STAGING"
    local base staged ext
    base=$(basename "$file")
    ext="${base##*.}"
    staged="$TG_MEDIA_STAGING/${TIMESTAMP}_${base}"
    if [ "$ext" = "scd" ]; then
        staged="$TG_MEDIA_STAGING/${TIMESTAMP}_${base%.scd}.txt"
        as_document=true
    fi
    cp "$file" "$staged"
    if [ "$as_document" = true ]; then
        openclaw message send --channel telegram --target 876543184 \
            --media "$staged" --message "$caption" --force-document 2>/dev/null || true
    else
        openclaw message send --channel telegram --target 876543184 \
            --media "$staged" --message "$caption" 2>/dev/null || true
    fi
    rm -f "$staged"
}

tg_send_best_results() {
    $TELEGRAM_NOTIFY || return 0
    [ -d "${RUN_DIR:-}" ] || return 0

    local best_info
    best_info=$(python3 - "$RUN_DIR" <<'PYEOF'
import json, re, sys
from pathlib import Path

run_dir = Path(sys.argv[1])
best_attempt = best_score = None
prog = run_dir / 'progress.json'
if prog.exists():
    p = json.loads(prog.read_text(encoding='utf-8'))
    best_attempt = p.get('best_attempt')
    best_score = p.get('best_score')
if best_attempt is None:
    for comp in sorted(run_dir.glob('comparison_*.txt'), key=lambda x: int(x.stem.split('_')[1])):
        m = re.search(r'^composite_score:\s*([\d.]+)', comp.read_text(encoding='utf-8'), re.M)
        if not m:
            continue
        score = float(m.group(1))
        attempt = int(comp.stem.split('_')[1])
        if best_score is None or score < best_score:
            best_score, best_attempt = score, attempt
if best_attempt is None:
    sys.exit(1)
score_s = f"{best_score:.4f}" if best_score is not None else "N/A"
print(f"{best_attempt} {score_s}")
PYEOF
) || return 0

    local best_attempt best_score
    read -r best_attempt best_score <<< "$best_info"
    local cap_base="$TARGET_BASENAME | best attempt $best_attempt | composite_score: $best_score"

    if [ -f "$RUN_DIR/target.wav" ]; then
        tg_send_media "$RUN_DIR/target.wav" "target.wav — $TARGET_BASENAME (reference)" false
    fi
    if [ -f "$RUN_DIR/final_result.scd" ]; then
        tg_send_media "$RUN_DIR/final_result.scd" "final_result.scd — $cap_base" true
    fi
    if [ -f "$RUN_DIR/attempt_${best_attempt}.wav" ]; then
        tg_send_media "$RUN_DIR/attempt_${best_attempt}.wav" "attempt_${best_attempt}.wav — $cap_base" false
    fi
}

RUN_STATUS="failed"
FAILURE_REASON="not started"
AGENT_EXIT_CODE=""
MONITOR_PID=""
AGENT_PID=""
ITERATION_LIMIT_STOP=false

finalize_run() {
    set +e

    if [ -n "${MONITOR_PID:-}" ]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi

    FINAL_COMP_COUNT=$(latest_scored_step "$RUN_DIR")
    FINAL_ATTEMPT_COUNT=$(find "$RUN_DIR" -maxdepth 1 -name 'attempt_*.scd' 2>/dev/null | wc -l)
    FINAL_SCORE=$(find "$RUN_DIR" -maxdepth 1 -name 'comparison_[0-9]*.txt' 2>/dev/null | sort -V | tail -1 \
        | xargs grep -m1 '^composite_score:\|^spectral_convergence:' 2>/dev/null | awk '{print $2}')

    if [ -n "$PREV_DEFAULT_MODEL" ]; then
        openclaw models set "$PREV_DEFAULT_MODEL" >/dev/null 2>&1 || true
    fi

    rm -f "${WORKSPACE_DIR}/current_run"

    tg_send "sc_claw_flucoma finished: $TARGET_BASENAME | status=$RUN_STATUS | steps=$FINAL_COMP_COUNT/$MAX_ITER | attempts=$FINAL_ATTEMPT_COUNT | best=${FINAL_SCORE:-N/A} | reason=$FAILURE_REASON"
}

if [ "$RESUME_MODE" = true ] && [ -n "$TARGET" ]; then
    echo "Error: --resume and --target are mutually exclusive"
    exit 1
fi

if [ "$RESUME_MODE" = false ] && [ -z "$TARGET" ]; then
    echo "Error: --target is required (or use --resume <run-dir>)"
    echo "Usage: $0 --target <audio.wav> [--max-iter N] [--threshold F] [--model MODEL]"
    echo "       $0 --resume <run-dir> [--max-iter N] [--model MODEL]"
    exit 1
fi

if [ "$RESUME_MODE" = false ] && [ ! -f "$TARGET" ]; then
    echo "Error: target file not found: $TARGET"
    exit 1
fi

case "$MODEL_ID" in
    qwen3-coder-next|qwen-coder|ollama/qwen3-coder-next*)
        MODEL_ID="ollama/qwen3-coder-next:latest"
        ;;
    gpt-5-mini|gpt5-mini|openai/gpt5-mini|gpt5mini)
        MODEL_ID="openai/gpt-5-mini"
        ;;
    claude-opus-4-6|claude|anthropic/claude-opus-4-6)
        MODEL_ID="anthropic/claude-opus-4-6"
        ;;
    claude-haiku-4-5|haiku|anthropic/claude-haiku-4-5)
        MODEL_ID="anthropic/claude-haiku-4-5"
        ;;
esac

AGENT_ID="sc_synth_flucoma"
PREV_DEFAULT_MODEL=$(openclaw config get agents.defaults.model.primary 2>/dev/null || true)

if [ "$RESUME_MODE" = true ]; then
    RUN_DIR=$(resolve_run_dir "$RESUME_DIR")
    RUN_BASENAME=$(basename "$RUN_DIR")
    TARGET="${RUN_DIR}/target.wav"
    if [[ "$RUN_BASENAME" =~ ^([0-9]{8}_[0-9]{6})_(.+)$ ]]; then
        TIMESTAMP="${BASH_REMATCH[1]}"
        TARGET_BASENAME="${BASH_REMATCH[2]}"
    else
        echo "Error: run directory name must match YYYYMMDD_HHMMSS_basename (got: $RUN_BASENAME)"
        exit 1
    fi
    SESSION_ID="run_${RUN_BASENAME}"

    if [ ! -f "$RUN_DIR/config.txt" ]; then
        echo "Error: missing config.txt in $RUN_DIR"
        exit 1
    fi
    if [ ! -f "$RUN_DIR/target.wav" ]; then
        echo "Error: missing target.wav in $RUN_DIR"
        exit 1
    fi
    if ! compgen -G "$RUN_DIR/comparison_*.txt" > /dev/null && [ ! -f "$RUN_DIR/progress.json" ]; then
        echo "Error: no comparison_*.txt or progress.json in $RUN_DIR — nothing to resume"
        exit 1
    fi

    mapfile -t _cfg < <(load_run_config "$RUN_DIR/config.txt")
    CONFIG_MAX_ITER="${_cfg[0]:-91}"
    MAX_ITER="$CONFIG_MAX_ITER"
    THRESHOLD="${_cfg[1]:-0.4}"
    OPTIMIZER_BUDGET="${_cfg[2]:-30}"
    SEED_COUNT="${_cfg[3]:-10}"
    SEED_OPT_BUDGET="${_cfg[4]:-10}"
    TARGET_DURATION="${_cfg[5]:-2.0}"

    CURRENT_STEP=$(latest_scored_step "$RUN_DIR")

    if [ "$MAX_ITER_CLI_PASSED" = true ]; then
        if [ "$CLI_MAX_ITER" -le "$CURRENT_STEP" ]; then
            echo "Error: --max-iter $CLI_MAX_ITER must be > current step $CURRENT_STEP"
            exit 1
        fi
        if [ "$CLI_MAX_ITER" -le "$CONFIG_MAX_ITER" ]; then
            echo "Error: --max-iter $CLI_MAX_ITER must be > config max_iterations ($CONFIG_MAX_ITER) to extend the budget"
            exit 1
        fi
        MAX_ITER="$CLI_MAX_ITER"
        echo "Extending step budget: $CURRENT_STEP/$CONFIG_MAX_ITER -> max $MAX_ITER"
        patch_max_iterations "$RUN_DIR" "$MAX_ITER"
    else
        if [ "$CURRENT_STEP" -ge "$MAX_ITER" ]; then
            echo "Step budget exhausted ($CURRENT_STEP/$MAX_ITER)."
            echo "Re-run with --max-iter N where N > $CURRENT_STEP to extend."
            exit 1
        fi
    fi

    clear_stale_finish

    HILL_CLIMB_SLOTS=$(( MAX_ITER - SEED_COUNT - 1 ))
    if [ "$HILL_CLIMB_SLOTS" -lt 0 ]; then
        HILL_CLIMB_SLOTS=0
    fi

    echo "============================================"
    echo "  sc_claw_flucoma — Resuming run"
    echo "============================================"
    echo "Run dir:      $RUN_DIR"
    echo "Target:       $TARGET"
    echo "Model:        $MODEL_ID"
    echo "Current step: $CURRENT_STEP"
    echo "Max iter:     $MAX_ITER"
    echo "Threshold:    $THRESHOLD"
    echo "============================================"
else
# Auto-adjust seed_count so at least one slot remains for Phase B
if [ "$SEED_COUNT" -ge "$MAX_ITER" ]; then
    ADJUSTED=$(( MAX_ITER > 1 ? MAX_ITER - 1 : 1 ))
    echo "Warning: seed_count $SEED_COUNT exceeds budget; reducing to $ADJUSTED (max_iter=$MAX_ITER must leave ≥1 slot for Phase B)"
    SEED_COUNT=$ADJUSTED
fi
HILL_CLIMB_SLOTS=$(( MAX_ITER - SEED_COUNT - 1 ))
if [ "$HILL_CLIMB_SLOTS" -lt 0 ]; then
    HILL_CLIMB_SLOTS=0
fi

# Create timestamped run directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TARGET_BASENAME=$(basename "$TARGET" .wav)
RUN_DIR="${SCRIPT_DIR}/runs/${TIMESTAMP}_${TARGET_BASENAME}"
mkdir -p "$RUN_DIR"
SESSION_ID="run_${TIMESTAMP}_${TARGET_BASENAME}"

# Seed optimizer budget: optimizer_budget / 3, minimum 8
SEED_OPT_BUDGET=$(( OPTIMIZER_BUDGET / 3 > 8 ? OPTIMIZER_BUDGET / 3 : 8 ))

echo "============================================"
echo "  sc_claw_flucoma — Sound Matching via OpenClaw + FluCoMa"
echo "============================================"
echo "Target:       $TARGET"
echo "Model:        $MODEL_ID"
echo "Max iter:     $MAX_ITER"
echo "Threshold:    $THRESHOLD"
echo "Opt budget:   $OPTIMIZER_BUDGET"
echo "Seed count:   $SEED_COUNT"
echo "Seed budget:  $SEED_OPT_BUDGET"
echo "Budget:       $SEED_COUNT seeds + 1 Phase B + $HILL_CLIMB_SLOTS hill-climb = $MAX_ITER total"
echo "Run dir:      $RUN_DIR"
echo "============================================"

# Copy target audio into run directory
cp "$TARGET" "$RUN_DIR/target.wav"
echo "Copied target audio to run directory."

# Pre-compute target evaluation
echo "Evaluating target audio..."
/home/ayk/miniconda3/bin/python3 "${SCRIPT_DIR}/evaluate.py" "$RUN_DIR/target.wav" -o "$RUN_DIR/target_eval.txt"
echo "Target evaluation saved."

# FluCoMa partials analysis
echo "Analyzing target partials (FluCoMa CLI)..."
/home/ayk/miniconda3/bin/python3 "${SCRIPT_DIR}/analyze_partials.py" "$RUN_DIR/target.wav" -o "$RUN_DIR/target_partials.txt"
echo "FluCoMa partials analysis saved."

# Pre-generate seed templates so the agent can copy them without guessing UGen APIs
echo "Generating seed templates..."
/home/ayk/miniconda3/bin/python3 "${SCRIPT_DIR}/compare.py" \
    --dump-templates "$RUN_DIR/seed_templates.txt" \
    --partials "$RUN_DIR/target_partials.txt"

# Measure target duration (seconds, rounded up to 1 decimal, minimum 2.0s)
TARGET_DURATION=$(/home/ayk/miniconda3/bin/python3 - "$RUN_DIR/target.wav" <<'PYEOF'
import sys, soundfile as sf, math
info = sf.info(sys.argv[1])
dur = max(2.0, math.ceil(info.duration * 10) / 10)
print(f"{dur:.1f}")
PYEOF
)
echo "Target duration: ${TARGET_DURATION}s"

# Write run config (includes target_duration so the agent can pass -d to wrap_for_recording.py)
cat > "$RUN_DIR/config.txt" <<EOF
max_iterations: $MAX_ITER
convergence_threshold: $THRESHOLD
target_duration: $TARGET_DURATION
optimizer_budget: $OPTIMIZER_BUDGET
seed_count: $SEED_COUNT
seed_optimizer_budget: $SEED_OPT_BUDGET
use_pnp: false
use_replay: false
use_sensitivity: false
use_neural_proxy: false
use_jtfs: false
envelope_seed: true
signal_chain_health: false
loss_config:
EOF
echo "Run config written."
fi

CURRENT_STEP=${CURRENT_STEP:-0}

echo ""
if [ "$RESUME_MODE" = true ]; then
    echo "Launching OpenClaw agent ($AGENT_ID) — resume at step $CURRENT_STEP/$MAX_ITER..."
else
    echo "Launching OpenClaw agent ($AGENT_ID)..."
fi
echo "============================================"
echo "Agent task: Iteratively refine SuperCollider synthesis to match target"
echo "  - Model: $MODEL_ID"
echo "  - Max steps: $MAX_ITER"
echo "  - Convergence goal: composite_score < $THRESHOLD"
echo "  - Timeout: ${TIMEOUT_SEC}s (8 hours)"
echo "  - Progress updates every 10s"
echo "============================================"

# Symlink the run directory into the agent's OpenClaw workspace (must match openclaw.json).
WORKSPACE_DIR=$(openclaw config get agents.list 2>/dev/null | python3 -c "
import json, sys
for a in json.load(sys.stdin):
    if a.get('id') == sys.argv[1]:
        print(a.get('workspace', ''))
        break
" "$AGENT_ID")
WORKSPACE_DIR="${WORKSPACE_DIR:-${SCRIPT_DIR}/workspace}"
if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "Error: agent workspace not found: $WORKSPACE_DIR"
    echo "Set agents.list workspace for $AGENT_ID in ~/.openclaw/openclaw.json"
    exit 1
fi
if [ -d "${WORKSPACE_DIR}/current_run" ] && [ ! -L "${WORKSPACE_DIR}/current_run" ]; then
    echo "Warning: removing stale current_run directory at ${WORKSPACE_DIR}/current_run"
    rm -rf "${WORKSPACE_DIR}/current_run"
fi
ln -sfn "$RUN_DIR" "${WORKSPACE_DIR}/current_run"
if [ ! -f "${WORKSPACE_DIR}/current_run/config.txt" ]; then
    echo "Error: agent cannot see run files via ${WORKSPACE_DIR}/current_run"
    exit 1
fi
trap finalize_run EXIT
echo "Linked ${WORKSPACE_DIR}/current_run -> $RUN_DIR"
if [ "$RESUME_MODE" = true ]; then
    tg_send "sc_claw_flucoma resumed: $TARGET_BASENAME | step=$CURRENT_STEP/$MAX_ITER | model=$MODEL_ID"
else
    tg_send "sc_claw_flucoma started: $TARGET_BASENAME | model=$MODEL_ID | max_steps=$MAX_ITER | threshold=$THRESHOLD"
fi

# Clear previous session to prevent context bloat (new runs only; resume preserves context).
AGENT_SESSION_DIR="${HOME}/.openclaw/agents/${AGENT_ID}/sessions"
if [ "$RESUME_MODE" = false ] && [ -d "$AGENT_SESSION_DIR" ]; then
    echo "Clearing previous session state..."
    rm -f "$AGENT_SESSION_DIR"/*.jsonl "$AGENT_SESSION_DIR"/*.jsonl.lock
    echo '{}' > "$AGENT_SESSION_DIR/sessions.json"
fi

cd "$RUN_DIR"

if ! openclaw models set "$MODEL_ID"; then
    FAILURE_REASON="failed to set requested model ($MODEL_ID)"
    echo "Error: OpenClaw could not select model: $MODEL_ID"
    exit 1
fi

# Start agent in background, then monitor progress and hard-stop at max_iter.
# Resume until iteration budget is exhausted, convergence, or final_result exists.

MAX_AGENT_ROUNDS=10
RUN_START_EPOCH=$(date +%s)
AGENT_EXIT_CODE=0
ITERATION_LIMIT_STOP=false
AGENT_ROUND=0

KICKOFF_MSG="Match the target sound. Your run directory is current_run/. Read current_run/config.txt (note seed_count=${SEED_COUNT} and seed_optimizer_budget=${SEED_OPT_BUDGET}), current_run/target_eval.txt, and current_run/target_partials.txt (FluCoMa analysis with ready-to-use SC templates). Follow AGENTS.md exactly. You have exactly ${MAX_ITER} scored steps total (${SEED_COUNT} seeds + Phase B + hill-climb). Phase A: produce ${SEED_COUNT} diverse architecture seeds (attempts 1..${SEED_COUNT}), one per family listed in the seeding table, each with a cheap optimizer pass (budget=${SEED_OPT_BUDGET}). Phase B: copy the best seed, run the full optimizer (budget=${OPTIMIZER_BUDGET}), then continue the hill-climb if budget remains. When comparison_N.txt contains MANDATORY FINISH, stop immediately and do the Finish step. Write all files to current_run/. IMPORTANT: When you reach max steps or convergence, you MUST do the Finish step (copy best attempt to final_result.scd and write report.md)."

start_monitor() {
    (
        set +eo pipefail
        sleep 5
        LAST_REPORTED=0
        LAST_ATTEMPT_REPORTED=0
        while kill -0 "$AGENT_PID" 2>/dev/null; do
            ATTEMPT_COUNT=$(find "$RUN_DIR" -maxdepth 1 -name 'attempt_*.scd' 2>/dev/null | wc -l)
            COMPARISON_COUNT=$(latest_scored_step "$RUN_DIR")

            if [ "$COMPARISON_COUNT" -ge "$MAX_ITER" ]; then
                echo "[$(date +%H:%M:%S)] Step limit reached ($COMPARISON_COUNT/$MAX_ITER) — stopping agent"
                touch "$RUN_DIR/ITERATION_LIMIT_REACHED"
                kill "$AGENT_PID" 2>/dev/null || true
                break
            fi

            if [ "$COMPARISON_COUNT" -gt "$LAST_REPORTED" ]; then
                LATEST_COMP=$(ls "$RUN_DIR"/comparison_[0-9]*.txt 2>/dev/null | sort -V | tail -1)
                LATEST_SCORE=$(grep -m1 '^composite_score:\|^spectral_convergence:' "$LATEST_COMP" 2>/dev/null | awk '{print $2}')
                echo "[$(date +%H:%M:%S)] Step $COMPARISON_COUNT/$MAX_ITER complete | score=${LATEST_SCORE:-N/A} | threshold=$THRESHOLD"
                tg_send "[$TARGET_BASENAME] Step $COMPARISON_COUNT/$MAX_ITER — composite_score: ${LATEST_SCORE:-N/A} (threshold: $THRESHOLD)"
                LAST_REPORTED=$COMPARISON_COUNT
            elif [ "$ATTEMPT_COUNT" -gt "$LAST_ATTEMPT_REPORTED" ]; then
                echo "[$(date +%H:%M:%S)] Attempt $ATTEMPT_COUNT drafted (step $COMPARISON_COUNT/$MAX_ITER scored so far)..."
                LAST_ATTEMPT_REPORTED=$ATTEMPT_COUNT
            fi

            sleep 10
        done
    ) &
    MONITOR_PID=$!
}

set +e

while true; do
  COMPLETED_COMPS=$(latest_scored_step "$RUN_DIR")
  COMPS_AT_ROUND_START=$COMPLETED_COMPS
  if [ "$COMPLETED_COMPS" -ge "$MAX_ITER" ]; then
    break
  fi
  if [ -f "$RUN_DIR/final_result.scd" ] && [ -f "$RUN_DIR/report.md" ]; then
    break
  fi
  if [ "$AGENT_ROUND" -ge "$MAX_AGENT_ROUNDS" ]; then
    echo "[$(date +%H:%M:%S)] Maximum agent rounds ($MAX_AGENT_ROUNDS) reached — stopping."
    break
  fi

    if [ "$AGENT_ROUND" -eq 0 ] && [ "$RESUME_MODE" = false ]; then
        AGENT_MSG="$KICKOFF_MSG"
    else
        CURRENT_STEP=$(latest_scored_step "$RUN_DIR")
        NEXT_ATTEMPT=$(( CURRENT_STEP + 1 ))
        RESUME_MSG="Continue the run per AGENTS.md. Read the latest comparison_N.txt in current_run/ and proceed from there. You are at step ${CURRENT_STEP}/${MAX_ITER}. Do NOT restart seeding. Write attempt_${NEXT_ATTEMPT}.scd next unless an unscored draft already exists."
        AGENT_MSG="$RESUME_MSG"
    fi

    # Compute remaining time budget
    NOW_EPOCH=$(date +%s)
    ELAPSED_SEC=$(( NOW_EPOCH - RUN_START_EPOCH ))
    REMAINING_SEC=$(( TIMEOUT_SEC - ELAPSED_SEC ))
    if [ "$REMAINING_SEC" -le 30 ]; then
        echo "No remaining time budget for agent resume — giving up."
        break
    fi

    echo "[$(date +%H:%M:%S)] Starting agent round $((AGENT_ROUND + 1))/$MAX_AGENT_ROUNDS (${COMPLETED_COMPS}/$MAX_ITER steps scored, timeout=${REMAINING_SEC}s)..."

    openclaw agent \
        --agent "$AGENT_ID" \
        --model "$MODEL_ID" \
        --session-id "$SESSION_ID" \
        --message "$AGENT_MSG" \
        --timeout "$REMAINING_SEC" &
    AGENT_PID=$!
    export AGENT_PID

    start_monitor

    wait $AGENT_PID
    AGENT_EXIT_CODE=$?

    if [ -f "$RUN_DIR/ITERATION_LIMIT_REACHED" ]; then
        ITERATION_LIMIT_STOP=true
    fi

    COMPLETED_COMPS=$(latest_scored_step "$RUN_DIR")

    if [ "$AGENT_EXIT_CODE" -eq 0 ] && [ "$COMPLETED_COMPS" -eq "$COMPS_AT_ROUND_START" ]; then
        if [ -f "$RUN_DIR/final_result.scd" ] && [ -f "$RUN_DIR/report.md" ] && [ "$COMPLETED_COMPS" -gt 0 ]; then
            break
        fi
        if [ "$COMPLETED_COMPS" -eq 0 ]; then
            echo "[$(date +%H:%M:%S)] Agent made no scored steps — stopping (check workspace symlink and AGENTS.md)."
            FAILURE_REASON="agent produced no comparison_N.txt files"
            RUN_STATUS="failed"
            break
        fi
    fi

    if [ "$AGENT_EXIT_CODE" -eq 0 ]; then
        if [ -f "$RUN_DIR/final_result.scd" ] && [ -f "$RUN_DIR/report.md" ]; then
            break
        fi
        if [ "$COMPLETED_COMPS" -ge "$MAX_ITER" ]; then
            break
        fi
    fi
    if [ "$ITERATION_LIMIT_STOP" = true ]; then
        break
    fi
    if [ -f "$RUN_DIR/final_result.scd" ] && [ -f "$RUN_DIR/report.md" ]; then
        break
    fi
    if [ "$COMPLETED_COMPS" -ge "$MAX_ITER" ]; then
        break
    fi

    AGENT_ROUND=$(( AGENT_ROUND + 1 ))
    echo "[$(date +%H:%M:%S)] Agent round $AGENT_ROUND exited (code $AGENT_EXIT_CODE, $COMPLETED_COMPS/$MAX_ITER steps scored) — resuming..."
    sleep 3
done

set -e

if [ "$AGENT_EXIT_CODE" -ne 0 ]; then
    if [ "$ITERATION_LIMIT_STOP" = true ]; then
        COMPLETED_COMPS=$(latest_scored_step "$RUN_DIR")
        if [ "$COMPLETED_COMPS" -ge "$MAX_ITER" ]; then
            RUN_STATUS="success"
            FAILURE_REASON="step limit reached (agent stopped)"
            echo "Agent stopped at step limit ($MAX_ITER comparisons completed)."
        fi
    elif [ "$AGENT_EXIT_CODE" -eq 124 ]; then
        FAILURE_REASON="launcher timeout (openclaw agent --timeout $TIMEOUT_SEC)"
    else
        FAILURE_REASON="openclaw agent exit code $AGENT_EXIT_CODE (after $AGENT_ROUND resume rounds)"
    fi
    if [ "$RUN_STATUS" != "success" ]; then
        echo "ERROR: OpenClaw agent failed."
        echo "  Exit code: $AGENT_EXIT_CODE"
        echo "  Reason:    $FAILURE_REASON"
    fi
else
    if [ "$RUN_STATUS" != "failed" ]; then
        RUN_STATUS="success"
        FAILURE_REASON="none"
    fi
fi

COMPLETED_ITERATIONS=$(latest_scored_step "$RUN_DIR")
if [ "$RUN_STATUS" = "success" ] && [ "$COMPLETED_ITERATIONS" -eq 0 ]; then
    RUN_STATUS="failed"
    FAILURE_REASON="agent exited without completing any scored step"
    echo "ERROR: OpenClaw agent exited but produced no completed steps."
fi

echo ""
echo "============================================"
echo "  Run complete: $RUN_DIR"
echo "============================================"

# Post-run: ensure final_result.scd and report.md exist
python3 "$SCRIPT_DIR/finish_run.py" "$RUN_DIR"

if [ ! -f "$RUN_DIR/final_result.scd" ]; then
    echo "Agent did not create final_result.scd — selecting best attempt..."
    BEST_ATTEMPT=""
    BEST_SCORE=""
    for comp_file in "$RUN_DIR"/comparison_[0-9]*.txt; do
        [ -f "$comp_file" ] || continue
        N=$(basename "$comp_file" | sed 's/comparison_\([0-9]*\)\.txt/\1/')
        SCORE=$(grep -m1 '^composite_score:\|^spectral_convergence:' "$comp_file" 2>/dev/null | awk '{print $2}')
        if [ -n "$SCORE" ]; then
            if [ -z "$BEST_SCORE" ] || python3 -c "exit(0 if $SCORE < $BEST_SCORE else 1)" 2>/dev/null; then
                BEST_SCORE="$SCORE"
                BEST_ATTEMPT="$N"
            fi
        fi
    done

    if [ -n "$BEST_ATTEMPT" ] && [ -f "$RUN_DIR/attempt_${BEST_ATTEMPT}.scd" ]; then
        cp "$RUN_DIR/attempt_${BEST_ATTEMPT}.scd" "$RUN_DIR/final_result.scd"
        echo "  -> Copied attempt_${BEST_ATTEMPT}.scd (score: ${BEST_SCORE}) as final_result.scd"
    else
        # Fallback: use the highest-numbered attempt
        LAST_ATTEMPT=$(find "$RUN_DIR" -maxdepth 1 -name 'attempt_*.scd' 2>/dev/null | sort -V | tail -1)
        if [ -n "$LAST_ATTEMPT" ]; then
            cp "$LAST_ATTEMPT" "$RUN_DIR/final_result.scd"
            echo "  -> Copied $(basename "$LAST_ATTEMPT") as final_result.scd (fallback)"
        else
            echo "Warning: no attempt files found."
        fi
    fi
fi

tg_send_best_results

if [ -f "$RUN_DIR/target.wav" ]; then
	tg_send_media "$RUN_DIR/target.wav" "target.wav — $TARGET_BASENAME (reference)" false
fi
if [ -f "$RUN_DIR/final_result.scd" ]; then
    echo "Final result: $RUN_DIR/final_result.scd"
fi

if [ -f "$RUN_DIR/report.md" ]; then
    echo "Report:       $RUN_DIR/report.md"
    echo ""
    cat "$RUN_DIR/report.md"
fi

# Print convergence summary
echo ""
echo "=== Convergence History ==="
for comp_file in "$RUN_DIR"/comparison_[0-9]*.txt; do
    [ -f "$comp_file" ] || continue
    N=$(basename "$comp_file" | sed 's/comparison_\([0-9]*\)\.txt/\1/')
    CSCORE=$(grep '^composite_score:' "$comp_file" 2>/dev/null | awk '{print $2}')
    SCSCORE=$(grep '^spectral_convergence:' "$comp_file" 2>/dev/null | awk '{print $2}')
    echo "  Step $N: composite_score = ${CSCORE:-N/A} | spectral_convergence = ${SCSCORE:-N/A}"
done

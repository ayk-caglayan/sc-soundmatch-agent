#!/bin/bash
#
# sc_claw_flucoma launcher — sets up a run directory, runs FluCoMa analysis,
# and invokes the OpenClaw agent to iteratively match a target sound with
# SuperCollider synthesis.
#
# Usage:
#   ./launcher.sh --target /path/to/audio.wav [--max-iter 85] [--threshold 0.4] [--model MODEL]
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
MAX_ITER=85
THRESHOLD=0.4
TIMEOUT_SEC=28800   # 8 hours
TARGET=""
TELEGRAM_NOTIFY=true
MODEL_ID="ollama/qwen3-coder-next:latest"
OPTIMIZER_BUDGET=30

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --max-iter)
            MAX_ITER="$2"
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
        --no-telegram)
            TELEGRAM_NOTIFY=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --target <audio.wav> [--max-iter N] [--threshold F] [--model MODEL] [--no-telegram]"
            echo ""
            echo "Arguments:"
            echo "  --target       Path to target audio file (required)"
            echo "  --max-iter     Maximum refinement iterations (default: 85)"
            echo "  --threshold    Spectral convergence threshold (default: 0.4)"
            echo "  --model        Model id to use (default: ollama/qwen3-coder-next:latest)"
            echo "  --optimizer-budget  Renders per parameter-optimization step (default: 30)"
            echo "  --no-telegram  Disable Telegram progress notifications"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

tg_send() {
    $TELEGRAM_NOTIFY || return 0
    openclaw message send --channel telegram --target 876543184 --message "$1" 2>/dev/null || true
}

RUN_STATUS="failed"
FAILURE_REASON="not started"
AGENT_EXIT_CODE=""
MONITOR_PID=""

finalize_run() {
    set +e

    if [ -n "${MONITOR_PID:-}" ]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi

    FINAL_COMP_COUNT=$(find "$RUN_DIR" -maxdepth 1 -name 'comparison_*.txt' 2>/dev/null | wc -l)
    FINAL_ATTEMPT_COUNT=$(find "$RUN_DIR" -maxdepth 1 -name 'attempt_*.scd' 2>/dev/null | wc -l)
    FINAL_SCORE=$(find "$RUN_DIR" -maxdepth 1 -name 'comparison_*.txt' 2>/dev/null | sort -V | tail -1 \
        | xargs grep -m1 '^composite_score:\|^spectral_convergence:' 2>/dev/null | awk '{print $2}')

    if [ -n "$PREV_DEFAULT_MODEL" ]; then
        openclaw models set "$PREV_DEFAULT_MODEL" >/dev/null 2>&1 || true
    fi

    rm -rf "${WORKSPACE_DIR}/current_run"

    tg_send "sc_claw_flucoma finished: $TARGET_BASENAME | status=$RUN_STATUS | iterations=$FINAL_COMP_COUNT/$MAX_ITER | attempts=$FINAL_ATTEMPT_COUNT | best=${FINAL_SCORE:-N/A} | reason=$FAILURE_REASON"
}

if [ -z "$TARGET" ]; then
    echo "Error: --target is required"
    echo "Usage: $0 --target <audio.wav> [--max-iter N] [--threshold F] [--model MODEL]"
    exit 1
fi

if [ ! -f "$TARGET" ]; then
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
esac

AGENT_ID="sc_synth_flucoma"
PREV_DEFAULT_MODEL=$(openclaw config get agents.defaults.model.primary 2>/dev/null || true)

# Create timestamped run directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TARGET_BASENAME=$(basename "$TARGET" .wav)
RUN_DIR="${SCRIPT_DIR}/runs/${TIMESTAMP}_${TARGET_BASENAME}"
mkdir -p "$RUN_DIR"

echo "============================================"
echo "  sc_claw_flucoma — Sound Matching via OpenClaw + FluCoMa"
echo "============================================"
echo "Target:       $TARGET"
echo "Model:        $MODEL_ID"
echo "Max iter:     $MAX_ITER"
echo "Threshold:    $THRESHOLD"
echo "Opt budget:   $OPTIMIZER_BUDGET"
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
EOF
echo "Run config written."

echo ""
echo "Launching OpenClaw agent ($AGENT_ID)..."
echo "============================================"
echo "Agent task: Iteratively refine SuperCollider synthesis to match target"
echo "  - Model: $MODEL_ID"
echo "  - Max iterations: $MAX_ITER"
echo "  - Convergence goal: composite_score < $THRESHOLD"
echo "  - Timeout: ${TIMEOUT_SEC}s (8 hours)"
echo "  - Progress updates every 30s"
echo "============================================"

# Symlink the run directory into the agent's workspace so the agent can read/write files.
# The agent's workspace is /home/ayk/sc_claw_flucoma/workspace — it can only access files inside it.
WORKSPACE_DIR="${SCRIPT_DIR}/workspace"
rm -rf "${WORKSPACE_DIR}/current_run"
ln -sfn "$RUN_DIR" "${WORKSPACE_DIR}/current_run"
trap finalize_run EXIT
echo "Linked workspace/current_run -> $RUN_DIR"
tg_send "sc_claw_flucoma started: $TARGET_BASENAME | model=$MODEL_ID | max_iter=$MAX_ITER | threshold=$THRESHOLD"

# Clear previous session to prevent context bloat.
# OpenClaw reuses the agent's main session across runs, accumulating 150K+ tokens
# which makes local models hang. Removing session files forces a fresh start.
AGENT_SESSION_DIR="${HOME}/.openclaw/agents/${AGENT_ID}/sessions"
if [ -d "$AGENT_SESSION_DIR" ]; then
    echo "Clearing previous session state..."
    rm -f "$AGENT_SESSION_DIR"/*.jsonl "$AGENT_SESSION_DIR"/*.jsonl.lock
    echo '{}' > "$AGENT_SESSION_DIR/sessions.json"
fi

SESSION_ID="run_${TIMESTAMP}_${TARGET_BASENAME}"
cd "$RUN_DIR"

if ! openclaw models set "$MODEL_ID"; then
    FAILURE_REASON="failed to set requested model ($MODEL_ID)"
    echo "Error: OpenClaw could not select model: $MODEL_ID"
    exit 1
fi

# Start a background monitor to show progress and send Telegram updates
(
    set +eo pipefail
    sleep 5
    LAST_REPORTED=0
    LAST_ATTEMPT_REPORTED=0
    while kill -0 $$ 2>/dev/null; do
        ATTEMPT_COUNT=$(find "$RUN_DIR" -maxdepth 1 -name 'attempt_*.scd' 2>/dev/null | wc -l)
        COMPARISON_COUNT=$(find "$RUN_DIR" -maxdepth 1 -name 'comparison_*.txt' 2>/dev/null | wc -l)

        if [ "$COMPARISON_COUNT" -gt "$LAST_REPORTED" ]; then
            LATEST_COMP=$(ls "$RUN_DIR"/comparison_*.txt 2>/dev/null | sort -V | tail -1)
            LATEST_SCORE=$(grep -m1 '^composite_score:\|^spectral_convergence:' "$LATEST_COMP" 2>/dev/null | awk '{print $2}')
            echo "[$(date +%H:%M:%S)] Iteration $COMPARISON_COUNT complete | score=${LATEST_SCORE:-N/A} | threshold=$THRESHOLD | progress=$COMPARISON_COUNT/$MAX_ITER"
            tg_send "[$TARGET_BASENAME] Iter $COMPARISON_COUNT/$MAX_ITER — composite_score: ${LATEST_SCORE:-N/A} (threshold: $THRESHOLD)"
            LAST_REPORTED=$COMPARISON_COUNT
        elif [ "$ATTEMPT_COUNT" -gt "$LAST_ATTEMPT_REPORTED" ]; then
            echo "[$(date +%H:%M:%S)] Iteration $ATTEMPT_COUNT started..."
            LAST_ATTEMPT_REPORTED=$ATTEMPT_COUNT
        fi

        sleep 10
    done
) &
MONITOR_PID=$!

set +e
openclaw agent \
    --agent "$AGENT_ID" \
    --session-id "$SESSION_ID" \
    --message "Match the target sound. Your run directory is current_run/. Read current_run/config.txt, current_run/target_eval.txt, and current_run/target_partials.txt (FluCoMa analysis with ready-to-use SC templates). Follow AGENTS.md exactly. Use Template D or E from target_partials.txt as your starting point. Write all files to current_run/. IMPORTANT: When you reach max iterations or convergence, you MUST do the Finish step (copy best attempt to final_result.scd and write report.md)." \
    --timeout $TIMEOUT_SEC
AGENT_EXIT_CODE=$?
set -e

if [ "$AGENT_EXIT_CODE" -ne 0 ]; then
    if [ "$AGENT_EXIT_CODE" -eq 124 ]; then
        FAILURE_REASON="launcher timeout (openclaw agent --timeout $TIMEOUT_SEC)"
    else
        FAILURE_REASON="openclaw agent exit code $AGENT_EXIT_CODE"
    fi
    echo "ERROR: OpenClaw agent failed."
    echo "  Exit code: $AGENT_EXIT_CODE"
    echo "  Reason:    $FAILURE_REASON"
else
    RUN_STATUS="success"
    FAILURE_REASON="none"
fi

COMPLETED_ITERATIONS=$(find "$RUN_DIR" -maxdepth 1 -name 'comparison_*.txt' 2>/dev/null | wc -l)
if [ "$RUN_STATUS" = "success" ] && [ "$COMPLETED_ITERATIONS" -eq 0 ]; then
    RUN_STATUS="failed"
    FAILURE_REASON="agent exited without completing any iteration"
    echo "ERROR: OpenClaw agent exited but produced no completed iterations."
fi

echo ""
echo "============================================"
echo "  Run complete: $RUN_DIR"
echo "============================================"

# Post-run: if agent didn't create final_result.scd, pick the best attempt
if [ ! -f "$RUN_DIR/final_result.scd" ]; then
    echo "Agent did not create final_result.scd — selecting best attempt..."
    BEST_ATTEMPT=""
    BEST_SCORE=""
    for comp_file in "$RUN_DIR"/comparison_*.txt; do
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
for comp_file in "$RUN_DIR"/comparison_*.txt; do
    [ -f "$comp_file" ] || continue
    N=$(basename "$comp_file" | sed 's/comparison_\([0-9]*\)\.txt/\1/')
    CSCORE=$(grep '^composite_score:' "$comp_file" 2>/dev/null | awk '{print $2}')
    SCSCORE=$(grep '^spectral_convergence:' "$comp_file" 2>/dev/null | awk '{print $2}')
    echo "  Iteration $N: composite_score = ${CSCORE:-N/A} | spectral_convergence = ${SCSCORE:-N/A}"
done

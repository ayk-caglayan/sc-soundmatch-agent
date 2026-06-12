#!/bin/bash
# Quick health checks for OpenClaw + model connectivity before a launcher run.
set -euo pipefail

AGENT_ID="${1:-sc_synth_flucoma}"
MODEL_ID="${2:-anthropic/claude-opus-4-8}"

echo "=== OpenClaw gateway ==="
openclaw gateway status 2>&1 | grep -E '^(Gateway|Runtime|Connectivity|Listening):' || openclaw gateway status

echo ""
echo "=== Model catalog (auth column must be 'yes') ==="
openclaw models list 2>&1 | grep -E "^${MODEL_ID//\//\\/}|^Model " || openclaw models list | head -5

echo ""
echo "=== Agent config model pin (overrides defaults if set) ==="
openclaw config get "agents.list" 2>/dev/null | python3 -c "
import json, sys
agent_id = sys.argv[1]
data = json.load(sys.stdin)
for a in data:
    if a.get('id') == agent_id:
        print(f'agent {agent_id}: model={a.get(\"model\", \"(inherits default)\")}')
        break
else:
    print(f'agent {agent_id}: not found')
" "$AGENT_ID" 2>/dev/null || echo "(could not read agent list)"

echo ""
echo "=== Live ping (agent + model override) ==="
echo "Sending: Reply with exactly: health-ok"
if timeout 90 openclaw agent \
    --agent "$AGENT_ID" \
    --model "$MODEL_ID" \
    --message "Reply with exactly: health-ok" \
    --timeout 75 2>&1 | tail -3; then
    echo "Ping succeeded."
else
    echo "Ping failed or timed out."
    exit 1
fi

echo ""
echo "=== Active sessions (watch for context >100%) ==="
openclaw status 2>&1 | sed -n '/^Sessions/,/^$/p' | head -12

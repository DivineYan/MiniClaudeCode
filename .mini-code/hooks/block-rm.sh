#!/usr/bin/env bash
# Blocks any Bash tool call that contains "rm " in the command.
# Exit code 2 = block the tool call and show the message below.

input="${MINI_CODE_TOOL_INPUT:-}"

if echo "$input" | grep -qE '"(rm |rm\b|-rf|--recursive)'; then
    echo "Blocked: rm commands are not allowed via hooks policy."
    exit 2
fi

exit 0

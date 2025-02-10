#!/bin/bash

# Enable strict mode
set -euo pipefail

# Log function
echo_log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Set script directory
SCRIPT_DIR="$(dirname "$0")"

# Validate environment
echo_log "[INFO] Validating environment..."
if ! command -v python3 &>/dev/null; then
    echo_log "[ERROR] Python 3 is not installed. Exiting."
    exit 1
fi

cd "$SCRIPT_DIR"

# Capture date argument if provided
DATE_ARG="${1:-$(date +'%Y-%m-%d')}"

echo_log "[INFO] Running Bronze job..."
python3 bronze_layer.py "$DATE_ARG"
if [[ $? -ne 0 ]]; then
    echo_log "[ERROR] Bronze job failed. Exiting."
    exit 2
fi

echo_log "[INFO] Running Silver job..."
python3 silver_layer.py "$DATE_ARG"
if [[ $? -ne 0 ]]; then
    echo_log "[ERROR] Silver job failed. Exiting."
    exit 3
fi

echo_log "[INFO] Running Gold job..."
python3 gold_layer.py "$DATE_ARG"
if [[ $? -ne 0 ]]; then
    echo_log "[ERROR] Gold job failed. Exiting."
    exit 4
fi

echo_log "[SUCCESS] Pipeline completed successfully."
exit 0
#!/usr/bin/env bash
# Train the example experiment using the main config.
# Usage: scripts/train.sh [path/to/config.yml]
set -euo pipefail

CONFIG="${1:-configs/experiment.yml}"

python -m repro_template.train --config "${CONFIG}"

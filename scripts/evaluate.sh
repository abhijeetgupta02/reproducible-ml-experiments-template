#!/usr/bin/env bash
# Evaluate a trained run (defaults to the most recent run).
# Usage: scripts/evaluate.sh [run_subdir] [path/to/config.yml]
set -euo pipefail

RUN="${1:-}"
CONFIG="${2:-configs/experiment.yml}"

if [[ -n "${RUN}" ]]; then
  python -m repro_template.evaluate --config "${CONFIG}" --run "${RUN}"
else
  python -m repro_template.evaluate --config "${CONFIG}"
fi

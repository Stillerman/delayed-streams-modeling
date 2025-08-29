#!/usr/bin/env bash
set -euo pipefail

# Print CUDA devices available (debug)
nvidia-smi || true

# Run server
exec moshi-server worker --config /app/configs/config-tts.toml --port 8000


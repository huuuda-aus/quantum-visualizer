#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8000}"

if [ ! -d "dist" ]; then
  echo "dist/ not found. Run ./build.sh first." >&2
  exit 1
fi

echo "Serving dist/ on: http://127.0.0.1:${PORT}/"
python3 -m http.server "${PORT}" --directory dist

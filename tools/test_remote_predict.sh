#!/usr/bin/env bash
# Usage: ./tools/test_remote_predict.sh https://your.url.here sample.wav
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <URL> <WAV_PATH>"
  exit 2
fi
URL="$1"
WAV="$2"
if command -v jq >/dev/null 2>&1; then
  curl -s -F file=@"${WAV}" "${URL%/}/predict" | jq .
else
  echo "Response from ${URL%/}/predict:" 
  curl -s -F file=@"${WAV}" "${URL%/}/predict"
  echo
fi

#!/usr/bin/env bash
# Collect diagnostic outputs from a deployed instance and optionally upload them.
# Usage: ./tools/collect_render_diag.sh https://accent-atlas.onrender.com [path/to/sample.wav]

set -euo pipefail
URL=${1:-}
WAV=${2:-sample_test.wav}
if [ -z "$URL" ]; then
  echo "Usage: $0 <URL> [wav_path]"
  exit 2
fi
if [ ! -f "$WAV" ]; then
  echo "WAV file '$WAV' not found in cwd. Please provide an existing WAV file as second arg." >&2
  exit 3
fi
OUTDIR=$(mktemp -d /tmp/accent_diag.XXXX)
echo "Collecting diagnostics for $URL into $OUTDIR"

# Files
DEBUG_RAW=$OUTDIR/debug.raw
DEBUG_ERR=$OUTDIR/debug.err
HEALTH_RAW=$OUTDIR/health.raw
HEALTH_ERR=$OUTDIR/health.err
VERSION_RAW=$OUTDIR/version.raw
VERSION_ERR=$OUTDIR/version.err
PRED_HEADERS=$OUTDIR/predict_headers.txt
PRED_BODY=$OUTDIR/predict_body.bin
PRED_ERR=$OUTDIR/predict_err.txt
UPLOAD_MAP=$OUTDIR/uploads.txt

# Collect endpoints
echo "-> GET /debug"
curl -v "$URL/debug" -o "$DEBUG_RAW" 2>"$DEBUG_ERR" || true

echo "-> GET /health"
curl -v "$URL/health" -o "$HEALTH_RAW" 2>"$HEALTH_ERR" || true

echo "-> GET /version"
curl -v "$URL/version" -o "$VERSION_RAW" 2>"$VERSION_ERR" || true

echo "-> POST /predict (this will upload $WAV)"
curl -v -H "Accept: application/json" -F file=@"$WAV" "$URL/predict" -D "$PRED_HEADERS" -o "$PRED_BODY" 2>"$PRED_ERR" || true

# Try uploading collected files to 0x0.st (simple paste service)
# If upload fails, we keep files locally and print paths.
echo "\nAttempting to upload collected files to https://0x0.st (may fail if blocked)"
> "$UPLOAD_MAP"
for f in "$DEBUG_RAW" "$DEBUG_ERR" "$HEALTH_RAW" "$HEALTH_ERR" "$VERSION_RAW" "$VERSION_ERR" "$PRED_HEADERS" "$PRED_BODY" "$PRED_ERR"; do
  if [ -f "$f" ]; then
    echo -n "Uploading $(basename "$f")... "
    url=$(curl -s -F file=@"$f" https://0x0.st || true)
    if [ -n "$url" ]; then
      echo "OK -> $url"
      echo "$(basename "$f") -> $url" >> "$UPLOAD_MAP"
    else
      echo "FAILED"
      echo "$(basename "$f") -> <local> $f" >> "$UPLOAD_MAP"
    fi
  fi
done

# Summary
cat <<EOF

Collection complete. Files saved in: $OUTDIR
Upload map (local file -> remote URL or local path):

$(cat "$UPLOAD_MAP")

If upload URLs are present, paste them here. Otherwise attach the following files or their contents:
- $HEALTH_RAW
- $VERSION_RAW
- $PRED_HEADERS
- $PRED_BODY (first 400 bytes shown below)

EOF

# show truncated body for quick inspection
if [ -f "$PRED_BODY" ]; then
  echo "--- First 400 bytes of predict body (hex) ---"
  head -c 400 "$PRED_BODY" | hexdump -C | sed -n '1,40p'
  echo
fi

exit 0

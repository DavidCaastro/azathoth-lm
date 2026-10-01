#!/usr/bin/env bash
# Downloads enwik8 (100MB, first 10^8 bytes of English Wikipedia XML dump)
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/../data" && pwd)"
ENWIK8="$DATA_DIR/enwik8"

if [ -f "$ENWIK8" ]; then
    echo "enwik8 already exists at $ENWIK8 ($(wc -c < "$ENWIK8") bytes)"
    exit 0
fi

echo "Downloading enwik8..."
curl -L "https://mattmahoney.net/dc/enwik8.zip" -o "$DATA_DIR/enwik8.zip"
echo "Extracting..."
unzip -o "$DATA_DIR/enwik8.zip" -d "$DATA_DIR"
rm "$DATA_DIR/enwik8.zip"
echo "Done: $ENWIK8 ($(wc -c < "$ENWIK8") bytes)"

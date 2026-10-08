#!/bin/bash
# T2b eval: 12 Silesia files × 100KB with --order-chain --neural-blend
# Saves state for each file to states/t2b-r56/
# Expected runtime: ~5-6h total

set -e

BINARY="cargo run --release --"
WEIGHTS="weights/rwkv7-0.1b"
BYTES=100000
FLAGS="--order-chain --neural-blend"
STATE_DIR="states/t2b-r56"
DATA_DIR="data/silesia"

FILES=(dickens mozilla mr nci ooffice osdb reymont samba sao webster x-ray xml)

echo "=== T2b R56: --order-chain --neural-blend, 100KB ==="
echo "Started: $(date)"
echo ""

for f in "${FILES[@]}"; do
    echo "--- $f ---"
    $BINARY hybrid-eval \
        --weights $WEIGHTS \
        --bytes $BYTES \
        --input "$DATA_DIR/$f" \
        $FLAGS \
        --save-state "$STATE_DIR/$f.bin" \
        2>&1 | grep -E "BPB:|time:|CM memory:|input:"
    echo ""
done

echo "=== T2b R56 COMPLETE ==="
echo "Finished: $(date)"

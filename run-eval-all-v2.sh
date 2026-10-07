#!/bin/bash
# Run T1b + T2b + T3 evals with telemetry + state serialization
# All 100KB except files smaller than 100KB (use full size)
set -e

BIN="./target/release/azathoth-lm.exe"
BASE="hybrid-eval --hierarchical --match --emb-surgery center0.3 --e8e9"
B100K="--bytes 100000"

mkdir -p docs/results/t1b docs/results/t2b docs/results/t3
mkdir -p states/t1b states/t2b states/t3

echo "=== T1b: enwik8 + OEIS (100KB) ==="
$BIN $BASE --input data/enwik8       $B100K --log docs/results/t1b/enwik8.jsonl --save-state states/t1b/enwik8.bin
$BIN $BASE --input data/oeis/stripped $B100K --log docs/results/t1b/oeis.jsonl   --save-state states/t1b/oeis.bin

echo "=== T2b: 12 Silesia (100KB) — includes T1b dickens/samba/mozilla ==="
for f in dickens mozilla mr nci ooffice osdb reymont samba sao webster x-ray xml; do
  echo "--- $f ---"
  $BIN $BASE --input "data/silesia/$f" $B100K --log "docs/results/t2b/$f.jsonl" --save-state "states/t2b/$f.bin"
done

echo "=== T3: AIT DCC A-H + local modern (100KB or full) ==="
for f in ait-A ait-B ait-C ait-D ait-E ait-F ait-G ait-H modern-x64-pe.bin; do
  echo "--- $f ---"
  $BIN $BASE --input "data/t3-modern/$f" $B100K --log "docs/results/t3/$f.jsonl" --save-state "states/t3/$f.bin"
done
# Small files: full size (< 100KB)
echo "--- ml-weights-safetensors.bin (10KB, full) ---"
$BIN $BASE --input "data/t3-modern/ml-weights-safetensors.bin" --bytes 10000 --log docs/results/t3/ml-weights-safetensors.jsonl --save-state states/t3/ml-weights-safetensors.bin
echo "--- structured-jsonl.bin (10KB, full) ---"
$BIN $BASE --input "data/t3-modern/structured-jsonl.bin" --bytes 10000 --log docs/results/t3/structured-jsonl.jsonl --save-state states/t3/structured-jsonl.bin

echo "=== ALL DONE ==="

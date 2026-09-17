#!/usr/bin/env bash
# Grow every example history and the morphology chart into ./out
set -euo pipefail
cd "$(dirname "$0")"
cargo build --release --quiet
mkdir -p out
for f in examples/*.toml; do
    name=$(basename "$f" .toml)
    echo "== $name"
    ./target/release/flakesim run "$f" --svg "out/$name.svg" --gif "out/$name.gif" --png "out/$name.png" --iso "out/$name.iso.png" --quiet
done
./target/release/flakesim diagram --out out/diagram.png > out/diagram.txt   # 0.5 µm cells, 3-D field: ~1.5 h on 64 cores
echo "outputs in ./out"

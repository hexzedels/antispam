#!/usr/bin/env bash
# Assemble a Hugging Face Space (Docker SDK) folder: Dockerfile and README at the root,
# same layout as the project root so service/Dockerfile works unchanged. Nothing is pushed.
#   service/space/build_space.sh [out_dir]    (default: service/space/out)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$ROOT/service/space/out}"

rm -rf "$OUT"
mkdir -p "$OUT/service" "$OUT/models" "$OUT/src/models"
cp "$ROOT/service/Dockerfile" "$OUT/Dockerfile"
cp "$ROOT/service/Dockerfile.dockerignore" "$OUT/.dockerignore"
cp "$ROOT/service/space/README.md" "$OUT/README.md"
cp -r "$ROOT/service/app" "$ROOT/service/__init__.py" "$ROOT/service/requirements.txt" "$OUT/service/"
cp "$ROOT/src/__init__.py" "$OUT/src/"
cp "$ROOT"/src/models/{__init__,features}.py "$OUT/src/models/"
[ -f "$ROOT/src/models/predict.py" ] && cp "$ROOT/src/models/predict.py" "$OUT/src/models/"
[ -d "$ROOT/src/category" ] && cp -r "$ROOT/src/category" "$OUT/src/"
cp "$ROOT/models/final_classic.joblib" "$OUT/models/"
[ -d "$ROOT/models/category" ] && cp -r "$ROOT/models/category" "$OUT/models/"
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} +

echo "Space folder: $OUT"
du -sh "$OUT"

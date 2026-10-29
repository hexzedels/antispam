#!/usr/bin/env bash
# Full training pipeline. Every step is skipped when its output already exists, so the
# script can be re-run after an interruption and continues from the first missing step.
#   scripts/run_all.sh            run missing steps
#   FORCE=step scripts/run_all.sh rerun one step (download|data|spambase|classic|e5|zeroshot|zeroshot_large|gold|export_e5|xlmr|category)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs

step() {  # step <name> <output that marks it done> <command...>
  local name=$1 done_file=$2; shift 2
  if [[ -e $done_file && ${FORCE:-} != "$name" ]]; then
    echo "[skip] $name ($done_file exists)"; return 0
  fi
  echo "[run ] $name -> logs/$name.log"
  if ! "$@" > "logs/$name.log" 2>&1; then
    echo "[FAIL] $name, see logs/$name.log"; tail -5 "logs/$name.log"; exit 1
  fi
  echo "[done] $name"
}

step download data/raw/.complete   bash -c "$PY -m src.data.download && touch data/raw/.complete"
step data     data/processed/course_test.parquet $PY -m src.data.build
step spambase reports/spambase_results.json      $PY -m src.models.spambase_baseline
step classic  models/final_classic.joblib        $PY -m src.models.train_classic
step e5       reports/e5small_results.json       $PY -m src.models.train_embed small
step zeroshot data/category/zs_mdeberta.parquet  $PY -m src.category.zeroshot --model mdeberta
step zeroshot_large data/category/zs_large.parquet $PY -m src.category.zeroshot --model large --batch 28
# Gold set: drawn once, labels filled and checked by hand (never overwritten).
step gold     data/category/gold.csv             $PY -m src.category.gold
step export_e5 models/category/e5/model.onnx     $PY -m src.category.export_e5
# Resumes inside the step too: finished folds are cached in models/xlmr_folds/.
step xlmr     reports/xlmr_results.json          $PY -m src.models.train_transformer
step category models/category/model.joblib       $PY -m src.category.train
echo "all steps done"

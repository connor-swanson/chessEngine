#!/usr/bin/env bash
# Run the whole training pipeline end to end, executing each notebook in place
# (outputs are saved into the .ipynb files).
#
#   cd train && ./run_pipeline.sh            # all four notebooks
#   ./run_pipeline.sh chess_mcts.ipynb       # just one
#
# Notebook 1 simulates the teacher's games into data/chess.db if needed, and
# each later notebook starts from checkpoints/sft.pt.  Total runtime on an
# Apple M4: about 2.5 hours, plus ~1 hour the first time to simulate games.
set -euo pipefail
cd "$(dirname "$0")"
NOTEBOOKS=("$@")
if [ ${#NOTEBOOKS[@]} -eq 0 ]; then
  NOTEBOOKS=(chess_sft.ipynb chess_reinforce.ipynb chess_ppo.ipynb chess_mcts.ipynb)
fi
for nb in "${NOTEBOOKS[@]}"; do
  echo "=== $(date '+%H:%M:%S') running $nb"
  start=$(date +%s)
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$nb"
  echo "=== $nb finished in $(( ($(date +%s) - start) / 60 )) min"
done

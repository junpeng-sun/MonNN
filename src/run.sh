#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${repo_root}/src/exps"

python_files=(
  "1_expsMLP.py"
  "2_expsWeightConstrained.py"
  "3_expsMM.py"
  "4_expsMMaux.py"
  "5_expsHLL.py"
  "6_expsUMNN.py"
  "7_expsLMN.py"
  "8_expsCoMNN.py"
  "9_expsSMNN.py"
  "10_expsPWL.py"
  "11_expsMixupPWL.py"
  "12_expsUniformPWL.py"
)

for file in "${python_files[@]}"; do
  echo "Running ${file}"
  python "${file}"
done

echo "All benchmark scripts completed."

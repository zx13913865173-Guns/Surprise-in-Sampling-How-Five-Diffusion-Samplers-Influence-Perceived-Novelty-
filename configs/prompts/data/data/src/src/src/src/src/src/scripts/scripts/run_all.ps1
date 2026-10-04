# Paper2 reproduction pipeline (Windows PowerShell)

Write-Host "=== Paper2 reproduction pipeline ==="

python src/generate.py --config configs/config.yaml --mode main
python src/generate.py --config configs/config.yaml --mode ablation
python src/generate.py --config configs/config.yaml --mode robustness

python src/compute_metrics.py --config configs/config.yaml

Rscript src/analyze_lmm.R
python src/analyze_lmm.py

python src/make_figures.py

Write-Host "=== Done ==="

## Beyond Additivity: Why Explaining Interactions in Machine Learning Survival Models is Difficult

Supplementary code for the AISTATS 2026 conference submission number 1322.

Create environment with the modified shapiq package (based on version 1.3.1).

```
conda env create -f env.yml
```

### Section 5.2: Experiments with real world data

Explain the models with `experiments/explain_actg.ipynb` and `experiments/explain_uvealmelanoma.ipynb`.

Run approximator benchmark with `sbatch experiments/run_approximators_benchmark.sh` (see `experiments/run_approximators_benchmark.py`).

Plot the benchmark results with `experiments/plot_approximators_benchmark.ipynb`.
## Beyond Additivity: Why Explaining Interactions in Machine Learning Survival Models is Difficult

Supplementary code for the AISTATS 2026 conference submission number 1322.

Create environment and install the modified shapiq package (based on version 1.3.1).

```
conda env create -f env.yml
conda activate survshapiq
cd shapiq .
pip install .
```

Explain the models with `explain_actg.ipynb` and `explain_uvealmelanoma.ipynb`.

Run approximator benchmark with `sbatch run_approximators_benchmark.sh` (see `run_approximators_benchmark.py`).

Plot the results with `plot_approximators_benchmark.ipynb`.
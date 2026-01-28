## Functional Decomposition and Shapley Interactions for Interpreting Survival Models

Supplementary code for the ICML 2026 conference submission number 1322.

### Section 5.1: Experiments with simulated data

Create environment with the modified shapiq package (based on version 1.3.1).

```
conda env create -f env.yml
```

Simulate data with `simulation_experiments/simulate_simsurv_final.r` and `simulation_experiments/simulate_simsurv_corr.r` (data in `simulation_experiments/data`).

Run experiments on randomly selected observation and plot results with `simulation_experiments/sim_survshapiq.py`, `simulation_experiments/sim_survshapiq_corr.py` and `simulation_experiments/sim_survshapiq_marg_cond.py`.

Obtain full explanations for simualted datasets with `simulation_experiments/sim_survshapiq_global.py` (results in `simulation_experiments/experiments`) and compute local accuracy results with `simulation_experiments/local_accuracy.py`.


### Section 5.2: Experiments with real world data

Create environment with the modified shapiq package (based on version 1.3.1).

```
conda env create -f env.yml
```

Explain the models with `experiments/explain_actg.ipynb` and `experiments/explain_uvealmelanoma.ipynb`.

Run approximator benchmark with `sbatch experiments/run_approximators_benchmark.sh` (see `experiments/run_approximators_benchmark.py`).

Plot the benchmark results with `experiments/plot_approximators_benchmark.ipynb`.

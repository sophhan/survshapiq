import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import simulation.survshapiq_func as survshapiq_func

# ----------------- Plotting function ----------------- #
def plot_global_shap(
    df,
    feature_cols,
    output_path="feature_trajectories.png",
    smooth=False,
    smooth_window=11,
    smooth_poly=3
):
    """
    Plot feature trajectories over time and save as PNG.
    """
    plt.figure(figsize=(10, 6))

    times = df["time"].values

    for col in feature_cols:
        y_vals = df[col].values

        # Apply smoothing if enabled
        if smooth and len(y_vals) >= smooth_window:
            y_vals = savgol_filter(y_vals, smooth_window, smooth_poly)

        plt.plot(times, y_vals, label=col, alpha=0.8)

    plt.xlabel("Time")
    plt.ylabel("Average SurvSHAPIQ attribution")
    plt.title("Global SurvSHAPIQ Values Over Time")
    plt.legend()
    plt.grid(True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

# ----------------- Processing pipeline ----------------- #
def process_and_plot(file_map, feature_cols, output_dir):
    """
    Load, average, and plot attributions for multiple datasets with dataset-specific parameters.

    Parameters
    ----------
    file_map : dict
        Mapping of dataset names to a dictionary with 'path' and optional plotting params:
        {
            "name": {"path": filepath, "smooth": True, "smooth_window": 50, "smooth_poly": 1},
            ...
        }
    feature_cols : list
        List of feature column names.
    output_dir : str
        Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    for name, options in file_map.items():
        print(f"Processing {name} ...")
        df = pd.read_csv(options["path"])
        df.columns = feature_cols + ["sample_idx", "time"]

        # Average over sample_idx
        avg_df = df.groupby("time", as_index=False).mean(numeric_only=True)

        # Output path
        output_path = os.path.join(output_dir, f"{name}.png")

        # Plot using dataset-specific options
        plot_global_shap(
            avg_df,
            feature_cols,
            output_path=output_path,
            smooth=options.get("smooth", False),
            smooth_window=options.get("smooth_window", 11),
            smooth_poly=options.get("smooth_poly", 3)
        )
        print(f"Saved plot: {output_path}")

# ----------------- Define feature columns ----------------- #
feature_cols = ['age', 'treatment', 'bmi', "age*treatment", "age*bmi", "treatment*bmi"]

# ----------------- Define datasets and parameters ----------------- #
file_map = {
    "cox_ti": {
        "path": "/home/slangbei/survshapiq/survshapiq/simulation/explanations/cox_attributions_ti.csv",
        "smooth": True,
        "smooth_window": 50,
        "smooth_poly": 1
    },
    "cox_td": {
        "path": "/home/slangbei/survshapiq/survshapiq/simulation/explanations/cox_attributions_td.csv",
        "smooth": True,
        "smooth_window": 50,
        "smooth_poly": 1
    },
    "gbsa_ti": {
        "path": "/home/slangbei/survshapiq/survshapiq/simulation/explanations/gbsa_attributions_ti.csv",
        "smooth": True,
        "smooth_window": 50,
        "smooth_poly": 1
    },
    "gbsa_td": {
        "path": "/home/slangbei/survshapiq/survshapiq/simulation/explanations/gbsa_attributions_td.csv",
        "smooth": True,
        "smooth_window": 50,
        "smooth_poly": 1
    },
    "hazard_ti": {
        "path": "/home/slangbei/survshapiq/survshapiq/simulation/explanations/hazard_attributions_ti.csv",
        "smooth": True,
        "smooth_window": 50,
        "smooth_poly": 1
    },
    "hazard_td": {
        "path": "/home/slangbei/survshapiq/survshapiq/simulation/explanations/hazard_attributions_td.csv",
        "smooth": True,
        "smooth_window": 50,
        "smooth_poly": 1
    },
    "log_hazard_ti": {
        "path": "/home/slangbei/survshapiq/survshapiq/simulation/explanations/log_hazard_attributions_ti.csv",
        "smooth": True,
        "smooth_window": 200,
        "smooth_poly": 1
    },
    "log_hazard_td": {
        "path": "/home/slangbei/survshapiq/survshapiq/simulation/explanations/log_hazard_attributions_td.csv",
        "smooth": True,
        "smooth_window": 50,
        "smooth_poly": 1
    },
    "survival_ti": {
        "path": "/home/slangbei/survshapiq/survshapiq/simulation/explanations/survival_attributions_ti.csv",
        "smooth": True,
        "smooth_window": 100,
        "smooth_poly": 1
    },
    "survival_td": {
        "path": "/home/slangbei/survshapiq/survshapiq/simulation/explanations/survival_attributions_td.csv",
        "smooth": True,
        "smooth_window": 100,
        "smooth_poly": 1
    }
}

# ----------------- Output directory ----------------- #
output_dir = "/home/slangbei/survshapiq/survshapiq/simulation/plots_global"

# ----------------- Run the pipeline ----------------- #
process_and_plot(file_map, feature_cols, output_dir)

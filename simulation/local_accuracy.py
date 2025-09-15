# load necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import os
import simulation.survshapiq_func as survshapiq_func
importlib.reload(survshapiq_func) 
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from collections import OrderedDict

# path settings
path_data = "/home/slangbei/survshapiq/survshapiq/simulation/data"
path_exp = "/home/slangbei/survshapiq/survshapiq/simulation/explanations"
path_combined = "/home/slangbei/survshapiq/survshapiq/simulation/plots_la/"
path_result = "/home/slangbei/survshapiq/survshapiq/simulation/plots_combined/"


# --- Global style settings ---
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# Color-blind friendly palette (Okabe–Ito)
cb_palette = ["#E69F00", "#56B4E9", "#009E73",
              "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]


#---------------------------
# 1) Linear G(t|x), TI (no interactions)
#---------------------------

# --- Dataset configs ---
datasets1 = [
    {"name": "GT h(t|x)",
     "exp_file": f"{path_exp}/hazard_attributions_linear_ti.csv",
     "data_file": f"{path_data}/1_simdata_linear_ti.csv",
     "survival_fn": hazard_wrap_linear_ti,
     "model": None},
    {"name": "GT log(h(t|x))",
     "exp_file": f"{path_exp}/log_hazard_attributions_linear_ti.csv",
     "data_file": f"{path_data}/1_simdata_linear_ti.csv",
     "survival_fn": log_hazard_wrap_linear_ti,
     "model": None},
    {"name": "GT S(t|x)",
     "exp_file": f"{path_exp}/survival_attributions_linear_ti.csv",
     "data_file": f"{path_data}/1_simdata_linear_ti.csv",
     "survival_fn": surv_from_hazard_linear_ti_wrap,
     "model": None},
    {"name": "CoxPH S(t|x)",
     "exp_file": f"{path_exp}/cox_attributions_linear_ti.csv",
     "data_file": f"{path_data}/1_simdata_linear_ti.csv",
     "survival_fn": None,
     "model": "Cox",
     "time_stride": 5},
    {"name": "GBSA S(t|x)",
     "exp_file": f"{path_exp}/gbsa_attributions_linear_ti.csv",
     "data_file": f"{path_data}/1_simdata_linear_ti.csv",
     "survival_fn": None,
     "model": "GBSA",
     "time_stride": 5},
]

def process_dataset(cfg):
    explanations_all = pd.read_csv(cfg["exp_file"])
    data_df = pd.read_csv(cfg["data_file"])
    return survshapiq_func.compute_local_accuracy(
        explanations_all,
        data_df,
        survival_fn=cfg.get("survival_fn"),
        model=cfg.get("model"),
        time_stride=cfg.get("time_stride", 1),
    )

if __name__ == "__main__":
    # Compute results
    results = []
    results_avg = []
    for d in datasets1:
        acc, times, avg = process_dataset(d)
        results.append((d["name"], times, acc))
        results_avg.append((d["name"], avg))

    # Plot
    plt.figure(figsize=(8, 5))
    for i, (name, times, acc) in enumerate(results):
        plt.plot(times, acc, lw=2, label=name,
                 color=cb_palette[i % len(cb_palette)])

    plt.xlabel("Time")
    plt.ylabel("Local Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path_combined}/1_la_linear_ti.pdf")
    plt.close()
    print(results_avg)
    
#---------------------------
# 2) Linear G(t|x), TD Main (no interactions)
#---------------------------

# --- Dataset configs ---
datasets2 = [
    {"name": "GT h(t|x)",
     "exp_file": f"{path_exp}/hazard_attributions_linear_tdmain.csv",
     "data_file": f"{path_data}/2_simdata_linear_tdmain.csv",
     "survival_fn": hazard_wrap_linear_tdmain,
     "model": None},
    {"name": "GT log(h(t|x))",
     "exp_file": f"{path_exp}/log_hazard_attributions_linear_tdmain.csv",
     "data_file": f"{path_data}/2_simdata_linear_tdmain.csv",
     "survival_fn": log_hazard_wrap_linear_tdmain,
     "model": None},
    {"name": "GT S(t|x): Linear G(t|x) TD Main",
     "exp_file": f"{path_exp}/survival_attributions_linear_tdmain.csv",
     "data_file": f"{path_data}/2_simdata_linear_tdmain.csv",
     "survival_fn": surv_from_hazard_linear_tdmain_wrap,
     "model": None},
    {"name": "CoxPH S(t|x)",
     "exp_file": f"{path_exp}/cox_attributions_linear_tdmain.csv",
     "data_file": f"{path_data}/2_simdata_linear_tdmain.csv",
     "survival_fn": None,
     "model": "Cox",
     "time_stride": 5},
    {"name": "GBSA S(t|x)",
     "exp_file": f"{path_exp}/gbsa_attributions_linear_tdmain.csv",
     "data_file": f"{path_data}/2_simdata_linear_tdmain.csv",
     "survival_fn": None,
     "model": "GBSA",
     "time_stride": 5},
]

def process_dataset(cfg):
    explanations_all = pd.read_csv(cfg["exp_file"])
    data_df = pd.read_csv(cfg["data_file"])
    return survshapiq_func.compute_local_accuracy(
        explanations_all,
        data_df,
        survival_fn=cfg.get("survival_fn"),
        model=cfg.get("model"),
        time_stride=cfg.get("time_stride", 1),
    )

if __name__ == "__main__":
    # Compute results
    results = []
    results_avg = []
    for d in datasets2:
        acc, times, avg = process_dataset(d)
        results.append((d["name"], times, acc))
        results_avg.append((d["name"], avg))

    # Plot
    plt.figure(figsize=(8, 5))
    for i, (name, times, acc) in enumerate(results):
        plt.plot(times, acc, lw=2, label=name,
                 color=cb_palette[i % len(cb_palette)])

    plt.xlabel("Time")
    plt.ylabel("Local Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path_combined}/la_linear_tdmain.pdf")
    plt.close()
    print(results_avg)
    
#---------------------------
# 3) Linear G(t|x), TI (interactions)
#---------------------------

# --- Dataset configs ---
datasets3 = [
    {"name": "GT h(t|x)",
     "exp_file": f"{path_exp}/hazard_attributions_linear_ti_inter.csv",
     "data_file": f"{path_data}/3_simdata_linear_ti_inter.csv",
     "survival_fn": hazard_wrap_linear_ti_inter,
     "model": None},
    {"name": "GT log(h(t|x))",
     "exp_file": f"{path_exp}/log_hazard_attributions_linear_ti_inter.csv",
     "data_file": f"{path_data}/3_simdata_linear_ti_inter.csv",
     "survival_fn": log_hazard_wrap_linear_ti_inter,
     "model": None},
    {"name": "GT S(t|x)",
     "exp_file": f"{path_exp}/survival_attributions_linear_ti_inter.csv",
     "data_file": f"{path_data}/3_simdata_linear_ti_inter.csv",
     "survival_fn": surv_from_hazard_linear_ti_inter_wrap,
     "model": None},
    {"name": "CoxPH S(t|x)",
     "exp_file": f"{path_exp}/cox_attributions_linear_ti_inter.csv",
     "data_file": f"{path_data}/3_simdata_linear_ti_inter.csv",
     "survival_fn": None,
     "model": "Cox",
     "time_stride": 5},
    {"name": "GBSA S(t|x)",
     "exp_file": f"{path_exp}/gbsa_attributions_linear_ti_inter.csv",
     "data_file": f"{path_data}/3_simdata_linear_ti_inter.csv",
     "survival_fn": None,
     "model": "GBSA",
     "time_stride": 5},
]

def process_dataset(cfg):
    explanations_all = pd.read_csv(cfg["exp_file"])
    data_df = pd.read_csv(cfg["data_file"])
    return survshapiq_func.compute_local_accuracy(
        explanations_all,
        data_df,
        survival_fn=cfg.get("survival_fn"),
        model=cfg.get("model"),
        time_stride=cfg.get("time_stride", 1),
    )

if __name__ == "__main__":
    # Compute results
    results = []
    results_avg = []
    for d in datasets3:
        acc, times, avg = process_dataset(d)
        results.append((d["name"], times, acc))
        results_avg.append((d["name"], avg))

    # Plot
    plt.figure(figsize=(8, 5))
    for i, (name, times, acc) in enumerate(results):
        plt.plot(times, acc, lw=2, label=name,
                 color=cb_palette[i % len(cb_palette)])

    plt.xlabel("Time")
    plt.ylabel("Local Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path_combined}/3_la_linear_ti_inter.pdf")
    plt.close()
    print(results_avg)
    
#---------------------------
# 4) Linear G(t|x), TD MAIN (interactions)
#---------------------------

# --- Dataset configs ---
datasets4 = [
    {"name": "GT h(t|x)",
     "exp_file": f"{path_exp}/hazard_attributions_linear_tdmain_inter.csv",
     "data_file": f"{path_data}/4_simdata_linear_tdmain_inter.csv",
     "survival_fn": hazard_wrap_linear_tdmain_inter,
     "model": None},
    {"name": "GT log(h(t|x))",
     "exp_file": f"{path_exp}/log_hazard_attributions_linear_tdmain_inter.csv",
     "data_file": f"{path_data}/4_simdata_linear_tdmain_inter.csv",
     "survival_fn": log_hazard_wrap_linear_tdmain_inter,
     "model": None},
    {"name": "GT S(t|x)",
     "exp_file": f"{path_exp}/survival_attributions_linear_tdmain_inter.csv",
     "data_file": f"{path_data}/4_simdata_linear_tdmain_inter.csv",
     "survival_fn": surv_from_hazard_linear_tdmain_inter_wrap,
     "model": None},
    {"name": "CoxPH S(t|x)",
     "exp_file": f"{path_exp}/cox_attributions_linear_tdmain_inter.csv",
     "data_file": f"{path_data}/4_simdata_linear_tdmain_inter.csv",
     "survival_fn": None,
     "model": "Cox",
     "time_stride": 5},
    {"name": "GBSA S(t|x)",
     "exp_file": f"{path_exp}/gbsa_attributions_linear_tdmain_inter.csv",
     "data_file": f"{path_data}/4_simdata_linear_tdmain_inter.csv",
     "survival_fn": None,
     "model": "GBSA",
     "time_stride": 5},
]

def process_dataset(cfg):
    explanations_all = pd.read_csv(cfg["exp_file"])
    data_df = pd.read_csv(cfg["data_file"])
    return survshapiq_func.compute_local_accuracy(
        explanations_all,
        data_df,
        survival_fn=cfg.get("survival_fn"),
        model=cfg.get("model"),
        time_stride=cfg.get("time_stride", 1),
    )

if __name__ == "__main__":
    # Compute results
    results = []
    results_avg = []
    for d in datasets4:
        acc, times, avg = process_dataset(d)
        results.append((d["name"], times, acc))
        results_avg.append((d["name"], avg))

    # Plot
    plt.figure(figsize=(8, 5))
    for i, (name, times, acc) in enumerate(results):
        plt.plot(times, acc, lw=2, label=name,
                 color=cb_palette[i % len(cb_palette)])

    plt.xlabel("Time")
    plt.ylabel("Local Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path_combined}/4_la_linear_tdmain_inter.pdf")
    plt.close()
    print(results_avg)

#---------------------------
# 5) Linear G(t|x), TD Inter (interactions)
#---------------------------

# --- Dataset configs ---
datasets5 = [
    {"name": "GT h(t|x)",
     "exp_file": f"{path_exp}/hazard_attributions_linear_tdinter.csv",
     "data_file": f"{path_data}/5_simdata_linear_tdinter.csv",
     "survival_fn": hazard_wrap_linear_tdinter,
     "model": None},
    {"name": "GT log(h(t|x))",
     "exp_file": f"{path_exp}/log_hazard_attributions_linear_tdinter.csv",
     "data_file": f"{path_data}/5_simdata_linear_tdinter.csv",
     "survival_fn": log_hazard_wrap_linear_tdinter,
     "model": None},
    {"name": "GT S(t|x)",
     "exp_file": f"{path_exp}/survival_attributions_linear_tdinter.csv",
     "data_file": f"{path_data}/5_simdata_linear_tdinter.csv",
     "survival_fn": surv_from_hazard_linear_tdinter_wrap,
     "model": None},
    {"name": "CoxPH S(t|x)",
     "exp_file": f"{path_exp}/cox_attributions_linear_tdinter.csv",
     "data_file": f"{path_data}/5_simdata_linear_tdinter.csv",
     "survival_fn": None,
     "model": "Cox",
     "time_stride": 5},
    {"name": "GBSA S(t|x)",
     "exp_file": f"{path_exp}/gbsa_attributions_linear_tdinter.csv",
     "data_file": f"{path_data}/5_simdata_linear_tdinter.csv",
     "survival_fn": None,
     "model": "GBSA",
     "time_stride": 5},
]

def process_dataset(cfg):
    explanations_all = pd.read_csv(cfg["exp_file"])
    data_df = pd.read_csv(cfg["data_file"])
    return survshapiq_func.compute_local_accuracy(
        explanations_all,
        data_df,
        survival_fn=cfg.get("survival_fn"),
        model=cfg.get("model"),
        time_stride=cfg.get("time_stride", 1),
    )

if __name__ == "__main__":
    # Compute results
    results = []
    results_avg = []
    for d in datasets5:
        acc, times, avg = process_dataset(d)
        results.append((d["name"], times, acc))
        results_avg.append((d["name"], avg))

    # Plot
    plt.figure(figsize=(8, 5))
    for i, (name, times, acc) in enumerate(results):
        plt.plot(times, acc, lw=2, label=name,
                 color=cb_palette[i % len(cb_palette)])

    plt.xlabel("Time")
    plt.ylabel("Local Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path_combined}/5_la_linear_tdinter.pdf")
    plt.close()
    print(results_avg)


#---------------------------
# 6) Generalized Additive G(t|x), TI (no interactions)
#---------------------------

# --- Dataset configs ---
datasets6 = [
    {"name": "GT h(t|x)",
     "exp_file": f"{path_exp}/hazard_attributions_genadd_ti.csv",
     "data_file": f"{path_data}/6_simdata_genadd_ti.csv",
     "survival_fn": hazard_wrap_genadd_ti,
     "model": None},
    {"name": "GT log(h(t|x))",
     "exp_file": f"{path_exp}/log_hazard_attributions_genadd_ti.csv",
     "data_file": f"{path_data}/6_simdata_genadd_ti.csv",
     "survival_fn": log_hazard_wrap_genadd_ti,
     "model": None},
    {"name": "GT S(t|x)",
     "exp_file": f"{path_exp}/survival_attributions_genadd_ti.csv",
     "data_file": f"{path_data}/6_simdata_genadd_ti.csv",
     "survival_fn": surv_from_hazard_genadd_ti_wrap,
     "model": None},
    {"name": "CoxPH S(t|x)",
     "exp_file": f"{path_exp}/cox_attributions_genadd_ti.csv",
     "data_file": f"{path_data}/6_simdata_genadd_ti.csv",
     "survival_fn": None,
     "model": "Cox",
     "time_stride": 5},
    {"name": "GBSA S(t|x)",
     "exp_file": f"{path_exp}/gbsa_attributions_genadd_ti.csv",
     "data_file": f"{path_data}/6_simdata_genadd_ti.csv",
     "survival_fn": None,
     "model": "GBSA",
     "time_stride": 5},
]

def process_dataset(cfg):
    explanations_all = pd.read_csv(cfg["exp_file"])
    data_df = pd.read_csv(cfg["data_file"])
    return survshapiq_func.compute_local_accuracy(
        explanations_all,
        data_df,
        survival_fn=cfg.get("survival_fn"),
        model=cfg.get("model"),
        time_stride=cfg.get("time_stride", 1),
    )

if __name__ == "__main__":
    # Compute results
    results = []
    results_avg = []
    for d in datasets6:
        acc, times, avg = process_dataset(d)
        results.append((d["name"], times, acc))
        results_avg.append((d["name"], avg))

    # Plot
    plt.figure(figsize=(8, 5))
    for i, (name, times, acc) in enumerate(results):
        plt.plot(times, acc, lw=2, label=name,
                 color=cb_palette[i % len(cb_palette)])

    plt.xlabel("Time")
    plt.ylabel("Local Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path_combined}/6_la_genadd_ti.pdf")
    plt.close()
    print(results_avg)

#---------------------------
# 7) Generalized Additive G(t|x), TD Main (no interactions)
#---------------------------

# --- Dataset configs ---
datasets7 = [
    {"name": "GT h(t|x)",
     "exp_file": f"{path_exp}/hazard_attributions_genadd_tdmain.csv",
     "data_file": f"{path_data}/7_simdata_genadd_tdmain.csv",
     "survival_fn": hazard_wrap_genadd_tdmain,
     "model": None},
    {"name": "GT log(h(t|x))",
     "exp_file": f"{path_exp}/log_hazard_attributions_genadd_tdmain.csv",
     "data_file": f"{path_data}/7_simdata_genadd_tdmain.csv",
     "survival_fn": log_hazard_wrap_genadd_tdmain,
     "model": None},
    {"name": "GT S(t|x)",
     "exp_file": f"{path_exp}/survival_attributions_genadd_tdmain.csv",
     "data_file": f"{path_data}/7_simdata_genadd_tdmain.csv",
     "survival_fn": surv_from_hazard_genadd_tdmain_wrap,
     "model": None},
    {"name": "CoxPH S(t|x)",
     "exp_file": f"{path_exp}/cox_attributions_genadd_tdmain.csv",
     "data_file": f"{path_data}/7_simdata_genadd_tdmain.csv",
     "survival_fn": None,
     "model": "Cox",
     "time_stride": 5},
    {"name": "GBSA S(t|x)",
     "exp_file": f"{path_exp}/gbsa_attributions_genadd_tdmain.csv",
     "data_file": f"{path_data}/7_simdata_genadd_tdmain.csv",
     "survival_fn": None,
     "model": "GBSA",
     "time_stride": 5},
]

def process_dataset(cfg):
    explanations_all = pd.read_csv(cfg["exp_file"])
    data_df = pd.read_csv(cfg["data_file"])
    return survshapiq_func.compute_local_accuracy(
        explanations_all,
        data_df,
        survival_fn=cfg.get("survival_fn"),
        model=cfg.get("model"),
        time_stride=cfg.get("time_stride", 1),
    )

if __name__ == "__main__":
    # Compute results
    results = []
    results_avg = []
    for d in datasets7:
        acc, times, avg = process_dataset(d)
        results.append((d["name"], times, acc))
        results_avg.append((d["name"], avg))

    # Plot
    plt.figure(figsize=(8, 5))
    for i, (name, times, acc) in enumerate(results):
        plt.plot(times, acc, lw=2, label=name,
                 color=cb_palette[i % len(cb_palette)])

    plt.xlabel("Time")
    plt.ylabel("Local Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path_combined}/7_la_genadd_tdmain.pdf")
    plt.close()
    print(results_avg)


#---------------------------
# 8) Generalized Additive G(t|x), TI (interactions)
#---------------------------

# --- Dataset configs ---
datasets8 = [
    {"name": "GT h(t|x)",
     "exp_file": f"{path_exp}/hazard_attributions_genadd_ti_inter.csv",
     "data_file": f"{path_data}/8_simdata_genadd_ti_inter.csv",
     "survival_fn": hazard_wrap_genadd_ti_inter,
     "model": None},
    {"name": "GT log(h(t|x))",
     "exp_file": f"{path_exp}/log_hazard_attributions_genadd_ti_inter.csv",
     "data_file": f"{path_data}/8_simdata_genadd_ti_inter.csv",
     "survival_fn": log_hazard_wrap_genadd_ti_inter,
     "model": None},
    {"name": "GT S(t|x)",
     "exp_file": f"{path_exp}/survival_attributions_genadd_ti_inter.csv",
     "data_file": f"{path_data}/8_simdata_genadd_ti_inter.csv",
     "survival_fn": surv_from_hazard_genadd_ti_inter_wrap,
     "model": None},
    {"name": "CoxPH S(t|x)",
     "exp_file": f"{path_exp}/cox_attributions_genadd_ti_inter.csv",
     "data_file": f"{path_data}/8_simdata_genadd_ti_inter.csv",
     "survival_fn": None,
     "model": "Cox",
     "time_stride": 5},
    {"name": "GBSA S(t|x)",
     "exp_file": f"{path_exp}/gbsa_attributions_genadd_ti_inter.csv",
     "data_file": f"{path_data}/8_simdata_genadd_ti_inter.csv",
     "survival_fn": None,
     "model": "GBSA",
     "time_stride": 5},
]

def process_dataset(cfg):
    explanations_all = pd.read_csv(cfg["exp_file"])
    data_df = pd.read_csv(cfg["data_file"])
    return survshapiq_func.compute_local_accuracy(
        explanations_all,
        data_df,
        survival_fn=cfg.get("survival_fn"),
        model=cfg.get("model"),
        time_stride=cfg.get("time_stride", 1),
    )

if __name__ == "__main__":
    # Compute results
    results = []
    results_avg = []
    for d in datasets8:
        acc, times, avg = process_dataset(d)
        results.append((d["name"], times, acc))
        results_avg.append((d["name"], avg))

    # Plot
    plt.figure(figsize=(8, 5))
    for i, (name, times, acc) in enumerate(results):
        plt.plot(times, acc, lw=2, label=name,
                 color=cb_palette[i % len(cb_palette)])

    plt.xlabel("Time")
    plt.ylabel("Local Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path_combined}/8_la_genadd_ti_inter.pdf")
    plt.close()
    print(results_avg)
    
    
#---------------------------
# 9) Generalized Additive G(t|x), TD MAIN (interactions)
#---------------------------

# --- Dataset configs ---
datasets9 = [
    {"name": "GT h(t|x)",
     "exp_file": f"{path_exp}/hazard_attributions_genadd_tdmain_inter.csv",
     "data_file": f"{path_data}/9_simdata_genadd_tdmain_inter.csv",
     "survival_fn": hazard_wrap_genadd_tdmain_inter,
     "model": None},
    {"name": "GT log(h(t|x))",
     "exp_file": f"{path_exp}/log_hazard_attributions_genadd_tdmain_inter.csv",
     "data_file": f"{path_data}/9_simdata_genadd_tdmain_inter.csv",
     "survival_fn": log_hazard_wrap_genadd_tdmain_inter,
     "model": None},
    {"name": "GT S(t|x)",
     "exp_file": f"{path_exp}/survival_attributions_genadd_tdmain_inter.csv",
     "data_file": f"{path_data}/9_simdata_genadd_tdmain_inter.csv",
     "survival_fn": surv_from_hazard_genadd_tdmain_inter_wrap,
     "model": None},
    {"name": "CoxPH S(t|x)",
     "exp_file": f"{path_exp}/cox_attributions_genadd_tdmain_inter.csv",
     "data_file": f"{path_data}/9_simdata_genadd_tdmain_inter.csv",
     "survival_fn": None,
     "model": "Cox",
     "time_stride": 5},
    {"name": "GBSA S(t|x)",
     "exp_file": f"{path_exp}/gbsa_attributions_genadd_tdmain_inter.csv",
     "data_file": f"{path_data}/9_simdata_genadd_tdmain_inter.csv",
     "survival_fn": None,
     "model": "GBSA",
     "time_stride": 5},
]

def process_dataset(cfg):
    explanations_all = pd.read_csv(cfg["exp_file"])
    data_df = pd.read_csv(cfg["data_file"])
    return survshapiq_func.compute_local_accuracy(
        explanations_all,
        data_df,
        survival_fn=cfg.get("survival_fn"),
        model=cfg.get("model"),
        time_stride=cfg.get("time_stride", 1),
    )

if __name__ == "__main__":
    # Compute results
    results = []
    results_avg = []
    for d in datasets9:
        acc, times, avg = process_dataset(d)
        results.append((d["name"], times, acc))
        results_avg.append((d["name"], avg))

    # Plot
    plt.figure(figsize=(8, 5))
    for i, (name, times, acc) in enumerate(results):
        plt.plot(times, acc, lw=2, label=name,
                 color=cb_palette[i % len(cb_palette)])

    plt.xlabel("Time")
    plt.ylabel("Local Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path_combined}/9_la_genadd_tdmain_inter.pdf")
    plt.close()
    print(results_avg)
    
    
#---------------------------
# 10) Generalized Additive G(t|x), TD INTER (interactions)
#---------------------------

# --- Dataset configs ---
datasets10 = [
    {"name": "GT h(t|x)",
     "exp_file": f"{path_exp}/hazard_attributions_genadd_tdinter.csv",
     "data_file": f"{path_data}/10_simdata_genadd_tdinter.csv",
     "survival_fn": hazard_wrap_genadd_tdinter,
     "model": None},
    {"name": "GT log(h(t|x))",
     "exp_file": f"{path_exp}/log_hazard_attributions_genadd_tdinter.csv",
     "data_file": f"{path_data}/10_simdata_genadd_tdinter.csv",
     "survival_fn": log_hazard_wrap_genadd_tdinter,
     "model": None},
    {"name": "GT S(t|x)",
     "exp_file": f"{path_exp}/survival_attributions_genadd_tdinter.csv",
     "data_file": f"{path_data}/10_simdata_genadd_tdinter.csv",
     "survival_fn": surv_from_hazard_genadd_tdinter_wrap,
     "model": None},
    {"name": "CoxPH S(t|x)",
     "exp_file": f"{path_exp}/cox_attributions_genadd_tdinter.csv",
     "data_file": f"{path_data}/10_simdata_genadd_tdinter.csv",
     "survival_fn": None,
     "model": "Cox",
     "time_stride": 5},
    {"name": "GBSA S(t|x)",
     "exp_file": f"{path_exp}/gbsa_attributions_genadd_tdinter.csv",
     "data_file": f"{path_data}/10_simdata_genadd_tdinter.csv",
     "survival_fn": None,
     "model": "GBSA",
     "time_stride": 5},
]

def process_dataset(cfg):
    explanations_all = pd.read_csv(cfg["exp_file"])
    data_df = pd.read_csv(cfg["data_file"])
    return survshapiq_func.compute_local_accuracy(
        explanations_all,
        data_df,
        survival_fn=cfg.get("survival_fn"),
        model=cfg.get("model"),
        time_stride=cfg.get("time_stride", 1),
    )

if __name__ == "__main__":
    # Compute results
    results = []
    results_avg = []
    for d in datasets10:
        acc, times, avg = process_dataset(d)
        results.append((d["name"], times, acc))
        results_avg.append((d["name"], avg))

    # Plot
    plt.figure(figsize=(8, 5))
    for i, (name, times, acc) in enumerate(results):
        plt.plot(times, acc, lw=2, label=name,
                 color=cb_palette[i % len(cb_palette)])

    plt.xlabel("Time")
    plt.ylabel("Local Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path_combined}/10_la_genadd_tdinter.pdf")
    plt.close()
    print(results_avg)


########### COMBINED PLOTS

# Collect all datasets lists
all_datasets = [datasets1, datasets2, datasets3, datasets4, datasets5,
                datasets6, datasets7, datasets8, datasets9, datasets10]

# Optional: Define custom titles for each subplot
subplot_titles = [
    "(1) Linear G(t|x) TI",
    "(2) Linear G(t|x) TD Main",
    "(3) Linear G(t|x) TI Inter",
    "(4) Linear G(t|x) TD Main Inter",
    "(5) Linear G(t|x) TD Inter",
    "(6) General Additive G(t|x) TI",
    "(7) General Additive G(t|x) TD Main",
    "(8) General Additive G(t|x) TI Inter",
    "(9) General Additive G(t|x) TD Main Inter",
    "(10) General Additive G(t|x) TD Inter",
]

def compute_results(datasets):
    results = []
    for d in datasets:
        acc, times, _ = process_dataset(d)
        results.append((d["name"], times, acc))
    return results

if __name__ == "__main__":
    all_results = [compute_results(ds) for ds in all_datasets]

    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(14, 20), sharex=True, sharey=True)
    axes = axes.flatten()

    global_handles, global_labels = [], []

    for ax, results, title in zip(axes, all_results, subplot_titles):
        for i, (name, times, acc) in enumerate(results):
            ax.plot(times, acc, lw=2.5, label=name,
                    color=cb_palette[i % len(cb_palette)])

        # --- Per-subplot labels ---
        ax.set_title(title, fontsize=16, loc="left")
        ax.set_xlabel("Time", fontsize=14)
        ax.set_ylabel("Local Accuracy", fontsize=14)

        # Collect handles & labels for global legend
        h, l = ax.get_legend_handles_labels()
        global_handles.extend(h)
        global_labels.extend(l)

    # Deduplicate legend entries
    by_label = OrderedDict(zip(global_labels, global_handles))

    # Remove subplot legends
    for ax in axes:
        ax.legend().remove()

    # Global legend (with black border and bigger font)
    leg = fig.legend(by_label.values(), by_label.keys(),
                     loc="lower center", ncol=3, fontsize=14, frameon=True)
    leg.get_frame().set_edgecolor("grey")
    leg.get_frame().set_linewidth(1)

    plt.tight_layout(rect=[0, 0.04, 1, 1])  # leave space for legend
    plt.savefig(f"{path_result}/la_all_datasets_grid.pdf")
    plt.close()
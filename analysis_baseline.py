import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
BASELINE_DIR = os.path.join(RESULTS_DIR, "baseline_experiments")
BASELINE_ANALYSIS_DIR = "analysis_baseline"
os.makedirs(BASELINE_ANALYSIS_DIR, exist_ok=True)

def load_baseline_results(baseline_dir, prefix):
    all_data = []
    for fname in os.listdir(baseline_dir):
        if fname.startswith(prefix) and fname.endswith(".jsonl"):
            with open(os.path.join(baseline_dir, fname)) as f:
                all_data.extend(json.loads(line) for line in f)
    if not all_data:
        raise RuntimeError(f"No baseline results found with prefix '{prefix}' in {baseline_dir}")
    return pd.DataFrame(all_data)

def ensure_baseline_types(df):
    df['raw_convergent'] = pd.to_numeric(df['raw_convergent'], errors='coerce')
    df['raw_divergent'] = pd.to_numeric(df['raw_divergent'], errors='coerce')
    return df

def load_evolution_log():
    all_logs = []
    for subdir in os.listdir(RESULTS_DIR):
        log_path = os.path.join(RESULTS_DIR, subdir, "map_elites_evolution_log.jsonl")
        if os.path.exists(log_path):
            if subdir.startswith("map_elites_macgyver_"):
                problem_id = subdir.replace("map_elites_macgyver_", "")
            else:
                problem_id = None
            with open(log_path) as f:
                log_entries = [json.loads(line) for line in f]
            for entry in log_entries:
                entry['experiment'] = subdir
                if problem_id is not None:
                    entry['problem_id'] = problem_id
            all_logs.extend(log_entries)
    if not all_logs:
        raise RuntimeError("No MAP-Elites evolution logs found!")
    return pd.DataFrame(all_logs)


def plot_per_category_figures(baseline_no_prompt_df, baseline_random_prompt_df, evo_log_df):
    # Ensure numeric types
    baseline_no_prompt_df = ensure_baseline_types(baseline_no_prompt_df)
    baseline_random_prompt_df = ensure_baseline_types(baseline_random_prompt_df)

    # Map problem_id to category using baseline data
    exp_to_cat = baseline_no_prompt_df.groupby('problem_id')['category'].first().to_dict()
    evo_log_df['category'] = evo_log_df['problem_id'].astype(str).map(
        {k.split('_')[-1]: v for k, v in exp_to_cat.items()}
    )
    categories = baseline_no_prompt_df['category'].unique()
    color_map = {
        'Outdoors': '#4C3575',
        'Neutral': '#205295',
        'Indoors/Household': '#3876BF'
    }

    for cat in categories:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False)
        cat_color = color_map.get(cat, None)

        # Filter data
        cat_no = baseline_no_prompt_df[baseline_no_prompt_df['category'] == cat]
        cat_rand = baseline_random_prompt_df[baseline_random_prompt_df['category'] == cat]
        cat_evo = evo_log_df[evo_log_df['category'] == cat]
        generations = sorted(cat_evo['generation'].unique())

        # --- Convergent subplot ---
        ax = axes[0]
        # Baseline: scatter all runs (X: run index, Y: score)
        ax.scatter(cat_no['run'], cat_no['raw_convergent'], color='red', alpha=0.7, label='Baseline No Prompt')
        ax.scatter(cat_rand['run'], cat_rand['raw_convergent'], color='blue', alpha=0.7, label='Baseline Random Prompt')
        # MAP-Elites: plot trend over generations
        if not cat_evo.empty:
            grouped = cat_evo.groupby('generation')['avg_convergent'].mean().reindex(generations)
            ax.plot(generations, grouped, color=cat_color, marker='o', linewidth=2.5, label='MAP-Elites')
        ax.set_title(f"Convergent Score ({cat})")
        ax.set_xlabel("Generation (MAP-Elites) / Run Index (Baseline)")
        ax.set_ylabel("Convergent Score")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        # --- Divergent subplot ---
        ax = axes[1]
        ax.scatter(cat_no['run'], cat_no['raw_divergent'], color='red', alpha=0.7, label='Baseline No Prompt')
        ax.scatter(cat_rand['run'], cat_rand['raw_divergent'], color='blue', alpha=0.7, label='Baseline Random Prompt')
        if not cat_evo.empty:
            grouped = cat_evo.groupby('generation')['avg_divergent_creativity'].mean().reindex(generations)
            ax.plot(generations, grouped, color=cat_color, marker='o', linewidth=2.5, label='MAP-Elites')
        ax.set_title(f"Divergent Score ({cat})")
        ax.set_xlabel("Generation (MAP-Elites) / Run Index (Baseline)")
        ax.set_ylabel("Divergent Score")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.suptitle(f"MAP-Elites vs Baselines: {cat}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(BASELINE_ANALYSIS_DIR, f"category_{cat.replace('/', '_')}_comparison.png"))
        plt.close()

def plot_comparison_with_baseline_mean(baseline_no_prompt_df, baseline_random_prompt_df, evo_log_df):
    baseline_no_prompt_df = ensure_baseline_types(baseline_no_prompt_df)
    baseline_random_prompt_df = ensure_baseline_types(baseline_random_prompt_df)
    exp_to_cat = baseline_no_prompt_df.groupby('problem_id')['category'].first().to_dict()
    evo_log_df['category'] = evo_log_df['problem_id'].astype(str).map(
        {k.split('_')[-1]: v for k, v in exp_to_cat.items()}
    )
    categories = baseline_no_prompt_df['category'].unique()
    color_map = {
        'Outdoors': '#4C3575',
        'Neutral': '#205295',
        'Indoors/Household': '#3876BF'
    }

    for cat in categories:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False)
        cat_color = color_map.get(cat, None)
        cat_no = baseline_no_prompt_df[baseline_no_prompt_df['category'] == cat]
        cat_rand = baseline_random_prompt_df[baseline_random_prompt_df['category'] == cat]
        cat_evo = evo_log_df[evo_log_df['category'] == cat]
        generations = sorted(cat_evo['generation'].unique())

        # --- Convergent subplot ---
        ax = axes[0]
        # Baseline means as horizontal lines
        ax.axhline(cat_no['raw_convergent'].mean(), color='red', linestyle='-', label='Baseline No Prompt (mean)')
        ax.axhline(cat_rand['raw_convergent'].mean(), color='blue', linestyle='--', label='Baseline Random Prompt (mean)')
        # MAP-Elites
        if not cat_evo.empty:
            grouped = cat_evo.groupby('generation')['avg_convergent'].mean().reindex(generations)
            ax.plot(generations, grouped, color=cat_color, marker='o', linewidth=2.5, label='MAP-Elites')
        ax.set_title(f"Convergent Score ({cat})")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Convergent Score")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        # --- Divergent subplot ---
        ax = axes[1]
        ax.axhline(cat_no['raw_divergent'].mean(), color='red', linestyle='-', label='Baseline No Prompt (mean)')
        ax.axhline(cat_rand['raw_divergent'].mean(), color='blue', linestyle='--', label='Baseline Random Prompt (mean)')
        if not cat_evo.empty:
            grouped = cat_evo.groupby('generation')['avg_divergent_creativity'].mean().reindex(generations)
            ax.plot(generations, grouped, color=cat_color, marker='o', linewidth=2.5, label='MAP-Elites')
        ax.set_title(f"Divergent Score ({cat})")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Divergent Score")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.suptitle(f"MAP-Elites vs Baselines: {cat}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(BASELINE_ANALYSIS_DIR, f"category_{cat.replace('/', '_')}_comparison_mean.png"))
        plt.close()

def print_all_data():
    print("=== Baseline No Prompt Data ===")
    baseline_no_prompt_df = load_baseline_results(BASELINE_DIR, prefix="baseline_no_prompt_results")
    print(baseline_no_prompt_df)
    print("\n=== Baseline Random Prompt Data ===")
    baseline_random_prompt_df = load_baseline_results(BASELINE_DIR, prefix="baseline_random_prompt_results")
    print(baseline_random_prompt_df)
    print("\n=== MAP-Elites Evolution Log Data ===")
    evo_log_df = load_evolution_log()
    print(evo_log_df)

if __name__ == "__main__":
    baseline_no_prompt_df = load_baseline_results(BASELINE_DIR, prefix="baseline_no_prompt_results")
    baseline_random_prompt_df = load_baseline_results(BASELINE_DIR, prefix="baseline_random_prompt_results")
    evo_log_df = load_evolution_log()
    plot_comparison_with_baseline_mean(baseline_no_prompt_df, baseline_random_prompt_df, evo_log_df)
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist # type: ignore
from sklearn.cluster import KMeans # type: ignore 
import warnings
import matplotlib.cm as cm

RESULTS_DIR = "results"
ANALYSIS_DIR = "analysis_map"
os.makedirs(ANALYSIS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="viridis")

def sanitize_filename(s):
    return str(s).replace("/", "_").replace("\\", "_").replace(" ", "_")

def load_mapelites_data():
    all_dfs = []
    for subdir in os.listdir(RESULTS_DIR):
        eval_path = os.path.join(RESULTS_DIR, subdir, "map_elites_evaluated_prompts.jsonl")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                evals = [json.loads(line) for line in f]
            df = pd.DataFrame(evals)
            df['experiment'] = subdir
            all_dfs.append(df)
    if not all_dfs:
        raise RuntimeError("No MAP-Elites experiment results found!")
    df = pd.concat(all_dfs, ignore_index=True)
    df[['raw_divergent', 'raw_convergent']] = pd.DataFrame(df['scores'].tolist(), index=df.index)
    df[['bd_dim1', 'bd_dim2']] = pd.DataFrame(df['bd_float_coords'].tolist(), index=df.index)
    df = df[df['solution_text'].notnull() & (df['solution_text'].str.strip() != "")]
    return df

def load_evolution_log():
    all_logs = []
    for subdir in os.listdir(RESULTS_DIR):
        log_path = os.path.join(RESULTS_DIR, subdir, "map_elites_evolution_log.jsonl")
        if os.path.exists(log_path):
            with open(log_path) as f:
                log_entries = [json.loads(line) for line in f]
            for entry in log_entries:
                entry['experiment'] = subdir
            all_logs.extend(log_entries)
    if not all_logs:
        raise RuntimeError("No evolution logs found!")
    df = pd.DataFrame(all_logs)
    return df

def plot_average_performance_by_category_from_log(log_df, df):
    # Map experiment to category using the main df
    exp_to_cat = df.groupby('experiment')['category'].first().to_dict()
    log_df['category'] = log_df['experiment'].map(exp_to_cat)

    categories = log_df['category'].dropna().unique()
    generations = sorted(log_df['generation'].unique())
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True)
    for cat in categories:
        cat_df = log_df[log_df['category'] == cat]
        grouped = cat_df.groupby('generation').agg({'avg_convergent': 'mean', 'avg_divergent_creativity': 'mean'}).reindex(generations)
        axes[0].plot(generations, grouped['avg_convergent'], marker='o', label=cat)
        axes[1].plot(generations, grouped['avg_divergent_creativity'], marker='s', label=cat)
    axes[0].set_title("Average Convergent Score by Problem Category") 
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Average Convergent Score")
    axes[0].legend(title="Problem Category")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[1].set_title("Average Divergent Score by Problem Category")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Average Divergent Score")
    axes[1].legend(title="Problem Category")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    plt.suptitle("Average Algorithm Performance Over Generations", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(ANALYSIS_DIR, "average_performance_by_category_from_log.png"))
    plt.close()

# def plot_archive_size_from_log():
#     all_logs = []
#     for subdir in os.listdir(RESULTS_DIR):
#         log_path = os.path.join(RESULTS_DIR, subdir, "map_elites_hypervolume_log.jsonl")
#         if os.path.exists(log_path):
#             with open(log_path) as f:
#                 log_entries = [json.loads(line) for line in f]
#             for entry in log_entries:
#                 entry['experiment'] = subdir  # Assign experiment folder name
#             all_logs.extend(log_entries)
#     if not all_logs:
#         print("No hypervolume logs found!")
#         return
#     df = pd.DataFrame(all_logs)
#     # Extract category using the same logic as load_mapelites_data
#     # Example: if your folder is "map_elites_macgyver_482", category is "macgyver"
#     if 'category' not in df.columns:
#         df['category'] = df['experiment'].apply(
#             lambda x: x.split('_')[2] if len(x.split('_')) > 2 else 'All'
#         )
#     categories = df['category'].unique()
#     generations = sorted(df['generation'].unique())
#     plt.figure(figsize=(10, 7))
#     for cat in categories:
#         cat_df = df[df['category'] == cat]
#         grouped = cat_df.groupby('generation')['archive_size'].mean().reindex(generations)
#         plt.plot(generations, grouped, marker='s', label=cat)
#     plt.title("Archive Size Over Generations")
#     plt.xlabel("Generation")
#     plt.ylabel("Archive Size")
#     plt.legend(title="Category")
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     plt.savefig(os.path.join(ANALYSIS_DIR, "archive_size_over_generations.png"))
#     plt.close()

def report_problem_solution_evolution(df, every_n=5, mode='best'):
    md_path = os.path.join(ANALYSIS_DIR, f"problem_solution_evolution_{mode}.md")
    prompt_col = [c for c in df.columns if 'prompt' in c][0] if any('prompt' in c for c in df.columns) else None
    with open(md_path, "w") as f:
        f.write(f"# Problem Solution Evolution ({'Best' if mode=='best' else 'Lowest'} Scores) by Generation\n\n")
        for pid, group in df.groupby('problem_id'):
            f.write(f"## Problem ID: {pid}\n")
            f.write("**Problem:** [problem text not available]\n\n")
            generations = sorted(group['generation'].unique())
            for gen in generations[::every_n]:
                gen_group = group[group['generation'] == gen]
                if not gen_group.empty:
                    if mode == 'best':
                        best_conv = gen_group.loc[gen_group['raw_convergent'].idxmax()]
                        best_div = gen_group.loc[gen_group['raw_divergent'].idxmax()]
                    else:
                        best_conv = gen_group.loc[gen_group['raw_convergent'].idxmin()]
                        best_div = gen_group.loc[gen_group['raw_divergent'].idxmin()]
                    # Best Convergent
                    f.write(f"### Generation {gen}\n")
                    f.write(f"- **{'Best' if mode=='best' else 'Lowest'} Convergent Score:** {best_conv['raw_convergent']:.2f}\n")
                    f.write(f"- **Divergent Score (for this solution):** {best_conv['raw_divergent']:.2f}\n")
                    if prompt_col and pd.notnull(best_conv[prompt_col]):
                        f.write(f"- **Prompt (Convergent):**\n```\n{best_conv[prompt_col]}\n```\n")
                    else:
                        f.write("- **Prompt (Convergent):** [prompt not available]\n")
                    f.write(f"- **Solution (Convergent):**\n```\n{best_conv['solution_text']}\n```\n")
                    # Best Divergent
                    f.write(f"- **{'Best' if mode=='best' else 'Lowest'} Divergent Score:** {best_div['raw_divergent']:.2f}\n")
                    f.write(f"- **Convergent Score (for this solution):** {best_div['raw_convergent']:.2f}\n")
                    if prompt_col and pd.notnull(best_div[prompt_col]):
                        f.write(f"- **Prompt (Divergent):**\n```\n{best_div[prompt_col]}\n```\n")
                    else:
                        f.write("- **Prompt (Divergent):** [prompt not available]\n")
                    f.write(f"- **Solution (Divergent):**\n```\n{best_div['solution_text']}\n```\n\n")
            f.write("\n---\n")
    print(f"Problem/solution evolution report saved to {md_path}")

def report_problem_solution_evolution_worst(df, every_n=5):
    md_path = os.path.join(ANALYSIS_DIR, "problem_solution_evolution_worst.md")
    possible_problem_cols = [c for c in df.columns if 'problem' in c and 'text' in c]
    problem_col = possible_problem_cols[0] if possible_problem_cols else None
    prompt_col = [c for c in df.columns if 'prompt' in c][0] if any('prompt' in c for c in df.columns) else None
    with open(md_path, "w") as f:
        f.write("# Problem Solution Evolution (Lowest Scores) by Generation\n\n")
        for pid, group in df.groupby('problem_id'):
            f.write(f"## Problem ID: {pid}\n")
            if problem_col:
                f.write(f"**Problem:** {group[problem_col].iloc[0]}\n\n")
            else:
                f.write("**Problem:** [problem text not available]\n\n")
            for gen in sorted(group['generation'].unique())[::every_n]:
                gen_group = group[group['generation'] == gen]
                if not gen_group.empty:
                    # Worst convergent
                    worst_conv = gen_group.loc[gen_group['raw_convergent'].idxmin()]
                    # Worst divergent
                    worst_div = gen_group.loc[gen_group['raw_divergent'].idxmin()]
                    f.write(f"### Generation {gen}\n")
                    # Convergent
                    f.write(f"- **Lowest Convergent Score:** {worst_conv['raw_convergent']:.2f}\n")
                    if prompt_col and pd.notnull(worst_conv[prompt_col]):
                        f.write(f"- **Prompt (Convergent):**\n```\n{worst_conv[prompt_col]}\n```\n")
                    else:
                        f.write("- **Prompt (Convergent):** [prompt not available]\n")
                    f.write(f"- **Solution (Convergent):**\n```\n{worst_conv['solution_text']}\n```\n")
                    # Divergent
                    f.write(f"- **Lowest Divergent Score:** {worst_div['raw_divergent']:.2f}\n")
                    if prompt_col and pd.notnull(worst_div[prompt_col]):
                        f.write(f"- **Prompt (Divergent):**\n```\n{worst_div[prompt_col]}\n```\n")
                    else:
                        f.write("- **Prompt (Divergent):** [prompt not available]\n")
                    f.write(f"- **Solution (Divergent):**\n```\n{worst_div['solution_text']}\n```\n\n")
            f.write("\n---\n")
    print(f"Worst-case problem/solution evolution report saved to {md_path}")

def plot_gene_score_by_category_multi(df, gene="role_instruction"):
    df['gene_value'] = df['genotype'].apply(lambda g: g.get(gene, ""))
    # Only keep gene_values present in all categories
    categories = set(df['category'].unique())
    counts = df.groupby('gene_value')['category'].nunique()
    valid_gene_values = counts[counts == len(categories)].index
    plot_df = df[df['gene_value'].isin(valid_gene_values)]
    # Compute mean across all categories for ranking
    mean_scores = plot_df.groupby('gene_value')['raw_convergent'].mean()
    ranked_gene_values = mean_scores.sort_values(ascending=False).index
    plot_df['gene_value'] = pd.Categorical(plot_df['gene_value'], categories=ranked_gene_values, ordered=True)
    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=plot_df,
        y="gene_value",
        x="raw_convergent",
        hue="category",
        estimator=np.mean,
        ci=None,
        palette="Blues"
    )
    plt.title(f"Mean Convergent Score by {gene.replace('_',' ').title()} (Grouped by Problem Category)")
    plt.xlabel("Mean Convergent Score")
    plt.ylabel(gene.replace('_',' ').title())
    plt.legend(title="Problem Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, f"barplot_convergent_by_{gene}_all_categories_filtered_ranked.png"))
    plt.close()

def plot_prompt_style_heatmap(df, gene, score_type="raw_convergent"):
    df['gene_value'] = df['genotype'].apply(lambda g: g.get(gene, ""))
    pivot = df.pivot_table(index='gene_value', columns='category', values=score_type, aggfunc='mean')
    pivot = pivot.dropna()  # Only keep prompt styles present in all categories
    # Rank by mean score
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    plt.figure(figsize=(12, max(6, 0.4*len(pivot))))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", cbar_kws={'label': f'Mean {score_type.replace("_", " ").title()}'})
    plt.title(f"Prompt Style vs. Category Mean {score_type.replace('_', ' ').title()} ({gene.replace('_',' ').title()})")
    plt.xlabel("Problem Category")
    plt.ylabel(gene.replace('_',' ').title())
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, f"heatmap_{score_type}_by_{gene}_all_categories.png"))
    plt.close()

def plot_behavioral_diversity(df, window=1):
    categories = df['category'].unique()
    generations = sorted(df['generation'].unique())
    plt.figure(figsize=(12,7))
    for cat in categories:
        safe_cat = sanitize_filename(cat)
        cat_df = df[df['category'] == cat]
        diversity = []
        for gen in generations:
            coords = cat_df[cat_df['generation'] == gen][['bd_dim1', 'bd_dim2']].values
            if len(coords) > 1:
                diversity.append(np.mean(pdist(coords)))
            else:
                diversity.append(0)
        diversity = np.array(diversity)
        plt.plot(generations, diversity, label=f"{cat}")
    plt.title("Behavioral Diversity of MAP-Elites Over Generations by Problem Type")
    plt.xlabel("Generation")
    plt.ylabel("Avg Pairwise Distance (Diversity in UMAP Space)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "behavioral_diversity_by_category.png"))
    plt.close()

def plot_behavioral_diversity_with_shading(df, window=3):
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.spatial.distance import pdist

    categories = df['category'].unique()
    generations = sorted(df['generation'].unique())
    plt.figure(figsize=(12, 7))

    for cat in categories:
        cat_df = df[df['category'] == cat]
        diversity_per_gen = []
        for gen in generations:
            gen_df = cat_df[cat_df['generation'] == gen]
            coords = gen_df[['bd_dim1', 'bd_dim2']].values
            if len(coords) > 1:
                diversity = pdist(coords).mean()
            else:
                diversity = 0
            diversity_per_gen.append(diversity)
        # Smoothing
        diversity_smooth = pd.Series(diversity_per_gen).rolling(window, min_periods=1, center=True).mean()
        # Variability (std)
        diversity_std = pd.Series(diversity_per_gen).rolling(window, min_periods=1, center=True).std().fillna(0)
        plt.plot(generations, diversity_smooth, label=f"{cat}")
        plt.fill_between(generations, diversity_smooth-diversity_std, diversity_smooth+diversity_std, alpha=0.2)
        # Annotate max
        max_idx = diversity_smooth.idxmax()
        plt.annotate(f"Max: {diversity_smooth[max_idx]:.1f}", (generations[max_idx], diversity_smooth[max_idx]), fontsize=9)

    plt.xlabel("Generation")
    plt.ylabel("Avg Pairwise Distance (Diversity in UMAP Space)")
    plt.title("Behavioral Diversity of MAP-Elites Over Generations (Smoothed, Shaded Variability)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "behavioral_diversity_over_generations_shaded.png"))
    plt.close()

def qualitative_report(df):
    report_path = os.path.join(ANALYSIS_DIR, "qualitative_report.txt")
    possible_problem_cols = [c for c in df.columns if 'problem' in c and 'text' in c]
    problem_col = possible_problem_cols[0] if possible_problem_cols else None
    with open(report_path, "w") as f:
        f.write("=== Best and Lowest Scoring Solutions by Problem ===\n\n")
        for pid, group in df.groupby('problem_id'):
            f.write(f"Problem ID: {pid}\n")
            f.write(f"Category: {group['category'].iloc[0]}\n")
            if problem_col:
                f.write(f"Problem: {group[problem_col].iloc[0]}\n\n")
            else:
                f.write("Problem: [problem text not available]\n\n")
            best_row = group.loc[group['raw_convergent'].idxmax()]
            f.write(f"  [Best Convergent Score: {best_row['raw_convergent']:.2f} | Divergent: {best_row['raw_divergent']:.2f}]\n")
            f.write(f"  Solution: {best_row['solution_text']}\n\n")
            worst_row = group.loc[group['raw_convergent'].idxmin()]
            f.write(f"  [Lowest Convergent Score: {worst_row['raw_convergent']:.2f} | Divergent: {worst_row['raw_divergent']:.2f}]\n")
            f.write(f"  Solution: {worst_row['solution_text']}\n\n")
    print(f"Qualitative report saved to {report_path}")

def plot_grid_coverage(df, bins=8):
    categories = df['category'].unique()
    for cat in categories:
        safe_cat = sanitize_filename(cat)
        cat_df = df[df['category'] == cat]
        H, xedges, yedges = np.histogram2d(cat_df['bd_dim1'], cat_df['bd_dim2'], bins=bins)
        plt.figure(figsize=(8,7))
        sns.heatmap(H.T, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': 'Num Elites'}, linewidths=0.5)
        plt.title(f"MAP-Elites Grid Coverage ({cat})\nGrid coverage: {100*np.count_nonzero(H)/H.size:.1f}% of cells filled")
        plt.xlabel("Grid X (Behavior Descriptor 1 bin)")
        plt.ylabel("Grid Y (Behavior Descriptor 2 bin)")
        plt.tight_layout()
        plt.figtext(0.5, -0.08,
            "Each cell represents a region in the 2D behavior descriptor space (e.g., UMAP dimensions).\n"
            "Grid X and Grid Y are the indices of the discretized behavior space. The number in each cell shows how many elite solutions were found in that region.\n"
            "A higher coverage percentage indicates more diverse exploration by MAP-Elites.",
            wrap=True, horizontalalignment='center', fontsize=10)
        plt.savefig(os.path.join(ANALYSIS_DIR, f"grid_coverage_{safe_cat}.png"), bbox_inches='tight')
        plt.close()

def plot_umap_landscape_by_category_with_clusters(df, n_clusters=5, annotate_examples=True):
    import matplotlib.cm as cm
    categories = df['category'].unique()
    for score_type, cmap, label in [
        ('raw_convergent', 'plasma', 'Convergent Creativity Score'),
        ('raw_divergent', 'viridis', 'Divergent Creativity Score')
    ]:
        fig, axes = plt.subplots(1, len(categories), figsize=(8*len(categories), 7), sharey=True)
        if len(categories) == 1:
            axes = [axes]
        for i, cat in enumerate(categories):
            cat_df = df[df['category'] == cat]
            coords = cat_df[['bd_dim1', 'bd_dim2']].values
            if len(coords) >= n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
                cat_df = cat_df.copy()
                cat_df['cluster'] = kmeans.labels_
            else:
                cat_df['cluster'] = 0
            # Color by cluster, shade by score
            cluster_colors = cm.tab10(cat_df['cluster'] / n_clusters)
            sc = axes[i].scatter(cat_df['bd_dim1'], cat_df['bd_dim2'], c=cat_df[score_type], cmap=cmap, s=80, alpha=0.8, edgecolor='k')
            # Mark cluster centers
            for cluster_id in np.unique(cat_df['cluster']):
                cluster_points = cat_df[cat_df['cluster'] == cluster_id]
                center = cluster_points[['bd_dim1', 'bd_dim2']].mean()
                axes[i].scatter(center['bd_dim1'], center['bd_dim2'], marker='X', s=200, label=f"Cluster {cluster_id}", edgecolor='black')
            # Annotate a few example prompts
            if annotate_examples:
                for idx, row in cat_df.sample(min(3, len(cat_df))).iterrows():
                    axes[i].annotate(row.get('prompt_text', '')[:30] + "...", (row['bd_dim1'], row['bd_dim2']),
                                     fontsize=8, color='blue', alpha=0.7, xytext=(5,5), textcoords='offset points')
            axes[i].set_title(f"{label} Landscape: {cat}", fontsize=14)
            axes[i].set_xlabel("UMAP Dimension 1")
            if i == 0:
                axes[i].set_ylabel("UMAP Dimension 2")
            axes[i].legend()
            cbar = plt.colorbar(sc, ax=axes[i], label=label)
        plt.suptitle(f"UMAP Landscape by Cluster and {label}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, f"umap_landscape_{score_type}_by_category_with_clusters_labeled.png"))
        plt.close()

def plot_umap_prompt_diversity(df, n_clusters=5, examples_per_cluster=2):
    import matplotlib.cm as cm
    categories = df['category'].unique()
    for cat in categories:
        cat_df = df[df['category'] == cat]
        coords = cat_df[['bd_dim1', 'bd_dim2']].values
        if len(coords) < n_clusters:
            continue
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
        cat_df = cat_df.copy()
        cat_df['cluster'] = kmeans.labels_
        cluster_colors = cm.tab10(cat_df['cluster'] / n_clusters)
        plt.figure(figsize=(10, 8))
        # Plot points colored by cluster
        plt.scatter(cat_df['bd_dim1'], cat_df['bd_dim2'], c=cluster_colors, s=60, alpha=0.7, edgecolor='k', label='Prompt')
        # Mark cluster centers
        centers = kmeans.cluster_centers_
        for cluster_id, center in enumerate(centers):
            plt.scatter(center[0], center[1], marker='X', s=250, color=cm.tab10(cluster_id / n_clusters), edgecolor='black', label=f"Cluster {cluster_id}")
            # Annotate a few example prompts per cluster
            cluster_points = cat_df[cat_df['cluster'] == cluster_id]
            for _, row in cluster_points.sample(min(examples_per_cluster, len(cluster_points)), random_state=42).iterrows():
                plt.annotate(row.get('prompt_text', '')[:40] + "...", (row['bd_dim1'], row['bd_dim2']),
                             fontsize=8, color=cm.tab10(cluster_id / n_clusters), alpha=0.8, xytext=(5,5), textcoords='offset points')
        plt.title(f"UMAP Prompt Diversity: {cat}\nClusters: {n_clusters} | Total Prompts: {len(cat_df)}")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, f"umap_prompt_diversity_{sanitize_filename(cat)}.png"))
        plt.close()

def save_umap_diversity_metrics_table(df, n_clusters=5):
    """
    Save a CSV table with: category, cluster_id, cluster_size, avg_pairwise_distance, total_clusters
    """
    import csv
    from scipy.spatial.distance import pdist

    metrics = []
    categories = df['category'].unique()
    for cat in categories:
        cat_df = df[df['category'] == cat]
        coords = cat_df[['bd_dim1', 'bd_dim2']].values
        if len(coords) < n_clusters:
            continue
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
        cat_df = cat_df.copy()
        cat_df['cluster'] = kmeans.labels_
        for cluster_id in range(n_clusters):
            cluster_points = cat_df[cat_df['cluster'] == cluster_id]
            mean_conv = cluster_points['raw_convergent'].mean()
            mean_div = cluster_points['raw_divergent'].mean()
            print(f"Category: {cat}, Cluster {cluster_id}: Mean Convergent={mean_conv:.2f}, Mean Divergent={mean_div:.2f}, Size={len(cluster_points)}")
            cluster_points = cat_df[cat_df['cluster'] == cluster_id][['bd_dim1', 'bd_dim2']].values
            cluster_size = len(cluster_points)
            if cluster_size > 1:
                avg_dist = pdist(cluster_points).mean()
            else:
                avg_dist = 0.0
            metrics.append({
                "category": cat,
                "cluster_id": cluster_id,
                "cluster_size": cluster_size,
                "avg_pairwise_distance": avg_dist,
                "total_clusters": n_clusters
            })
    # Save as CSV
    out_path = os.path.join(ANALYSIS_DIR, "umap_diversity_metrics.csv")
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(out_path, index=False)
    print(f"UMAP diversity metrics table saved to {out_path}")

def main():
    df = load_mapelites_data()
    log_df = load_evolution_log()
    plot_average_performance_by_category_from_log(log_df, df)
    # plot_archive_size_from_log()
    report_problem_solution_evolution(df, every_n=5)
    report_problem_solution_evolution_worst(df, every_n=5)
    for gene in ["role_instruction", "creativity_instruction", "combination_instruction", "format_instruction"]:
        for score_type in ["raw_convergent", "raw_divergent"]:
            plot_prompt_style_heatmap(df, gene, score_type)
    # plot_behavioral_diversity(df, window=3)
    plot_behavioral_diversity_with_shading(df, window=3)
    qualitative_report(df)
    plot_grid_coverage(df, bins=8)
    plot_umap_landscape_by_category_with_clusters(df)
    plot_umap_prompt_diversity(df)
    save_umap_diversity_metrics_table(df)
    print("All MAP-Elites analysis plots and qualitative report saved in", ANALYSIS_DIR)

if __name__ == "__main__":
    main()

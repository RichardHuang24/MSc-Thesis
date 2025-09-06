import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist  # type: ignore
import warnings
import glob
from sklearn.cluster import KMeans # type: ignore

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
    # Remove empty prompts or solutions if any
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



def plot_average_performance_by_category_from_log(df):
    """
    Plots average convergent and divergent scores over generations, grouped by problem category,
    using the evolution log.
    """
    categories = df['category'].unique() if 'category' in df.columns else ['All']
    generations = sorted(df['generation'].unique())
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True)
    for cat in categories:
        cat_df = df[df['category'] == cat] if 'category' in df.columns else df
        # Group by generation and take the mean (in case of multiple runs/experiments)
        grouped = cat_df.groupby('generation').agg({'avg_convergent_creativity': 'mean', 'avg_divergent_creativity': 'mean'}).reindex(generations)
        axes[0].plot(generations, grouped['avg_convergent_creativity'], marker='o', label=cat)
        axes[1].plot(generations, grouped['avg_convergent_creativity'], marker='s', label=cat)
    axes[0].set_title("Average Convergent Score (Quality) by Problem Type")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Average Convergent Score")
    axes[0].legend(title="Problem Type")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[1].set_title("Average Divergent Score (Novelty) by Problem Type")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Average Divergent Score")
    axes[1].legend(title="Problem Type")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    plt.suptitle("Average Algorithm Performance Over Generations by Problem Type", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(ANALYSIS_DIR, "average_performance_by_category_from_log.png"))
    plt.close()

def plot_umap_landscape_by_category(df):
    """
    Plots UMAP landscapes colored by convergent and divergent scores, faceted by problem category.
    Warns and skips if UMAP is degenerate (all points on a line or collapsed).
    """
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
            # Check for degenerate UMAP
            if np.isclose(cat_df['bd_dim1'].std(), 0) or np.isclose(cat_df['bd_dim2'].std(), 0):
                warnings.warn(f"UMAP for category '{cat}' is degenerate (collapsed to a line or point). Skipping plot.")
                axes[i].set_visible(False)
                continue
            sc = axes[i].scatter(cat_df['bd_dim1'], cat_df['bd_dim2'], c=cat_df[score_type], cmap=cmap, s=80, alpha=0.85, edgecolor='k')
            axes[i].set_title(f"{label} Score Landscape: {cat}", fontsize=14)
            axes[i].set_xlabel("UMAP Dimension 1")
            if i == 0:
                axes[i].set_ylabel("UMAP Dimension 2")
            plt.colorbar(sc, ax=axes[i], label=label)
            axes[i].grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, f"umap_landscape_{score_type}_by_category.png"))
        plt.close()

def plot_umap_landscape_by_category_with_clusters(df, n_clusters=5, annotate_examples=True):
    """
    Improved UMAP plot: clusters, color by score, annotate example solutions.
    """
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
                centers = kmeans.cluster_centers_
            else:
                cat_df['cluster'] = 0
                centers = np.array([[cat_df['bd_dim1'].mean(), cat_df['bd_dim2'].mean()]])
            sc = axes[i].scatter(cat_df['bd_dim1'], cat_df['bd_dim2'], c=cat_df[score_type], cmap=cmap, s=100, alpha=0.7, edgecolor='k')
            # Annotate clusters with less overlap
            for cluster_id, center in enumerate(centers):
                axes[i].text(center[0]+0.5, center[1]+0.5, f"Cluster {cluster_id}", fontsize=13, weight='bold', color='red',
                             ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))
            # Optionally annotate example solutions
            if annotate_examples:
                for cluster_id in np.unique(cat_df['cluster']):
                    cluster_points = cat_df[cat_df['cluster'] == cluster_id]
                    if not cluster_points.empty:
                        example = cluster_points.sample(1).iloc[0]
                        axes[i].annotate(
                            example.get('solution_text', '')[:40] + "...",
                            (example['bd_dim1'], example['bd_dim2']),
                            fontsize=9, color='blue', alpha=0.7, xytext=(5,5), textcoords='offset points'
                        )
            axes[i].set_title(f"{label} Score Landscape: {cat}", fontsize=14)
            axes[i].set_xlabel("UMAP Dimension 1")
            if i == 0:
                axes[i].set_ylabel("UMAP Dimension 2")
            plt.colorbar(sc, ax=axes[i], label=label)
            axes[i].grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, f"umap_landscape_{score_type}_by_category_with_clusters.png"))
        plt.close()

def plot_gene_score_by_category(df, gene="creativity_instruction", top_n=8):
    """
    Plots mean convergent and divergent scores for each gene value, grouped by problem category.
    """
    df['gene_value'] = df['genotype'].apply(lambda g: g.get(gene, ""))
    top_values = df['gene_value'].value_counts().nlargest(top_n).index
    plot_df = df[df['gene_value'].isin(top_values)]
    categories = plot_df['category'].unique()
    for cat in categories:
        safe_cat = sanitize_filename(cat)
        cat_df = plot_df[plot_df['category'] == cat]
        if cat_df.empty:
            continue
        mean_scores = cat_df.groupby('gene_value')[['raw_convergent', 'raw_divergent']].mean().reset_index()
        mean_scores = mean_scores.sort_values('raw_convergent', ascending=False)
        plt.figure(figsize=(12,6))
        sns.barplot(data=mean_scores, x='raw_convergent', y='gene_value', palette='Blues_d')
        plt.title(f"Mean Convergent Score by {gene.replace('_',' ').title()} ({cat})")
        plt.xlabel("Mean Convergent Score")
        plt.ylabel(gene.replace('_',' ').title())
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, f"barplot_convergent_by_{gene}_{safe_cat}.png"))
        plt.close()
        mean_scores = mean_scores.sort_values('raw_divergent', ascending=False)
        plt.figure(figsize=(12,6))
        sns.barplot(data=mean_scores, x='raw_divergent', y='gene_value', palette='Oranges_d')
        plt.title(f"Mean Divergent Score by {gene.replace('_',' ').title()} ({cat})")
        plt.xlabel("Mean Divergent Score")
        plt.ylabel(gene.replace('_',' ').title())
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, f"barplot_divergent_by_{gene}_{safe_cat}.png"))
        plt.close()

def qualitative_report(df):
    """
    Extracts and saves:
    - Best and lowest scoring solutions (with problem and solution text)
    - For every 2 generations, lists the problem and solution for the top solution, showing evolution
    """
    report_path = os.path.join(ANALYSIS_DIR, "qualitative_report.txt")
    # Find the correct problem text column
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
            # Best
            best_row = group.loc[group['raw_convergent'].idxmax()]
            f.write(f"  [Best Convergent Score: {best_row['raw_convergent']:.2f} | Divergent: {best_row['raw_divergent']:.2f}]\n")
            f.write(f"  Solution: {best_row['solution_text']}\n\n")
            # Lowest
            worst_row = group.loc[group['raw_convergent'].idxmin()]
            f.write(f"  [Lowest Convergent Score: {worst_row['raw_convergent']:.2f} | Divergent: {worst_row['raw_divergent']:.2f}]\n")
            f.write(f"  Solution: {worst_row['solution_text']}\n\n")
        f.write("\n=== Solution Evolution (Every 2 Generations) ===\n\n")
        for pid, group in df.groupby('problem_id'):
            f.write(f"Problem ID: {pid}\n")
            f.write(f"Category: {group['category'].iloc[0]}\n")
            if problem_col:
                f.write(f"Problem: {group[problem_col].iloc[0]}\n")
            else:
                f.write("Problem: [problem text not available]\n")
            for gen in sorted(group['generation'].unique())[::2]:
                gen_group = group[group['generation'] == gen]
                if not gen_group.empty:
                    top_row = gen_group.loc[gen_group['raw_convergent'].idxmax()]
                    f.write(f"  Generation {gen}: [Convergent: {top_row['raw_convergent']:.2f} | Divergent: {top_row['raw_divergent']:.2f}]\n")
                    f.write(f"    Solution: {top_row['solution_text']}\n")
            f.write("\n")
    print(f"Qualitative report saved to {report_path}")

def plot_grid_coverage(df, bins=8):
    """
    Plots MAP-Elites grid coverage (heatmap) for each problem type.
    """
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

def plot_behavioral_diversity(df, window=3):
    """
    Plots behavioral diversity (average pairwise UMAP distance) over generations, grouped by problem type.
    """
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
        smoothed = np.convolve(diversity, np.ones(window)/window, mode='same')
        plt.plot(generations, diversity, alpha=0.3, label=f"{cat} (raw)")
        plt.plot(generations, smoothed, label=f"{cat} (smoothed, window={window})")
    plt.title("Behavioral Diversity of MAP-Elites Over Generations by Problem Type")
    plt.xlabel("Generation")
    plt.ylabel("Avg Pairwise Distance (Diversity in UMAP Space)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "behavioral_diversity_by_category.png"))
    plt.close()

def plot_best_scores_table(df):
    best = df.groupby(['category', 'problem_id']).agg({
        'raw_convergent': 'max',
        'raw_divergent': 'max'
    }).reset_index()
    best = best.sort_values(by=['category', 'raw_convergent'], ascending=[True, False])
    best.to_csv(os.path.join(ANALYSIS_DIR, "best_scores_per_problem_by_category.csv"), index=False)
    print("Best scores table saved to best_scores_per_problem_by_category.csv")
    with open(os.path.join(ANALYSIS_DIR, "best_scores_per_problem_by_category.md"), "w") as f:
        f.write("| Category | Problem ID | Best Convergent | Best Divergent |\n")
        f.write("| --- | --- | --- | --- |\n")
        for _, row in best.iterrows():
            f.write(f"| {row['category']} | {row['problem_id']} | {row['raw_convergent']:.3f} | {row['raw_divergent']:.3f} |\n")

def qualitative_report_markdown(df):
    """
    Save a markdown report with highest and lowest divergent solutions, and interpretation.
    """
    md_path = os.path.join(ANALYSIS_DIR, "qualitative_report.md")
    # Find the correct problem text column
    possible_problem_cols = [c for c in df.columns if 'problem' in c and 'text' in c]
    problem_col = possible_problem_cols[0] if possible_problem_cols else None

    # Highest and lowest divergent
    best_row = df.loc[df['raw_divergent'].idxmax()]
    worst_row = df.loc[df['raw_divergent'].idxmin()]

    with open(md_path, "w") as f:
        f.write("# Part 5 & 6: Qualitative Analysis of Elite Prompts\n\n")
        f.write("## Comparison: Highest vs. Lowest Divergent Creativity\n\n")
        f.write("### Highest Divergence\n\n")
        f.write(f"**Scores:** Divergent={best_row['raw_divergent']:.4f}, Convergent={best_row['raw_convergent']:.4f}\n")
        f.write(f"**Problem ID:** `{best_row['problem_id']}`\n\n")
        if problem_col:
            f.write("**Prompt:**\n```\n" + str(best_row[problem_col]) + "\n```\n\n")
        else:
            f.write("**Prompt:**\n[problem text not available]\n\n")
        f.write("**Solution:**\n```\n" + str(best_row['solution_text']) + "\n```\n\n")
        f.write("**Prompt Genotype (Genes):**\n```json\n" + json.dumps(best_row['genotype'], indent=2) + "\n```\n\n---\n\n")
        f.write("### Lowest Divergence\n\n")
        f.write(f"**Scores:** Divergent={worst_row['raw_divergent']:.4f}, Convergent={worst_row['raw_convergent']:.4f}\n")
        f.write(f"**Problem ID:** `{worst_row['problem_id']}`\n\n")
        if problem_col:
            f.write("**Prompt:**\n```\n" + str(worst_row[problem_col]) + "\n```\n\n")
        else:
            f.write("**Prompt:**\n[problem text not available]\n\n")
        f.write("**Solution:**\n```\n" + str(worst_row['solution_text']) + "\n```\n\n")
        f.write("**Prompt Genotype (Genes):**\n```json\n" + json.dumps(worst_row['genotype'], indent=2) + "\n```\n\n---\n\n")
        f.write("## Interpretation\n")
        f.write("Compare the above examples to observe how prompt structure and content influence the creativity of the generated solution. High-divergence prompts typically encourage more novel, unexpected, or unconventional solutions, while low-divergence prompts may result in more standard or predictable responses.\n")
    print(f"Qualitative markdown report saved to {md_path}")

def plot_gene_score_by_category_multi(df, genes=None, top_n=8):
    """
    Plots mean convergent and divergent scores for each gene value, grouped by gene and colored by problem category.
    """
    if genes is None:
        genes = ["role_instruction", "creativity_instruction", "constraint_instruction", "format_instruction"]
    for gene in genes:
        df['gene_value'] = df['genotype'].apply(lambda g: g.get(gene, ""))
        top_values = df['gene_value'].value_counts().nlargest(top_n).index
        plot_df = df[df['gene_value'].isin(top_values)]
        if plot_df.empty:
            continue
        # Convergent
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
        plt.savefig(os.path.join(ANALYSIS_DIR, f"barplot_convergent_by_{gene}_all_categories.png"))
        plt.close()
        # Divergent
        plt.figure(figsize=(14, 7))
        sns.barplot(
            data=plot_df,
            y="gene_value",
            x="raw_divergent",
            hue="category",
            estimator=np.mean,
            ci=None,
            palette="Oranges"
        )
        plt.title(f"Mean Divergent Score by {gene.replace('_',' ').title()} (Grouped by Problem Category)")
        plt.xlabel("Mean Divergent Score")
        plt.ylabel(gene.replace('_',' ').title())
        plt.legend(title="Problem Category", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, f"barplot_divergent_by_{gene}_all_categories.png"))
        plt.close()

def check_umap_variance(df):
    """
    Print variance of UMAP coordinates for debugging.
    """
    print("UMAP bd_dim1 variance:", np.var(df['bd_dim1']))
    print("UMAP bd_dim2 variance:", np.var(df['bd_dim2']))
    print("bd_dim1 min/max:", df['bd_dim1'].min(), df['bd_dim1'].max())
    print("bd_dim2 min/max:", df['bd_dim2'].min(), df['bd_dim2'].max())
    print("Sample bd_float_coords:", df[['bd_dim1', 'bd_dim2']].head(10).values)

def report_problem_solution_evolution(df):
    """
    For each problem, for each generation, report the best solution (by convergent score).
    """
    md_path = os.path.join(ANALYSIS_DIR, "problem_solution_evolution.md")
    # Find the correct problem text column
    possible_problem_cols = [c for c in df.columns if 'problem' in c and 'text' in c]
    problem_col = possible_problem_cols[0] if possible_problem_cols else None

    with open(md_path, "w") as f:
        f.write("# Problem Solution Evolution by Generation\n\n")
        for pid, group in df.groupby('problem_id'):
            f.write(f"## Problem ID: {pid}\n")
            if problem_col:
                f.write(f"**Problem:** {group[problem_col].iloc[0]}\n\n")
            else:
                f.write("**Problem:** [problem text not available]\n\n")
            for gen in sorted(group['generation'].unique()):
                gen_group = group[group['generation'] == gen]
                if not gen_group.empty:
                    top_row = gen_group.loc[gen_group['raw_convergent'].idxmax()]
                    f.write(f"### Generation {gen}\n")
                    f.write(f"- **Convergent Score:** {top_row['raw_convergent']:.2f}\n")
                    f.write(f"- **Divergent Score:** {top_row['raw_divergent']:.2f}\n")
                    f.write(f"- **Solution:**\n```\n{top_row['solution_text']}\n```\n\n")
            f.write("\n---\n")
    print(f"Problem/solution evolution report saved to {md_path}")

def plot_all_gene_values_by_category(df, genes=None, top_n=10):
    """
    Plots mean convergent and divergent scores for each gene value, grouped by gene and colored by problem category.
    """
    if genes is None:
        # Use all keys in genotype
        sample_genotype = df['genotype'].iloc[0]
        genes = list(sample_genotype.keys())
    for gene in genes:
        df['gene_value'] = df['genotype'].apply(lambda g: g.get(gene, ""))
        top_values = df['gene_value'].value_counts().nlargest(top_n).index
        plot_df = df[df['gene_value'].isin(top_values)]
        if plot_df.empty:
            continue
        # Convergent
        plt.figure(figsize=(16, 8))
        sns.barplot(
            data=plot_df,
            y="gene_value",
            x="raw_convergent",
            hue="category",
            estimator=np.mean,
            ci=None,
            palette="Blues"
        )
        plt.title(f"Mean Convergent Score by {gene.replace('_',' ').title()} (All Categories)")
        plt.xlabel("Mean Convergent Score")
        plt.ylabel(gene.replace('_',' ').title())
        plt.legend(title="Problem Category", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, f"barplot_convergent_by_{gene}_all_categories.png"))
        plt.close()
        # Divergent
        plt.figure(figsize=(16, 8))
        sns.barplot(
            data=plot_df,
            y="gene_value",
            x="raw_divergent",
            hue="category",
            estimator=np.mean,
            ci=None,
            palette="Oranges"
        )
        plt.title(f"Mean Divergent Score by {gene.replace('_',' ').title()} (All Categories)")
        plt.xlabel("Mean Divergent Score")
        plt.ylabel(gene.replace('_',' ').title())
        plt.legend(title="Problem Category", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, f"barplot_divergent_by_{gene}_all_categories.png"))
        plt.close()

def plot_pareto_front_by_category():
    """
    Plots the Pareto front of best solutions for each problem category.
    """
    csv_path = os.path.join(ANALYSIS_DIR, "best_scores_per_problem_by_category.csv")
    if not os.path.exists(csv_path):
        print("best_scores_per_problem_by_category.csv not found. Run plot_best_scores_table() first.")
        return
    df = pd.read_csv(csv_path)
    categories = df['category'].unique()
    plt.figure(figsize=(10, 7))
    for cat in categories:
        sub = df[df['category'] == cat]
        plt.scatter(sub['raw_convergent'], sub['raw_divergent'], label=cat, alpha=0.7)
        # Pareto front: sort by convergent, keep points with highest divergent so far
        pareto = sub.sort_values('raw_convergent', ascending=False)
        pf = []
        max_div = -float('inf')
        for _, row in pareto.iterrows():
            if row['raw_divergent'] > max_div:
                pf.append(row)
                max_div = row['raw_divergent']
        pf = pd.DataFrame(pf)
        plt.plot(pf['raw_convergent'], pf['raw_divergent'], marker='o', linestyle='-', label=f"{cat} Pareto Front")
    plt.xlabel('Best Convergent Score')
    plt.ylabel('Best Divergent Score')
    plt.title('Pareto Fronts of Best Solutions by Problem Category')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "pareto_front_by_category.png"))
    plt.close()

def plot_umap_by_prompt_style(df, problem_id, style_gene="format_instruction"):
    """
    Plots UMAP landscape for a single problem, colored by prompt style (gene).
    """
    sub = df[df['problem_id'] == problem_id].copy()
    if sub.empty:
        print(f"No data for problem_id={problem_id}")
        return
    sub['style'] = sub['genotype'].apply(lambda g: g.get(style_gene, "unknown"))
    plt.figure(figsize=(10,8))
    for style in sub['style'].unique():
        style_df = sub[sub['style'] == style]
        plt.scatter(style_df['bd_dim1'], style_df['bd_dim2'], label=style, s=80, alpha=0.8)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.title(f"Prompt Style Diversity in UMAP Space ({style_gene})\nProblem ID: {problem_id}")
    plt.legend(title="Prompt Style")
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, f"umap_prompt_style_{style_gene}_{problem_id}.png"))
    plt.close()

def plot_umap_creativity_scores(df, problem_id):
    """
    Plots UMAP landscape for a single problem, colored by convergent and divergent scores.
    """
    sub = df[df['problem_id'] == problem_id].copy()
    if sub.empty:
        print(f"No data for problem_id={problem_id}")
        return
    for score_type, cmap, label in [
        ('raw_convergent', 'plasma', 'Convergent Creativity Score'),
        ('raw_divergent', 'viridis', 'Divergent Creativity Score')
    ]:
        plt.figure(figsize=(10,8))
        sc = plt.scatter(sub['bd_dim1'], sub['bd_dim2'], c=sub[score_type], cmap=cmap, s=80, alpha=0.85, edgecolor='k')
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.title(f"{label} in UMAP Space\nProblem ID: {problem_id}")
        plt.colorbar(sc, label=label)
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, f"umap_{score_type}_{problem_id}.png"))
        plt.close()

def plot_pareto_front_by_prompt_style(df, problem_id, style_gene="format_instruction"):
    """
    Plots Pareto front of prompt styles for a single problem.
    """
    sub = df[df['problem_id'] == problem_id].copy()
    if sub.empty:
        print(f"No data for problem_id={problem_id}")
        return
    sub['style'] = sub['genotype'].apply(lambda g: g.get(style_gene, "unknown"))
    plt.figure(figsize=(10,8))
    for style in sub['style'].unique():
        style_df = sub[sub['style'] == style]
        plt.scatter(style_df['raw_convergent'], style_df['raw_divergent'], label=style, alpha=0.7)
        # Pareto front for this style
        pareto = style_df.sort_values('raw_convergent', ascending=False)
        pf = []
        max_div = -float('inf')
        for _, row in pareto.iterrows():
            if row['raw_divergent'] > max_div:
                pf.append(row)
                max_div = row['raw_divergent']
        pf = pd.DataFrame(pf)
        if not pf.empty:
            plt.plot(pf['raw_convergent'], pf['raw_divergent'], marker='o', linestyle='-', label=f"{style} Pareto")
    plt.xlabel('Convergent Score')
    plt.ylabel('Divergent Score')
    plt.title(f"Pareto Front by Prompt Style\nProblem ID: {problem_id}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, f"pareto_prompt_style_{style_gene}_{problem_id}.png"))
    plt.close()

def analyze_prompt_style_vs_creativity(df, style_gene="format_instruction"):
    """
    For each problem, plot:
    - UMAP prompt style diversity
    - UMAP colored by creativity scores
    - Pareto front by prompt style
    """
    problem_ids = df['problem_id'].unique()
    for pid in problem_ids:
        print(f"Analyzing prompt style vs. creativity for problem_id: {pid}")
        plot_umap_by_prompt_style(df, pid, style_gene=style_gene)
        plot_umap_creativity_scores(df, pid)
        plot_pareto_front_by_prompt_style(df, pid, style_gene=style_gene)

def plot_hypervolume_and_archive_size_from_log(results_dir=RESULTS_DIR):
    """
    Plots hypervolume and archive size over generations using the map_elites_hypervolume_log.jsonl files.
    """
    import json

    all_logs = []
    for subdir in os.listdir(results_dir):
        log_path = os.path.join(results_dir, subdir, "map_elites_hypervolume_log.jsonl")
        if os.path.exists(log_path):
            with open(log_path) as f:
                log_entries = [json.loads(line) for line in f]
            for entry in log_entries:
                entry['experiment'] = subdir
            all_logs.extend(log_entries)

    if not all_logs:
        print("No hypervolume logs found!")
        return

    df = pd.DataFrame(all_logs)
    # Try to get category from experiment name or set as 'All'
    if 'category' not in df.columns:
        df['category'] = df['experiment'].apply(lambda x: x.split('_')[-1] if '_' in x else 'All')

    categories = df['category'].unique()
    generations = sorted(df['generation'].unique())
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True)
    for cat in categories:
        cat_df = df[df['category'] == cat]
        axes[0].plot(cat_df['generation'], cat_df['hypervolume'], marker='o', label=cat)
        axes[1].plot(cat_df['generation'], cat_df['archive_size'], marker='s', label=cat)
    axes[0].set_title("Hypervolume Over Generations")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Hypervolume")
    axes[0].legend(title="Problem Type")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[1].set_title("Archive Size Over Generations")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Archive Size")
    axes[1].legend(title="Problem Type")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    plt.suptitle("MAP-Elites Convergence: Hypervolume and Archive Size", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(ANALYSIS_DIR, "hypervolume_and_archive_size.png"))
    plt.close()

def main():
    df = load_mapelites_data()
    plot_umap_landscape_by_category_with_clusters(df)
    plot_gene_score_by_category(df, gene="creativity_instruction")
    plot_grid_coverage(df, bins=8)
    plot_behavioral_diversity(df, window=3)
    qualitative_report(df)
    plot_gene_score_by_category_multi(df)  # FIX: pass df, not RESULTS_DIR
    qualitative_report_markdown(df)        # FIX: pass df, not RESULTS_DIR
    report_problem_solution_evolution(df)  # FIX: pass df, not RESULTS_DIR
    plot_hypervolume_and_archive_size_from_log(RESULTS_DIR)
    # For average performance, use evolution log, so keep as is:
    plot_average_performance_by_category_from_log(load_evolution_log())
    print("All MAP-Elites analysis plots and qualitative report saved in", ANALYSIS_DIR)

if __name__ == "__main__":
    main()
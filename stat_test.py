import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import f_oneway, kruskal
import seaborn as sns
import glob
from itertools import combinations
from scipy.stats import ttest_ind, mannwhitneyu

# Ensure analysis.py is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from analysis import load_mapelites_data

def cluster_and_stat_test(df, n_clusters=5):
    results = []
    for cat in df['category'].unique():
        cat_df = df[df['category'] == cat]
        coords = cat_df[['bd_dim1', 'bd_dim2']].dropna().values
        if len(coords) < n_clusters:
            print(f"Skipping category {cat}: not enough points for clustering")
            continue
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
        cat_df = cat_df.iloc[:len(kmeans.labels_)].copy()
        cat_df['cluster'] = kmeans.labels_
        clusters = [cat_df[cat_df['cluster'] == i]['raw_convergent'].values for i in range(n_clusters)]
        if all(len(c) > 1 for c in clusters):
            f_stat, p_val = f_oneway(*clusters)
            print(f"[{cat}] ANOVA (Convergent): F={f_stat:.2f}, p={p_val:.4f}")
            k_stat, kp_val = kruskal(*clusters)
            print(f"[{cat}] Kruskal-Wallis (Convergent): H={k_stat:.2f}, p={kp_val:.4f}")
        clusters_div = [cat_df[cat_df['cluster'] == i]['raw_divergent'].values for i in range(n_clusters)]
        if all(len(c) > 1 for c in clusters_div):
            f_stat, p_val = f_oneway(*clusters_div)
            print(f"[{cat}] ANOVA (Divergent): F={f_stat:.2f}, p={p_val:.4f}")
            k_stat, kp_val = kruskal(*clusters_div)
            print(f"[{cat}] Kruskal-Wallis (Divergent): H={k_stat:.2f}, p={kp_val:.4f}")
        results.append(cat_df)
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return None

def pairwise_cluster_tests(df, out_md="stats_test/pairwise_cluster_stats.md"):
    os.makedirs("stats_test", exist_ok=True)
    results = []
    for cat in df['category'].unique():
        cat_df = df[df['category'] == cat]
        if cat_df.empty or 'cluster' not in cat_df:
            continue
        clusters_present = sorted(cat_df['cluster'].dropna().unique())
        for score_type in ['raw_convergent', 'raw_divergent']:
            for c1, c2 in combinations(clusters_present, 2):
                vals1 = cat_df[cat_df['cluster'] == c1][score_type].dropna()
                vals2 = cat_df[cat_df['cluster'] == c2][score_type].dropna()
                if len(vals1) < 2 or len(vals2) < 2:
                    continue
                # t-test
                t_stat, t_p = ttest_ind(vals1, vals2, equal_var=False)
                # Mann-Whitney U
                u_stat, u_p = mannwhitneyu(vals1, vals2, alternative='two-sided')
                results.append({
                    "category": cat,
                    "score_type": score_type,
                    "cluster_1": c1,
                    "cluster_2": c2,
                    "mean_1": vals1.mean(),
                    "mean_2": vals2.mean(),
                    "t_stat": t_stat,
                    "t_p": t_p,
                    "u_stat": u_stat,
                    "u_p": u_p,
                    "n_1": len(vals1),
                    "n_2": len(vals2)
                })
    # Write to markdown
    with open(out_md, "w") as f:
        f.write("# Pairwise Cluster Statistical Tests\n\n")
        f.write("For each cluster pair, two tests are reported:\n")
        f.write("- t-test (t-p): p-value for the independent t-test\n")
        f.write("- Mann-Whitney U (U-p): p-value for the Mann-Whitney U test\n\n")
        for cat in set(r["category"] for r in results):
            f.write(f"## Category: {cat}\n\n")
            for score_type in ['raw_convergent', 'raw_divergent']:
                f.write(f"### {score_type}\n\n")
                f.write("| Cluster 1 | Cluster 2 | N1 | N2 | Mean 1 | Mean 2 | t-test p | U-test p |\n")
                f.write("|-----------|-----------|----|----|--------|--------|----------|----------|\n")
                for r in results:
                    if r["category"] == cat and r["score_type"] == score_type:
                        f.write(f"| {r['cluster_1']} | {r['cluster_2']} | {r['n_1']} | {r['n_2']} | {r['mean_1']:.2f} | {r['mean_2']:.2f} | {r['t_p']:.4f} | {r['u_p']:.4f} |\n")
                f.write("\n")
    print(f"Pairwise cluster stats written to {out_md}")

def pca_prompt_styles(clustered_df, prompt_fields, n_components=2, out_dir="stats_test"):
    os.makedirs(out_dir, exist_ok=True)
    for cat in clustered_df['category'].unique():
        group = clustered_df[clustered_df['category'] == cat]
        grouped = group.groupby('cluster')
        features = []
        metas = []
        for cl, subg in grouped:
            prompt_vec = []
            for field in prompt_fields:
                codes, _ = pd.factorize(subg[field].fillna(""))
                prompt_vec.append(codes.mean())
            features.append(prompt_vec)
            metas.append({
                'cluster': cl,
                'mean_conv': subg['raw_convergent'].mean(),
                'mean_div': subg['raw_divergent'].mean()
            })
        X = np.array(features)
        if len(X) < 2:
            print(f"Not enough clusters for PCA visualization in category {cat}.")
            continue
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=[m['mean_conv'] for m in metas],
            s=80 + 120 * np.array([m['mean_div'] for m in metas]),  # marker size
            cmap='plasma'
        )
        plt.colorbar(scatter, label='Mean Convergent Score')
        for i, meta in enumerate(metas):
            plt.text(
                X_pca[i, 0], X_pca[i, 1],
                f"C{meta['cluster']}", fontsize=10, weight='bold'
            )
            plt.annotate(
                f"Conv: {meta['mean_conv']:.2f}\nDiv: {meta['mean_div']:.2f}",
                (X_pca[i, 0], X_pca[i, 1]),
                textcoords="offset points", xytext=(0,10), ha='center', fontsize=8
            )
        plt.title(f"PCA of Prompt Styles by Cluster ({cat})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"pca_prompt_styles_{cat.replace('/', '_')}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"PCA plot saved to {out_path}")

def pca_prompt_styles_biplot(clustered_df, prompt_fields, n_components=2, out_dir="stats_test"):
    os.makedirs(out_dir, exist_ok=True)
    for cat in clustered_df['category'].unique():
        group = clustered_df[clustered_df['category'] == cat]
        grouped = group.groupby('cluster')
        features = []
        metas = []
        for cl, subg in grouped:
            prompt_vec = []
            for field in prompt_fields:
                # Factorize categorical fields to numeric codes
                codes, _ = pd.factorize(subg[field].fillna(""))
                prompt_vec.append(codes.mean())
            features.append(prompt_vec)
            metas.append({
                'cluster': cl,
                'mean_conv': subg['raw_convergent'].mean(),
                'mean_div': subg['raw_divergent'].mean()
            })
        X = np.array(features)
        if len(X) < 2:
            print(f"Not enough clusters for PCA visualization in category {cat}.")
            continue
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        # Get loadings (directions of original variables)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=[m['mean_conv'] for m in metas], cmap='plasma', s=100)
        for i, meta in enumerate(metas):
            plt.text(X_pca[i, 0], X_pca[i, 1], f"C{meta['cluster']}", fontsize=10, weight='bold')
            plt.annotate(
                f"Conv: {meta['mean_conv']:.2f}\nDiv: {meta['mean_div']:.2f}",
                (X_pca[i, 0], X_pca[i, 1]),
                textcoords="offset points", xytext=(0,10), ha='center', fontsize=8
            )

        # Draw arrows for each prompt style variable
        for i, field in enumerate(prompt_fields):
            plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='black', alpha=0.7, head_width=0.05)
            plt.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, field, color='black', ha='center', va='center', fontsize=11)

        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
        plt.title(f"PCA Biplot of Prompt Styles by Cluster ({cat})")
        plt.colorbar(scatter, label='Mean Convergent Score')
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"pca_biplot_{cat.replace('/', '_')}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"PCA biplot saved to {out_path}")

def triangle_pairwise_table(df, out_dir="stats_test"):
    os.makedirs(out_dir, exist_ok=True)
    for cat in df['category'].unique():
        cat_df = df[df['category'] == cat]
        if cat_df.empty or 'cluster' not in cat_df:
            continue
        clusters_present = sorted(cat_df['cluster'].dropna().unique())
        n = len(clusters_present)
        for score_type in ['raw_convergent', 'raw_divergent']:
            # Prepare empty matrices
            p_matrix = [["" for _ in range(n)] for _ in range(n)]
            t_matrix = [["" for _ in range(n)] for _ in range(n)]
            for i, c1 in enumerate(clusters_present):
                for j, c2 in enumerate(clusters_present):
                    if i >= j:
                        continue  # Fill only upper triangle
                    vals1 = cat_df[cat_df['cluster'] == c1][score_type].dropna()
                    vals2 = cat_df[cat_df['cluster'] == c2][score_type].dropna()
                    if len(vals1) < 2 or len(vals2) < 2:
                        p_matrix[i][j] = "-"
                        t_matrix[i][j] = "-"
                        continue
                    t_stat, t_p = ttest_ind(vals1, vals2, equal_var=False)
                    p_matrix[i][j] = f"{t_p:.4f}"
                    t_matrix[i][j] = f"{t_stat:.2f}"
            # Write to markdown
            md_path = os.path.join(out_dir, f"triangle_pairwise_{cat.replace('/', '_')}_{score_type}.md")
            with open(md_path, "w") as f:
                f.write(f"# Pairwise t-test p-values for {cat} ({score_type})\n\n")
                f.write("Upper triangle: p-value (t-test)\n\n")
                # Header
                f.write("|   | " + " | ".join([f"Cluster {c}" for c in clusters_present]) + " |\n")
                f.write("|---" * (n+1) + "|\n")
                for i, c1 in enumerate(clusters_present):
                    row = [f"Cluster {c1}"]
                    for j in range(n):
                        if i == j:
                            row.append("—")
                        elif i < j:
                            row.append(p_matrix[i][j])
                        else:
                            row.append("")
                    f.write("| " + " | ".join(row) + " |\n")
                f.write("\n\n")
                f.write("Upper triangle: t-statistic (t-test)\n\n")
                f.write("|   | " + " | ".join([f"Cluster {c}" for c in clusters_present]) + " |\n")
                f.write("|---" * (n+1) + "|\n")
                for i, c1 in enumerate(clusters_present):
                    row = [f"Cluster {c1}"]
                    for j in range(n):
                        if i == j:
                            row.append("—")
                        elif i < j:
                            row.append(t_matrix[i][j])
                        else:
                            row.append("")
                    f.write("| " + " | ".join(row) + " |\n")
            print(f"Triangle table saved to {md_path}")

def extract_significant_prompt_examples(
    df, 
    triangle_dir="stats_test", 
    p_threshold=0.05, 
    n_examples=2
):
    import re
    import glob
    os.makedirs(triangle_dir, exist_ok=True)
    triangle_files = glob.glob(os.path.join(triangle_dir, "triangle_pairwise_*_raw_*.md"))
    significant_pairs = []
    for triangle_path in triangle_files:
        match = re.match(r".*triangle_pairwise_(.+)_(raw_\w+)\.md", triangle_path)
        if not match:
            continue
        cat = match.group(1).replace('_', '/')
        score_type = match.group(2)
        with open(triangle_path, "r") as f:
            lines = f.readlines()
        clusters = []
        header_found = False
        for line in lines:
            if line.startswith("|   |"):
                clusters = [int(x.strip().replace("Cluster ", "")) for x in line.strip().split("|")[2:-1]]
                header_found = True
                continue
            if header_found and line.startswith("| Cluster"):
                parts = [x.strip() for x in line.strip().split("|")[1:-1]]
                c1 = int(parts[0].replace("Cluster ", ""))
                for j, val in enumerate(parts[1:]):
                    if val == "—" or val == "" or val == "-":
                        continue
                    try:
                        p_val = float(val)
                    except ValueError:
                        continue
                    c2 = clusters[j]
                    if c1 < c2 and p_val < p_threshold:
                        significant_pairs.append((cat, score_type, c1, c2))
    # Write only significant pairs
    for cat in set([x[0] for x in significant_pairs]):
        md_path = os.path.join(triangle_dir, f"significant_prompt_examples_{cat.replace('/', '_')}.md")
        with open(md_path, "w") as f:
            f.write(f"# Significant Cluster Pairs and Prompt Examples for {cat}\n\n")
            for score_type in ['raw_convergent', 'raw_divergent']:
                pairs = [x for x in significant_pairs if x[0] == cat and x[1] == score_type]
                if not pairs:
                    continue
                f.write(f"## {score_type}\n\n")
                for _, stype, c1, c2 in pairs:
                    f.write(f"### Cluster {c1} vs Cluster {c2}\n")
                    dcat = df[(df['category'] == cat)]
                    d1 = dcat[dcat['cluster'] == c1]
                    d2 = dcat[dcat['cluster'] == c2]
                    mean_conv1 = d1['raw_convergent'].mean()
                    mean_div1 = d1['raw_divergent'].mean()
                    mean_conv2 = d2['raw_convergent'].mean()
                    mean_div2 = d2['raw_divergent'].mean()
                    f.write(f"- **Cluster {c1}**: mean convergent = {mean_conv1:.2f}, mean divergent = {mean_div1:.2f}\n")
                    for i, row in d1.head(n_examples).iterrows():
                        f.write(f"    - Prompt: `{row['prompt_text']}`\n")
                    f.write(f"- **Cluster {c2}**: mean convergent = {mean_conv2:.2f}, mean divergent = {mean_div2:.2f}\n")
                    for i, row in d2.head(n_examples).iterrows():
                        f.write(f"    - Prompt: `{row['prompt_text']}`\n")
                    f.write("\n")
        print(f"Significant prompt examples written to {md_path}")

def main():
    print("Loading MAP-Elites data using analysis.py loader...")
    df = load_mapelites_data()
    print(f"Loaded {len(df)} records.")

    # --- Extract prompt style fields from genotype dict ---
    prompt_fields = ["role_instruction", "creativity_instruction", "combination_instruction", "format_instruction"]
    for field in prompt_fields:
        df[field] = df["genotype"].apply(lambda g: g.get(field, "") if isinstance(g, dict) else "")

    print("Clustering and statistical testing...")
    clustered_df = cluster_and_stat_test(df, n_clusters=5)
    if clustered_df is not None:
        print("Running PCA visualization...")
        pca_prompt_styles(
            clustered_df,
            prompt_fields
        )
        pca_prompt_styles_biplot(
            clustered_df,
            prompt_fields
        )
        print("Running pairwise cluster tests...")
        pairwise_cluster_tests(clustered_df)
        print("Generating triangle pairwise tables...")
        triangle_pairwise_table(clustered_df)
        print("Extracting significant prompt examples...")
        extract_significant_prompt_examples(clustered_df)
    else:
        print("No clustered data to visualize.")

if __name__ == "__main__":
    main()
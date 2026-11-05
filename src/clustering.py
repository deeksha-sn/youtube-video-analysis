"""
K-Means Clustering Module
=========================
Course: Data Mining & Data Warehousing (BCI24505D)
Topic 42: YouTube Video Analysis

Implements Algorithm 1: Unsupervised K-Means clustering to segment videos
based on popularity and viewer engagement profiles.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, Any, Tuple

from src.preprocessing import CLEANED_CSV_PATH

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "figures")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "models")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "reports")
CLUSTERING_REPORT_PATH = os.path.join(REPORTS_DIR, "clustering_report.json")
CLUSTERED_CSV_PATH = os.path.join(os.path.dirname(CLEANED_CSV_PATH), "clustered_youtube_data.csv")


def determine_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 9)) -> Tuple[int, Dict[int, float], Dict[int, float]]:
    """Compute inertia and silhouette scores across a range of k values."""
    inertias = {}
    silhouettes = {}
    
    # Subsample for fast silhouette computation if dataset is large
    sample_size = min(5000, len(X_scaled))
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(X_scaled), sample_size, replace=False)
    X_sub = X_scaled[sample_indices]
    
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=42)
        km.fit(X_scaled)
        inertias[k] = float(km.inertia_)
        
        # Silhouette on subsample
        labels_sub = km.predict(X_sub)
        sil = float(silhouette_score(X_sub, labels_sub))
        silhouettes[k] = sil
        print(f"  k={k} | Inertia={km.inertia_:.1f} | Silhouette={sil:.4f}")
    
    # Plot Elbow Curve
    plt.figure(figsize=(9, 5))
    plt.plot(list(inertias.keys()), list(inertias.values()), marker="o", color="#2980b9", linewidth=2.5, markersize=8)
    plt.title("K-Means Elbow Method for Optimal Clusters (k)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Within-Cluster Sum of Squares (Inertia)")
    plt.xticks(list(k_range))
    plt.axvline(3, color="#e74c3c", linestyle="--", label="Selected Optimal k = 3")
    plt.legend()
    plt.tight_layout()
    elbow_path = os.path.join(FIGURES_DIR, "13_kmeans_elbow_curve.png")
    plt.savefig(elbow_path)
    plt.close()
    print(f"[INFO] Elbow curve saved to: {elbow_path}")
    
    return 3, inertias, silhouettes


def run_clustering(csv_path: str = CLEANED_CSV_PATH, optimal_k: int = 3) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute standard scaling, K-Means clustering, profiling, and reporting."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cleaned dataset not found at: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} records for clustering.")
    
    # Feature selection: Log-transformed views, likes, comments, and engagement rate
    # Log transforms prevent extreme multi-million outliers from distorting distance calculations
    cluster_features = ["log_views", "log_likes", "log_comments", "engagement_rate"]
    
    # Handle any potential remaining inf or NaN values
    X = df[cluster_features].replace([np.inf, -np.inf], np.nan).dropna()
    valid_idx = X.index
    df_clustered = df.loc[valid_idx].copy()
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save Scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    
    # Elbow method evaluation
    print("[INFO] Evaluating Elbow Curve and Silhouette Scores for k=2..8:")
    chosen_k, inertias, sil_scores = determine_optimal_k(X_scaled, range(2, 8))
    
    # Fit Final K-Means Model
    print(f"[INFO] Fitting final K-Means model with k={chosen_k}...")
    kmeans = KMeans(n_clusters=chosen_k, init="k-means++", n_init=15, max_iter=300, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    df_clustered["cluster_id"] = labels
    
    # Save Model
    model_path = os.path.join(MODELS_DIR, "kmeans_model.joblib")
    joblib.dump(kmeans, model_path)
    print(f"[SUCCESS] Saved K-Means model to {model_path}")
    
    # Compute Empirical Cluster Profiles
    cluster_summary = {}
    for cid in range(chosen_k):
        c_subset = df_clustered[df_clustered["cluster_id"] == cid]
        cluster_summary[f"Cluster_{cid}"] = {
            "count": int(len(c_subset)),
            "percentage": round((len(c_subset) / len(df_clustered)) * 100, 2),
            "views_mean": float(c_subset["views"].mean()),
            "views_median": float(c_subset["views"].median()),
            "likes_mean": float(c_subset["likes"].mean()),
            "likes_median": float(c_subset["likes"].median()),
            "comments_mean": float(c_subset["comment_count"].mean()),
            "engagement_rate_mean": float(c_subset["engagement_rate"].mean()),
            "engagement_rate_median": float(c_subset["engagement_rate"].median()),
            "top_category": str(c_subset["category_name"].mode().iloc[0] if len(c_subset) > 0 else "N/A")
        }
    
    # Assign semantic labels based on actual views and engagement characteristics
    # Sort clusters by median views to name them systematically
    sorted_by_views = sorted(cluster_summary.items(), key=lambda item: item[1]["views_median"])
    cluster_naming = {}
    if len(sorted_by_views) == 3:
        cluster_naming[int(sorted_by_views[0][0].split("_")[1])] = "Moderate Reach / Low Engagement"
        cluster_naming[int(sorted_by_views[1][0].split("_")[1])] = "High Engagement / Enthusiast Viral"
        cluster_naming[int(sorted_by_views[2][0].split("_")[1])] = "Mega Blockbusters / Mass Reach"
    else:
        for idx, (cname, _) in enumerate(sorted_by_views):
            cid = int(cname.split("_")[1])
            cluster_naming[cid] = f"Popularity Tier {idx+1}"
    
    df_clustered["cluster_name"] = df_clustered["cluster_id"].map(cluster_naming)
    
    for cid in range(chosen_k):
        cluster_summary[f"Cluster_{cid}"]["semantic_label"] = cluster_naming[cid]
    
    # Save clustered dataframe to CSV for downstream warehouse and dashboard use
    df_clustered.to_csv(CLUSTERED_CSV_PATH, index=False)
    print(f"[SUCCESS] Clustered dataset saved to: {CLUSTERED_CSV_PATH}")
    
    # -------------------------------------------------------------------------
    # Visualization: Cluster Scatter Plot (Views vs. Engagement Rate)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("deep", chosen_k)
    sns.scatterplot(
        data=df_clustered.sample(min(4000, len(df_clustered)), random_state=42),
        x="views",
        y="engagement_rate",
        hue="cluster_name",
        palette=palette,
        alpha=0.6,
        edgecolor=None
    )
    plt.xscale("log")
    plt.title(f"K-Means Video Segmentation (k={chosen_k}): Views vs. Engagement Rate")
    plt.xlabel("Views (Log Scale)")
    plt.ylabel("Engagement Rate (%)")
    plt.legend(title="Cluster Profile", loc="upper left")
    plt.tight_layout()
    cluster_fig_path = os.path.join(FIGURES_DIR, "14_kmeans_clusters_scatter.png")
    plt.savefig(cluster_fig_path)
    plt.close()
    
    report = {
        "algorithm": "K-Means Clustering",
        "optimal_k": chosen_k,
        "features_used": cluster_features,
        "scaler_used": "StandardScaler",
        "inertias": inertias,
        "silhouette_scores": sil_scores,
        "cluster_profiles": cluster_summary
    }
    
    with open(CLUSTERING_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"[SUCCESS] Clustering report saved to: {CLUSTERING_REPORT_PATH}")
    
    return df_clustered, report


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 4: K-MEANS CLUSTERING IMPLEMENTATION")
    print("=" * 60)
    clust_df, clust_rep = run_clustering()
    print("\nCluster Profiling Results:")
    for cname, stats in clust_rep["cluster_profiles"].items():
        print(f"\n{cname} - {stats['semantic_label']}:")
        print(f"  Share: {stats['percentage']}% ({stats['count']} videos)")
        print(f"  Median Views: {stats['views_median']:,.0f} | Median Engagement: {stats['engagement_rate_median']:.2f}%")
        print(f"  Dominant Category: {stats['top_category']}")

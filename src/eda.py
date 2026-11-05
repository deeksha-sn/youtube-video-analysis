"""
Exploratory Data Analysis (EDA) Module
======================================
Course: Data Mining & Data Warehousing (BCI24505D)
Topic 42: YouTube Video Analysis

Generates 12 publication-quality visualizations and stores them in outputs/figures/,
along with a quantitative summary of observations in outputs/reports/eda_findings.json.
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any

from src.preprocessing import CLEANED_CSV_PATH

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "figures")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "reports")
EDA_REPORT_PATH = os.path.join(REPORTS_DIR, "eda_findings.json")


def setup_plot_style():
    """Configure modern, readable aesthetic for charts."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["figure.dpi"] = 150


def run_eda(csv_path: str = CLEANED_CSV_PATH) -> Dict[str, Any]:
    """Execute all 12 EDA visualizations and extract empirical insights."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    setup_plot_style()
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cleaned dataset not found at: {csv_path}. Run preprocessing first.")
    
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded cleaned data for EDA: {len(df)} records.")
    
    findings = {}
    
    # -------------------------------------------------------------------------
    # 1. Top 10 Videos Based on Views
    # -------------------------------------------------------------------------
    top10_videos = df.nlargest(10, "views")[["title", "channel_title", "views", "likes", "engagement_rate"]]
    # Shorten long titles for clean plotting
    top10_videos["short_title"] = top10_videos["title"].apply(lambda t: (t[:45] + "...") if len(str(t)) > 45 else str(t))
    
    plt.figure(figsize=(10, 6))
    bar = sns.barplot(
        data=top10_videos,
        y="short_title",
        x="views",
        palette="viridis",
        hue="short_title",
        legend=False
    )
    plt.title("Top 10 Most Viewed YouTube Trending Videos")
    plt.xlabel("Total Views (in Millions)")
    plt.ylabel("Video Title")
    # Format x-axis in millions
    bar.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M"))
    for p in bar.patches:
        width = p.get_width()
        bar.annotate(f"{width/1e6:.1f}M",
                     (width, p.get_y() + p.get_height() / 2.),
                     ha='left', va='center', xytext=(5, 0), textcoords='offset points', fontsize=9)
    plt.tight_layout()
    p1 = os.path.join(FIGURES_DIR, "01_top_10_videos_views.png")
    plt.savefig(p1)
    plt.close()
    
    findings["top_videos"] = {
        "highest_view_video": str(top10_videos.iloc[0]["title"]),
        "highest_views": int(top10_videos.iloc[0]["views"]),
        "highest_view_channel": str(top10_videos.iloc[0]["channel_title"])
    }
    
    # -------------------------------------------------------------------------
    # 2. Top 10 Channels Based on Total Views
    # -------------------------------------------------------------------------
    top_channels = (
        df.groupby("channel_title")["views"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "total_views", "count": "video_count"})
        .nlargest(10, "total_views")
        .reset_index()
    )
    
    plt.figure(figsize=(10, 6))
    bar2 = sns.barplot(
        data=top_channels,
        y="channel_title",
        x="total_views",
        palette="rocket",
        hue="channel_title",
        legend=False
    )
    plt.title("Top 10 YouTube Channels by Cumulative Trending Views")
    plt.xlabel("Total Views (in Millions)")
    plt.ylabel("Channel Name")
    bar2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M"))
    for p in bar2.patches:
        width = p.get_width()
        bar2.annotate(f"{width/1e6:.1f}M",
                      (width, p.get_y() + p.get_height() / 2.),
                      ha='left', va='center', xytext=(5, 0), textcoords='offset points', fontsize=9)
    plt.tight_layout()
    p2 = os.path.join(FIGURES_DIR, "02_top_10_channels_views.png")
    plt.savefig(p2)
    plt.close()
    
    findings["top_channels"] = {
        "leading_channel": str(top_channels.iloc[0]["channel_title"]),
        "leading_channel_views": int(top_channels.iloc[0]["total_views"]),
        "leading_channel_videos": int(top_channels.iloc[0]["video_count"])
    }
    
    # -------------------------------------------------------------------------
    # 3. Views Distribution (Original & Log1p)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["views"] / 1e6, bins=40, kde=True, ax=axes[0], color="#2b5c8f")
    axes[0].set_title("Views Distribution (Original Scale - Heavily Right-Skewed)")
    axes[0].set_xlabel("Views (Millions)")
    axes[0].set_ylabel("Count")
    
    sns.histplot(df["log_views"], bins=40, kde=True, ax=axes[1], color="#27ae60")
    axes[1].set_title("Views Distribution (Log-Transformed: log1p(Views))")
    axes[1].set_xlabel("log(1 + Views)")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    p3 = os.path.join(FIGURES_DIR, "03_views_distribution.png")
    plt.savefig(p3)
    plt.close()
    
    # -------------------------------------------------------------------------
    # 4. Likes Distribution (Original & Log1p)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["likes"] / 1e3, bins=40, kde=True, ax=axes[0], color="#e74c3c")
    axes[0].set_title("Likes Distribution (Thousands)")
    axes[0].set_xlabel("Likes (Thousands)")
    axes[0].set_ylabel("Count")
    
    sns.histplot(df["log_likes"], bins=40, kde=True, ax=axes[1], color="#9b59b6")
    axes[1].set_title("Likes Distribution (Log-Transformed)")
    axes[1].set_xlabel("log(1 + Likes)")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    p4 = os.path.join(FIGURES_DIR, "04_likes_distribution.png")
    plt.savefig(p4)
    plt.close()
    
    # -------------------------------------------------------------------------
    # 5. Comments Distribution (Original & Log1p)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["comment_count"] / 1e3, bins=40, kde=True, ax=axes[0], color="#f39c12")
    axes[0].set_title("Comments Distribution (Thousands)")
    axes[0].set_xlabel("Comment Count (Thousands)")
    axes[0].set_ylabel("Count")
    
    sns.histplot(df["log_comments"], bins=40, kde=True, ax=axes[1], color="#16a085")
    axes[1].set_title("Comments Distribution (Log-Transformed)")
    axes[1].set_xlabel("log(1 + Comments)")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    p5 = os.path.join(FIGURES_DIR, "05_comments_distribution.png")
    plt.savefig(p5)
    plt.close()
    
    # -------------------------------------------------------------------------
    # 6. Video Category vs Average Views
    # -------------------------------------------------------------------------
    cat_views = df.groupby("category_name")["views"].mean().sort_values(ascending=False).reset_index()
    
    plt.figure(figsize=(10, 6))
    bar6 = sns.barplot(
        data=cat_views,
        y="category_name",
        x="views",
        palette="Blues_r",
        hue="category_name",
        legend=False
    )
    plt.title("Average Views Across Video Categories")
    plt.xlabel("Average Views (in Millions)")
    plt.ylabel("Category")
    bar6.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.2f}M"))
    for p in bar6.patches:
        width = p.get_width()
        bar6.annotate(f"{width/1e6:.2f}M",
                      (width, p.get_y() + p.get_height() / 2.),
                      ha='left', va='center', xytext=(5, 0), textcoords='offset points', fontsize=9)
    plt.tight_layout()
    p6 = os.path.join(FIGURES_DIR, "06_category_vs_avg_views.png")
    plt.savefig(p6)
    plt.close()
    
    findings["highest_avg_view_category"] = str(cat_views.iloc[0]["category_name"])
    findings["highest_avg_views"] = float(cat_views.iloc[0]["views"])
    
    # -------------------------------------------------------------------------
    # 7. Video Category vs Engagement Rate
    # -------------------------------------------------------------------------
    cat_eng = df.groupby("category_name")["engagement_rate"].mean().sort_values(ascending=False).reset_index()
    
    plt.figure(figsize=(10, 6))
    bar7 = sns.barplot(
        data=cat_eng,
        y="category_name",
        x="engagement_rate",
        palette="magma",
        hue="category_name",
        legend=False
    )
    plt.title("Average Viewer Engagement Rate (%) by Video Category")
    plt.xlabel("Average Engagement Rate (%) [Likes + Comments / Views * 100]")
    plt.ylabel("Category")
    for p in bar7.patches:
        width = p.get_width()
        bar7.annotate(f"{width:.2f}%",
                      (width, p.get_y() + p.get_height() / 2.),
                      ha='left', va='center', xytext=(5, 0), textcoords='offset points', fontsize=9)
    plt.tight_layout()
    p7 = os.path.join(FIGURES_DIR, "07_category_vs_engagement_rate.png")
    plt.savefig(p7)
    plt.close()
    
    findings["highest_engagement_category"] = str(cat_eng.iloc[0]["category_name"])
    findings["highest_engagement_rate"] = float(cat_eng.iloc[0]["engagement_rate"])
    
    # -------------------------------------------------------------------------
    # 8. Correlation Heatmap
    # -------------------------------------------------------------------------
    corr_cols = [
        "views", "likes", "dislikes", "comment_count",
        "engagement_rate", "like_ratio", "comment_ratio",
        "title_length", "tag_count", "desc_length"
    ]
    corr_matrix = df[corr_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5)
    plt.title("Correlation Matrix of Quantitative Metrics & Video Metadata")
    plt.tight_layout()
    p8 = os.path.join(FIGURES_DIR, "08_correlation_heatmap.png")
    plt.savefig(p8)
    plt.close()
    
    findings["correlations"] = {
        "views_vs_likes": float(corr_matrix.loc["views", "likes"]),
        "views_vs_comments": float(corr_matrix.loc["views", "comment_count"]),
        "likes_vs_comments": float(corr_matrix.loc["likes", "comment_count"]),
        "views_vs_engagement_rate": float(corr_matrix.loc["views", "engagement_rate"])
    }
    
    # -------------------------------------------------------------------------
    # 9. Publishing Hour vs Average Views
    # -------------------------------------------------------------------------
    hour_views = df.groupby("publish_hour")["views"].mean().reset_index()
    
    plt.figure(figsize=(11, 5))
    sns.lineplot(data=hour_views, x="publish_hour", y="views", marker="o", color="#2980b9", linewidth=2.5)
    plt.title("Publishing Hour (UTC) vs. Average Views")
    plt.xlabel("Hour of Publication (0 to 23)")
    plt.ylabel("Average Views")
    plt.xticks(range(0, 24))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e3:.0f}k"))
    peak_hour = int(hour_views.loc[hour_views["views"].idxmax()]["publish_hour"])
    plt.axvline(peak_hour, color="#c0392b", linestyle="--", alpha=0.7, label=f"Peak Hour: {peak_hour}:00")
    plt.legend()
    plt.tight_layout()
    p9 = os.path.join(FIGURES_DIR, "09_publish_hour_vs_views.png")
    plt.savefig(p9)
    plt.close()
    
    findings["peak_publishing_hour"] = peak_hour
    findings["peak_hour_views"] = float(hour_views.loc[hour_views["views"].idxmax()]["views"])
    
    # -------------------------------------------------------------------------
    # 10. Publishing Weekday vs Average Engagement
    # -------------------------------------------------------------------------
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_eng = df.groupby("publish_day_name")["engagement_rate"].mean().reindex(days_order).reset_index()
    
    plt.figure(figsize=(9, 5))
    bar10 = sns.barplot(
        data=day_eng,
        x="publish_day_name",
        y="engagement_rate",
        palette="crest",
        hue="publish_day_name",
        legend=False
    )
    plt.title("Average Viewer Engagement Rate (%) by Publishing Weekday")
    plt.xlabel("Publishing Weekday")
    plt.ylabel("Average Engagement Rate (%)")
    for p in bar10.patches:
        height = p.get_height()
        bar10.annotate(f"{height:.2f}%",
                       (p.get_x() + p.get_width() / 2., height),
                       ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=9)
    plt.tight_layout()
    p10 = os.path.join(FIGURES_DIR, "10_publish_weekday_vs_engagement.png")
    plt.savefig(p10)
    plt.close()
    
    findings["best_weekday_for_engagement"] = str(day_eng.loc[day_eng["engagement_rate"].idxmax()]["publish_day_name"])
    
    # -------------------------------------------------------------------------
    # 11. Likes vs Views Scatter Plot
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    sample_df = df.sample(min(5000, len(df)), random_state=42)
    sns.scatterplot(
        data=sample_df,
        x="views",
        y="likes",
        alpha=0.4,
        color="#2980b9",
        edgecolor=None
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Likes vs. Views (Log-Log Scale with Trend)")
    plt.xlabel("Views (Log Scale)")
    plt.ylabel("Likes (Log Scale)")
    # Overlay linear fit on log values
    m, b = np.polyfit(sample_df["log_views"], sample_df["log_likes"], 1)
    x_vals = np.linspace(sample_df["views"].min(), sample_df["views"].max(), 100)
    y_vals = np.exp(b) * (x_vals ** m)
    plt.plot(x_vals, y_vals, color="#e74c3c", linewidth=2, label=f"Power Law Trend (Slope = {m:.2f})")
    plt.legend()
    plt.tight_layout()
    p11 = os.path.join(FIGURES_DIR, "11_likes_vs_views_scatter.png")
    plt.savefig(p11)
    plt.close()
    
    # -------------------------------------------------------------------------
    # 12. Comments vs Views Scatter Plot
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=sample_df,
        x="views",
        y="comment_count",
        alpha=0.4,
        color="#8e44ad",
        edgecolor=None
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Comments vs. Views (Log-Log Scale with Trend)")
    plt.xlabel("Views (Log Scale)")
    plt.ylabel("Comments (Log Scale)")
    # Overlay linear fit on log values
    valid_sample = sample_df[sample_df["comment_count"] > 0]
    mc, bc = np.polyfit(valid_sample["log_views"], valid_sample["log_comments"], 1)
    x_valsc = np.linspace(valid_sample["views"].min(), valid_sample["views"].max(), 100)
    y_valsc = np.exp(bc) * (x_valsc ** mc)
    plt.plot(x_valsc, y_valsc, color="#d35400", linewidth=2, label=f"Power Law Trend (Slope = {mc:.2f})")
    plt.legend()
    plt.tight_layout()
    p12 = os.path.join(FIGURES_DIR, "12_comments_vs_views_scatter.png")
    plt.savefig(p12)
    plt.close()
    
    with open(EDA_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=4)
    print(f"[SUCCESS] All 12 EDA figures saved to: {FIGURES_DIR}")
    print(f"[SUCCESS] EDA summary report saved to: {EDA_REPORT_PATH}")
    
    return findings


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    findings = run_eda()
    print("\nKey EDA Observations:")
    for k, v in findings.items():
        print(f"  {k}: {v}")

"""
Jupyter Notebook Generator
==========================
Generates a complete, interactive, beautifully documented academic Jupyter Notebook
for YouTube_Video_Analysis.ipynb.
"""

import nbformat as nbf
import os

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_PATH = os.path.join(NOTEBOOK_DIR, "YouTube_Video_Analysis.ipynb")

nb = nbf.v4.new_notebook()
nb.metadata = {
    "language_info": {
        "name": "python",
        "version": "3.11"
    },
    "kernelspec": {
        "name": "python3",
        "display_name": "Python 3 (ipykernel)"
    }
}

cells = []

# Cell 1: Markdown Title & Academic Header
cells.append(nbf.v4.new_markdown_cell("""# YouTube Video Analysis: Identifying Factors Influencing Video Popularity and Viewer Engagement
### Subject: Data Mining & Data Warehousing (BCI24505D) - Activity-Based Assessment (AAT)
**Course:** 3rd Year B.E. Computer Science & Engineering (AI & ML)  
**Topic 42:** YouTube Video Analysis  
**Academic Year:** 2026  

---
## Project Overview
This study mines historical video metadata from the **Kaggle YouTube Trending Video Statistics (India Region)** dataset. Using both unsupervised and supervised data mining algorithms alongside an analytical **Star Schema Data Warehouse (SQLite)**, we identify key determinants of video virality, audience engagement rates, and viewer response behaviors.

### Mandatory AAT Components Covered:
1. **Real-world Dataset Acquisition & Integrity Validation**
2. **Comprehensive Preprocessing & Feature Engineering**
3. **Dual Data Mining Implementations:**
   - *Algorithm 1:* **K-Means Clustering** for Unsupervised Video Segmentation
   - *Algorithm 2:* **Decision Tree Classifier** for Metadata-based Engagement Prediction (Zero-Leakage)
4. **Empirical Results & Visualizations (17 Generated Figures)**
5. **Data Warehousing Star Schema & Analytical OLAP SQL Queries**
6. **Detailed Synthesis of 8 Academic Research Questions**
"""))

# Cell 2: Imports & Environment Configuration
cells.append(nbf.v4.new_code_cell("""# 1. Imports and Reproducibility Setup
import os
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Ensure reproducibility
np.random.seed(42)

# Styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 130
plt.rcParams['font.size'] = 10

print("Environment initialized successfully with Python", sys.version.split()[0])
"""))

# Cell 3: Markdown - Dataset Acquisition
cells.append(nbf.v4.new_markdown_cell("""## 1. Dataset Loading and Schema Inspection
We load the raw India trending dataset (`INvideos.csv`) and category mappings (`IN_category_id.json`).
"""))

# Cell 4: Code - Data Loading
cells.append(nbf.v4.new_code_cell("""# Load raw dataset and category JSON mapping
from src.data_loader import load_raw_data, load_category_mapping, validate_dataset

raw_df = load_raw_data()
cat_map = load_category_mapping()
is_valid, report = validate_dataset(raw_df)

print(f"Initial Dataset Dimensions: {raw_df.shape[0]:,} rows x {raw_df.shape[1]} columns")
print(f"Mapped Categories Count:   {len(cat_map)}")
raw_df[['video_id', 'title', 'channel_title', 'views', 'likes', 'dislikes', 'comment_count']].head()
"""))

# Cell 5: Markdown - Data Preprocessing
cells.append(nbf.v4.new_markdown_cell("""## 2. Rigorous Data Preprocessing & Feature Engineering
Key steps implemented:
1. **Handling Missing Values:** Imputed missing video descriptions.
2. **Deduplication:** Trending videos appear across multiple consecutive dates. We deduplicate by keeping the maximum view snapshot per unique `video_id`.
3. **Invalid Value Removal:** Filtered zero-view and error/removed videos.
4. **Temporal Extraction:** Extracted publish year, month, day, hour, and day of week.
5. **Metric Formulation:**
   $$\\text{Engagement Rate (\\%)} = \\frac{\\text{Likes} + \\text{Comments}}{\\text{Views}} \\times 100$$
   $$\\text{Like Ratio (\\%)} = \\frac{\\text{Likes}}{\\text{Views}} \\times 100$$
   $$\\text{Comment Ratio (\\%)} = \\frac{\\text{Comments}}{\\text{Views}} \\times 100$$
6. **Feature Engineering:** Title character length, description length, and tag counts.
7. **Log Transformations:** `log_views`, `log_likes`, and `log_comments` to address heavy right-skewness.
"""))

# Cell 6: Code - Preprocessing
cells.append(nbf.v4.new_code_cell("""# Run full preprocessing pipeline
from src.preprocessing import preprocess_data

clean_df, prep_report = preprocess_data()

print(f"Raw Records:              {prep_report['raw_records']:,}")
print(f"Duplicates Removed:       {prep_report['duplicate_snapshots_removed']:,}")
print(f"Unique Cleaned Videos:    {prep_report['final_cleaned_records']:,}")
print(f"Median Engagement Rate:   {prep_report['median_engagement_rate_threshold']:.3f}%")

clean_df[['video_id', 'title', 'category_name', 'views', 'likes', 'engagement_rate', 'high_engagement']].head()
"""))

# Cell 7: Markdown - Exploratory Data Analysis
cells.append(nbf.v4.new_markdown_cell("""## 3. Exploratory Data Analysis (EDA)
Investigating relationships across views, engagement rates, publishing timing, and content categories.
"""))

# Cell 8: Code - Top Videos & Channels
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top 10 Videos
top10_v = clean_df.nlargest(10, "views")[["title", "views"]].copy()
top10_v["short_title"] = top10_v["title"].apply(lambda t: (t[:32] + "...") if len(str(t)) > 32 else str(t))
sns.barplot(data=top10_v, y="short_title", x="views", ax=axes[0], palette="viridis", hue="short_title", legend=False)
axes[0].set_title("Top 10 Most Viewed Trending Videos")
axes[0].set_xlabel("Views (Millions)")
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M"))

# Top 10 Channels
top_ch = clean_df.groupby("channel_title")["views"].sum().nlargest(10).reset_index()
sns.barplot(data=top_ch, y="channel_title", x="views", ax=axes[1], palette="rocket", hue="channel_title", legend=False)
axes[1].set_title("Top 10 Channels by Cumulative Views")
axes[1].set_xlabel("Views (Millions)")
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M"))

plt.tight_layout()
plt.show()
"""))

# Cell 9: Code - Distributions & Correlations
cells.append(nbf.v4.new_code_cell("""# Correlation Matrix
corr_cols = ["views", "likes", "dislikes", "comment_count", "engagement_rate", "title_length", "tag_count", "desc_length"]
corr_mat = clean_df[corr_cols].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Matrix of Quantitative Metrics & Video Metadata")
plt.tight_layout()
plt.show()
"""))

# Cell 10: Code - Category vs Views & Engagement
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cat_v = clean_df.groupby("category_name")["views"].mean().sort_values(ascending=False).reset_index()
sns.barplot(data=cat_v, y="category_name", x="views", ax=axes[0], palette="Blues_r", hue="category_name", legend=False)
axes[0].set_title("Average Views by Category")
axes[0].set_xlabel("Views (Millions)")
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.2f}M"))

cat_e = clean_df.groupby("category_name")["engagement_rate"].mean().sort_values(ascending=False).reset_index()
sns.barplot(data=cat_e, y="category_name", x="engagement_rate", ax=axes[1], palette="magma", hue="category_name", legend=False)
axes[1].set_title("Average Engagement Rate (%) by Category")
axes[1].set_xlabel("Engagement Rate (%)")

plt.tight_layout()
plt.show()
"""))

# Cell 11: Markdown - Algorithm 1
cells.append(nbf.v4.new_markdown_cell("""## 4. Algorithm 1: K-Means Clustering
### Objective
Unsupervised segmentation of YouTube videos into distinct popularity and engagement profiles.
Using `[log_views, log_likes, log_comments, engagement_rate]` scaled via `StandardScaler`.
"""))

# Cell 12: Code - K-Means Execution
cells.append(nbf.v4.new_code_cell("""from src.clustering import run_clustering

clustered_df, clust_report = run_clustering()

print(f"Optimal Clusters: k = {clust_report['optimal_k']}")
print("\\nCluster Characteristic Profiles:")
for cname, p in clust_report['cluster_profiles'].items():
    print(f"\\n--- {cname}: {p['semantic_label']} ---")
    print(f"  Share: {p['percentage']}% ({p['count']} videos)")
    print(f"  Median Views: {p['views_median']:,.0f} | Median Engagement: {p['engagement_rate_median']:.2f}%")
    print(f"  Dominant Genre: {p['top_category']}")
"""))

# Cell 13: Markdown - Algorithm 2
cells.append(nbf.v4.new_markdown_cell("""## 5. Algorithm 2: Decision Tree Classifier (Zero Data Leakage)
### Objective
Predict whether a video achieves **High Engagement** (above median) using strictly pre-publication metadata:
`[publish_hour, publish_weekday, title_length, tag_count, desc_length, category_dummies]`.
Views, likes, comments, and engagement rates are strictly excluded to avoid data leakage!
"""))

# Cell 14: Code - Decision Tree Execution
cells.append(nbf.v4.new_code_cell("""from src.classification import run_classification

dt_model, clf_report = run_classification()
metrics = clf_report["evaluation_metrics"]

print(f"Decision Tree Accuracy:  {metrics['accuracy']*100:.2f}%")
print(f"Decision Tree Precision: {metrics['precision']*100:.2f}%")
print(f"Decision Tree Recall:    {metrics['recall']*100:.2f}%")
print(f"Decision Tree F1-Score:  {metrics['f1_score']*100:.2f}%")
"""))

# Cell 15: Markdown - Data Warehouse
cells.append(nbf.v4.new_markdown_cell("""## 6. Data Warehousing Component: SQLite Star Schema
A Dimensional Data Warehouse comprising:
- **Fact Table:** `fact_video_performance` (Views, Likes, Comments, Engagement Rate, Foreign Keys)
- **Dimension Tables:** `dim_category`, `dim_channel`, `dim_date`
"""))

# Cell 16: Code - Warehouse ETL and SQL Execution
cells.append(nbf.v4.new_code_cell("""from src.database import populate_warehouse, run_analytical_queries

etl_counts = populate_warehouse()
sql_results = run_analytical_queries()
"""))

# Cell 17: Markdown - Research Synthesis
cells.append(nbf.v4.new_markdown_cell("""## 7. Master Evaluation & Synthesis of Research Questions
Comprehensive answers derived from our empirical findings.
"""))

# Cell 18: Code - Evaluation Synthesis
cells.append(nbf.v4.new_code_cell("""from src.evaluation import synthesize_research_answers

summary = synthesize_research_answers()

for rk, rdata in summary["research_questions_answers"].items():
    print("=" * 70)
    print(f"{rk}: {rdata['question']}")
    print(f"-> {rdata['empirical_finding']}")
    print(f"   [Basis]: {rdata['statistical_basis']}")
"""))

nb.cells = cells

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"[SUCCESS] Jupyter Notebook created at: {NOTEBOOK_PATH}")

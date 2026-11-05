"""
End-to-End Pipeline Execution Script
====================================
Course: Data Mining & Data Warehousing (BCI24505D)
Topic 42: YouTube Video Analysis

Executes the entire analytics pipeline in strict sequential order:
  Step 1: Dataset Acquisition & Validation
  Step 2: Preprocessing, Deduplication & Feature Engineering
  Step 3: Exploratory Data Analysis (12 Visualizations)
  Step 4: Algorithm 1 - K-Means Clustering
  Step 5: Algorithm 2 - Decision Tree Classification
  Step 6: Data Warehousing (Star Schema ETL + Analytical Queries)
  Step 7: Master Evaluation & Research Synthesis
"""

import sys
import os

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import ensure_dataset_present, load_raw_data, validate_dataset
from src.preprocessing import preprocess_data
from src.eda import run_eda
from src.clustering import run_clustering
from src.classification import run_classification
from src.database import populate_warehouse, run_analytical_queries
from src.evaluation import synthesize_research_answers


def main():
    print("=" * 75)
    print("  DMDW AAT: YOUTUBE VIDEO POPULARITY & ENGAGEMENT ANALYSIS")
    print("  Course: BCI24505D | BE CSE (AI & ML) | Academic Year 2026")
    print("=" * 75)
    
    # -------------------------------------------------------------------------
    # STEP 1: Dataset Acquisition & Validation
    # -------------------------------------------------------------------------
    print("\n>>> [STEP 1/7] ACQUIRING & VALIDATING DATASET...")
    ensure_dataset_present()
    raw_df = load_raw_data()
    passed, v_report = validate_dataset(raw_df)
    print(f"  [+] Loaded {v_report['total_rows']:,} raw records with {v_report['total_columns']} attributes.")
    
    # -------------------------------------------------------------------------
    # STEP 2: Preprocessing & Feature Engineering
    # -------------------------------------------------------------------------
    print("\n>>> [STEP 2/7] PREPROCESSING & FEATURE ENGINEERING...")
    clean_df, p_report = preprocess_data()
    print(f"  [+] Deduplicated to {p_report['final_cleaned_records']:,} unique videos.")
    print(f"  [+] Removed {p_report['duplicate_snapshots_removed']:,} duplicate trending snapshots.")
    print(f"  [+] Median Engagement Threshold: {p_report['median_engagement_rate_threshold']:.3f}%")
    
    # -------------------------------------------------------------------------
    # STEP 3: Exploratory Data Analysis (12 Visualizations)
    # -------------------------------------------------------------------------
    print("\n>>> [STEP 3/7] GENERATING 12 EDA VISUALIZATIONS...")
    eda_findings = run_eda()
    print(f"  [+] Saved 12 figures to 'outputs/figures/'.")
    print(f"  [+] Top category by average views: {eda_findings.get('highest_avg_view_category')}")
    print(f"  [+] Peak publishing hour: {eda_findings.get('peak_publishing_hour')}:00 UTC")
    
    # -------------------------------------------------------------------------
    # STEP 4: Algorithm 1 - K-Means Clustering
    # -------------------------------------------------------------------------
    print("\n>>> [STEP 4/7] ALGORITHM 1: K-MEANS CLUSTERING...")
    clust_df, c_report = run_clustering()
    print(f"  [+] Evaluated Elbow Curve & Silhouette scores.")
    print(f"  [+] Clustered into k={c_report['optimal_k']} popularity groups.")
    
    # -------------------------------------------------------------------------
    # STEP 5: Algorithm 2 - Decision Tree Classification
    # -------------------------------------------------------------------------
    print("\n>>> [STEP 5/7] ALGORITHM 2: DECISION TREE CLASSIFICATION...")
    dt_model, clf_report = run_classification()
    metrics = clf_report["evaluation_metrics"]
    print(f"  [+] Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  [+] Precision: {metrics['precision']*100:.2f}%")
    print(f"  [+] Recall:    {metrics['recall']*100:.2f}%")
    print(f"  [+] F1-Score:  {metrics['f1_score']*100:.2f}%")
    
    # -------------------------------------------------------------------------
    # STEP 6: Data Warehousing (Star Schema ETL + Analytical Queries)
    # -------------------------------------------------------------------------
    print("\n>>> [STEP 6/7] DATA WAREHOUSE (SQLITE STAR SCHEMA)...")
    etl_counts = populate_warehouse()
    print(f"  [+] Fact records loaded: {etl_counts['fact_records']:,}")
    print(f"  [+] Dimensions loaded: {etl_counts['dim_categories']} categories, {etl_counts['dim_channels']:,} channels, {etl_counts['dim_dates']:,} dates.")
    sql_results = run_analytical_queries()
    print(f"  [+] Executed 5+ analytical SQL queries successfully.")
    
    # -------------------------------------------------------------------------
    # STEP 7: Master Evaluation & Research Synthesis
    # -------------------------------------------------------------------------
    print("\n>>> [STEP 7/7] MASTER EVALUATION & RESEARCH SYNTHESIS...")
    master_summary = synthesize_research_answers()
    print("  [+] Synthesized comprehensive answers for all 8 Research Questions.")
    
    print("\n" + "=" * 75)
    print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("  Cleaned Data:    data/processed/cleaned_youtube_data.csv")
    print("  Warehouse DB:    data/youtube_warehouse.db")
    print("  Visualizations:  outputs/figures/ (17 total figures generated)")
    print("  Models:          outputs/models/")
    print("  Reports:         outputs/reports/")
    print("=" * 75)


if __name__ == "__main__":
    main()

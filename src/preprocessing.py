"""
Data Preprocessing & Feature Engineering Module
===============================================
Course: Data Mining & Data Warehousing (BCI24505D)
Topic 42: YouTube Video Analysis

This module performs rigorous, reproducible data cleaning, feature engineering,
and transformation on the raw YouTube Trending dataset.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from src.data_loader import load_raw_data, load_category_mapping, CSV_PATH, JSON_PATH

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
OUTPUT_REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "reports")
CLEANED_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_youtube_data.csv")
PREPROCESSING_REPORT_PATH = os.path.join(OUTPUT_REPORTS_DIR, "preprocessing_report.json")


def detect_outliers_iqr(series: pd.Series) -> Dict[str, float]:
    """Calculate IQR bounds and count of outliers for a numerical series."""
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower_bound = float(q1 - 1.5 * iqr)
    upper_bound = float(q3 + 1.5 * iqr)
    outliers_count = int(((series < lower_bound) | (series > upper_bound)).sum())
    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": outliers_count,
        "outlier_pct": round((outliers_count / len(series)) * 100, 2) if len(series) > 0 else 0.0
    }


def preprocess_data(raw_csv: str = CSV_PATH, category_json: str = JSON_PATH) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute end-to-end data preprocessing pipeline:
      1. Load raw data and category mapping
      2. Missing value audit and imputation
      3. Deduplication (retain highest-view snapshot per unique video)
      4. Invalid/zero value filtering
      5. Datetime decomposition (year, month, day, hour, weekday)
      6. Category name mapping
      7. Ratio and engagement rate calculations
      8. Skewness handling via log transformations
      9. Text metadata feature engineering (title length, tag count, desc length)
      10. Outlier detection via IQR
      11. Target variable assignment for classification
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_REPORTS_DIR, exist_ok=True)
    
    print("[1/8] Loading raw data & category metadata...")
    df_raw = load_raw_data(raw_csv)
    cat_map = load_category_mapping(category_json)
    
    initial_shape = df_raw.shape
    initial_nulls = df_raw.isnull().sum().to_dict()
    initial_dups = int(df_raw.duplicated(subset=["video_id"]).sum())
    
    df = df_raw.copy()
    
    # [2/8] Missing value handling
    # Descriptions frequently have nulls for videos with no text description
    null_desc_count = int(df["description"].isnull().sum())
    df["description"] = df["description"].fillna("")
    
    # Drop rows missing crucial identifiers or timestamps if any exist
    df = df.dropna(subset=["video_id", "title", "publish_time"])
    
    # [3/8] Handling duplicates
    # A video may trend on multiple consecutive days. For video-level factor analysis,
    # we sort by views descending and keep the final/maximum snapshot per video_id.
    df = df.sort_values(by="views", ascending=False)
    dedup_before = len(df)
    df = df.drop_duplicates(subset=["video_id"], keep="first")
    duplicates_removed = dedup_before - len(df)
    
    # [4/8] Invalid & zero value filtering
    # Remove videos that were flagged as error/removed or had 0 views
    initial_filter_len = len(df)
    df = df[df["video_error_or_removed"] == False]
    df = df[df["views"] > 0]
    invalid_removed = initial_filter_len - len(df)
    
    # [5/8] Datetime parsing & temporal feature extraction
    df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
    df = df.dropna(subset=["publish_time"])
    
    df["publish_year"] = df["publish_time"].dt.year
    df["publish_month"] = df["publish_time"].dt.month
    df["publish_day"] = df["publish_time"].dt.day
    df["publish_hour"] = df["publish_time"].dt.hour
    df["publish_weekday"] = df["publish_time"].dt.weekday  # 0=Monday, 6=Sunday
    
    weekday_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    df["publish_day_name"] = df["publish_weekday"].map(weekday_names)
    
    # [6/8] Category ID cleaning & mapping
    df["category_id"] = df["category_id"].astype(int)
    df["category_name"] = df["category_id"].map(cat_map).fillna("Other / Unknown")
    
    # [7/8] Metric & Ratio Calculations (safe division: views guaranteed > 0)
    # Engagement Rate (%) = ((Likes + Comments) / Views) * 100
    df["engagement_rate"] = ((df["likes"] + df["comment_count"]) / df["views"]) * 100
    # Like Ratio (%) = (Likes / Views) * 100
    df["like_ratio"] = (df["likes"] / df["views"]) * 100
    # Comment Ratio (%) = (Comments / Views) * 100
    df["comment_ratio"] = (df["comment_count"] / df["views"]) * 100
    # Dislike Ratio (%) = (Dislikes / Views) * 100
    df["dislike_ratio"] = (df["dislikes"] / df["views"]) * 100
    
    # Cap ratio anomalies at 100% in case of metric reporting lags
    df["engagement_rate"] = df["engagement_rate"].clip(upper=100.0)
    df["like_ratio"] = df["like_ratio"].clip(upper=100.0)
    df["comment_ratio"] = df["comment_ratio"].clip(upper=100.0)
    df["dislike_ratio"] = df["dislike_ratio"].clip(upper=100.0)
    
    # [8/8] Feature Engineering & Transformations
    # Title length (number of characters)
    df["title_length"] = df["title"].astype(str).str.len()
    
    # Tag count (splitting tags on '|', treating '[none]' as 0)
    def parse_tag_count(tag_str: str) -> int:
        if pd.isna(tag_str) or tag_str.strip() in ["[none]", '""', ""]:
            return 0
        return len([t for t in tag_str.split("|") if t.strip()])
    
    df["tag_count"] = df["tags"].apply(parse_tag_count)
    
    # Description character length
    df["desc_length"] = df["description"].astype(str).str.len()
    
    # Log transformations for heavily skewed numerical distributions
    df["log_views"] = np.log1p(df["views"])
    df["log_likes"] = np.log1p(df["likes"])
    df["log_comments"] = np.log1p(df["comment_count"])
    df["log_dislikes"] = np.log1p(df["dislikes"])
    
    # Target variable for classification: High engagement vs Normal/Low
    # Threshold = median engagement rate across all trending videos
    median_engagement = float(df["engagement_rate"].median())
    df["high_engagement"] = (df["engagement_rate"] >= median_engagement).astype(int)
    
    # Outlier detection statistics
    outlier_stats = {
        "views": detect_outliers_iqr(df["views"]),
        "likes": detect_outliers_iqr(df["likes"]),
        "comment_count": detect_outliers_iqr(df["comment_count"]),
        "engagement_rate": detect_outliers_iqr(df["engagement_rate"])
    }
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Save cleaned dataset
    df.to_csv(CLEANED_CSV_PATH, index=False)
    print(f"[SUCCESS] Cleaned dataset saved to: {CLEANED_CSV_PATH} ({len(df)} records)")
    
    # Generate comprehensive report
    report = {
        "raw_records": initial_shape[0],
        "raw_features": initial_shape[1],
        "null_descriptions_imputed": null_desc_count,
        "duplicate_snapshots_removed": duplicates_removed,
        "invalid_or_zero_view_records_removed": invalid_removed,
        "final_cleaned_records": len(df),
        "final_features_count": len(df.columns),
        "median_engagement_rate_threshold": round(median_engagement, 4),
        "high_engagement_count": int(df["high_engagement"].sum()),
        "low_engagement_count": int((df["high_engagement"] == 0).sum()),
        "outlier_iqr_analysis": outlier_stats,
        "summary_statistics": {
            "views": {
                "mean": float(df["views"].mean()),
                "median": float(df["views"].median()),
                "std": float(df["views"].std()),
                "min": float(df["views"].min()),
                "max": float(df["views"].max())
            },
            "likes": {
                "mean": float(df["likes"].mean()),
                "median": float(df["likes"].median()),
                "std": float(df["likes"].std()),
                "min": float(df["likes"].min()),
                "max": float(df["likes"].max())
            },
            "comment_count": {
                "mean": float(df["comment_count"].mean()),
                "median": float(df["comment_count"].median()),
                "std": float(df["comment_count"].std()),
                "min": float(df["comment_count"].min()),
                "max": float(df["comment_count"].max())
            },
            "engagement_rate": {
                "mean": float(df["engagement_rate"].mean()),
                "median": float(df["engagement_rate"].median()),
                "std": float(df["engagement_rate"].std()),
                "min": float(df["engagement_rate"].min()),
                "max": float(df["engagement_rate"].max())
            }
        }
    }
    
    with open(PREPROCESSING_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"[SUCCESS] Preprocessing report saved to: {PREPROCESSING_REPORT_PATH}")
    
    return df, report


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 2: DATA PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 60)
    clean_df, prep_report = preprocess_data()
    print("\nSummary of Processed Dataset:")
    print(f"Total Unique Videos: {prep_report['final_cleaned_records']}")
    print(f"Duplicates Removed: {prep_report['duplicate_snapshots_removed']}")
    print(f"Median Engagement Rate Threshold: {prep_report['median_engagement_rate_threshold']:.3f}%")
    print(f"Views Mean: {prep_report['summary_statistics']['views']['mean']:,.0f} | Median: {prep_report['summary_statistics']['views']['median']:,.0f}")
    print(f"Likes Mean: {prep_report['summary_statistics']['likes']['mean']:,.0f} | Median: {prep_report['summary_statistics']['likes']['median']:,.0f}")

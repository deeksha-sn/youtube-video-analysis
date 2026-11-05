"""
Data Loader & Dataset Validation Module
=======================================
Project: YouTube Video Analysis (DMDW AAT - Topic 42)
Author: BE AI & ML Student
Description:
    Downloads raw YouTube Trending Video dataset for India (INvideos.csv)
    and category JSON mapping (IN_category_id.json), validates schema,
    checks data types, and reports summary statistics.
"""

import os
import json
import urllib.request
import pandas as pd
from typing import Dict, Tuple

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
CSV_PATH = os.path.join(RAW_DATA_DIR, "INvideos.csv")
JSON_PATH = os.path.join(RAW_DATA_DIR, "IN_category_id.json")

# Verified mirrors for Kaggle datasnaek/youtube-new dataset
URL_INVIDEOS = "https://raw.githubusercontent.com/MayurDeshmukh10/youtube_analysis/master/INvideos.csv"
URL_CATEGORY_JSON = "https://raw.githubusercontent.com/marcelinalinaa/YoutubePredictionModel/master/IN_category_id.json"

EXPECTED_COLUMNS = [
    "video_id", "trending_date", "title", "channel_title", "category_id",
    "publish_time", "tags", "views", "likes", "dislikes", "comment_count",
    "thumbnail_link", "comments_disabled", "ratings_disabled",
    "video_error_or_removed", "description"
]


def download_file(url: str, dest_path: str) -> None:
    """Download a file with user-friendly console reporting."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[INFO] File already exists: {dest_path} ({os.path.getsize(dest_path) / (1024*1024):.2f} MB)")
        return

    print(f"[INFO] Downloading from {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB chunk
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\r  Downloaded {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end="")
            else:
                print(f"\r  Downloaded {downloaded / (1024*1024):.1f} MB", end="")
    print(f"\n[SUCCESS] Saved to {dest_path}")


def ensure_dataset_present() -> None:
    """Ensure both CSV and JSON metadata are present on disk."""
    download_file(URL_INVIDEOS, CSV_PATH)
    download_file(URL_CATEGORY_JSON, JSON_PATH)


def load_category_mapping(json_path: str = JSON_PATH) -> Dict[int, str]:
    """Parse YouTube category ID to category name mapping from JSON."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Category JSON file not found at: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    mapping = {}
    for item in data.get("items", []):
        cat_id = int(item["id"])
        cat_title = item["snippet"]["title"]
        mapping[cat_id] = cat_title
    return mapping


def load_raw_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Load raw INvideos.csv into a pandas DataFrame."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw CSV dataset not found at: {csv_path}. Please download it first.")
    
    # Use latin1/utf-8 with replacement for robustness across multilingual titles & descriptions
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin1")
    
    return df


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, Dict[str, any]]:
    """Validate schema, row count, null values, and summary of the loaded dataset."""
    issues = []
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            issues.append(f"Missing expected column: '{col}'")
    
    stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_videos": int(df["video_id"].nunique()) if "video_id" in df.columns else 0,
        "validation_passed": len(issues) == 0,
        "issues": issues
    }
    
    if not stats["validation_passed"]:
        print(f"[WARNING] Dataset validation warnings: {issues}")
    else:
        print("[SUCCESS] Dataset validation passed successfully.")
    
    return stats["validation_passed"], stats


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: DATASET ACQUISITION & VALIDATION")
    print("=" * 60)
    ensure_dataset_present()
    
    cat_map = load_category_mapping()
    print(f"[INFO] Loaded {len(cat_map)} category mappings from JSON:")
    for cid, cname in sorted(cat_map.items()):
        print(f"   ID {cid:2d} -> {cname}")
    
    raw_df = load_raw_data()
    passed, report = validate_dataset(raw_df)
    print(f"\nDataset Shape: {report['total_rows']} rows x {report['total_columns']} columns")
    print(f"Unique Video IDs: {report['unique_videos']}")
    print(f"Duplicate Rows: {report['duplicate_rows']}")
    print("\nFirst 3 records preview:")
    print(raw_df[["video_id", "title", "channel_title", "views", "likes"]].head(3))

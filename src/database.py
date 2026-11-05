"""
Data Warehouse Database & ETL Module
====================================
Course: Data Mining & Data Warehousing (BCI24505D)
Topic 42: YouTube Video Analysis

Implements the SQLite Star Schema Data Warehouse:
  - Executes schema.sql (Fact + Dimensions)
  - Performs ETL pipeline from cleaned/clustered dataset
  - Executes analytical SQL queries and generates reports
"""

import os
import sqlite3
import json
import pandas as pd
from typing import Dict, Any, List

from src.preprocessing import CLEANED_CSV_PATH

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "youtube_warehouse.db")
SQL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sql")
SCHEMA_SQL_PATH = os.path.join(SQL_DIR, "schema.sql")
QUERIES_SQL_PATH = os.path.join(SQL_DIR, "analysis_queries.sql")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "reports")
SQL_REPORT_PATH = os.path.join(REPORTS_DIR, "sql_query_results.json")


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Establish connection to SQLite database with foreign keys enabled."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_database(db_path: str = DB_PATH, schema_path: str = SCHEMA_SQL_PATH) -> None:
    """Execute DDL statements to construct Star Schema."""
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema DDL not found at: {schema_path}")
    
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    
    conn = get_db_connection(db_path)
    with conn:
        conn.executescript(schema_sql)
    conn.close()
    print(f"[SUCCESS] Initialized Star Schema at: {db_path}")


def populate_warehouse(csv_path: str = None, db_path: str = DB_PATH) -> Dict[str, int]:
    """ETL Pipeline: Extracts cleaned data, transforms dimensions, and loads into fact table."""
    if csv_path is None:
        clustered_path = os.path.join(os.path.dirname(CLEANED_CSV_PATH), "clustered_youtube_data.csv")
        csv_path = clustered_path if os.path.exists(clustered_path) else CLEANED_CSV_PATH
    
    df = pd.read_csv(csv_path)
    print(f"[INFO] Running ETL pipeline for {len(df)} records from {csv_path}...")
    
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # Ensure tables exist
    init_database(db_path)
    
    # -------------------------------------------------------------------------
    # 1. Populate dim_category
    # -------------------------------------------------------------------------
    unique_cats = df[["category_id", "category_name"]].drop_duplicates().sort_values("category_id")
    for _, row in unique_cats.iterrows():
        cid = int(row["category_id"])
        cname = str(row["category_name"])
        cursor.execute(
            "INSERT OR IGNORE INTO dim_category (category_key, category_id, category_name) VALUES (?, ?, ?)",
            (cid, cid, cname)
        )
    print(f"  Loaded {len(unique_cats)} records into dim_category.")
    
    # -------------------------------------------------------------------------
    # 2. Populate dim_channel
    # -------------------------------------------------------------------------
    unique_channels = sorted(df["channel_title"].dropna().unique())
    for ch in unique_channels:
        cursor.execute(
            "INSERT OR IGNORE INTO dim_channel (channel_title) VALUES (?)",
            (str(ch),)
        )
    print(f"  Loaded {len(unique_channels)} records into dim_channel.")
    
    # Fetch channel mapping
    cursor.execute("SELECT channel_title, channel_key FROM dim_channel")
    channel_map = {row[0]: row[1] for row in cursor.fetchall()}
    
    # -------------------------------------------------------------------------
    # 3. Populate dim_date
    # -------------------------------------------------------------------------
    df["publish_time"] = pd.to_datetime(df["publish_time"])
    unique_dates = df["publish_time"].dt.date.unique()
    
    for d in unique_dates:
        date_ts = pd.Timestamp(d)
        date_key = int(date_ts.strftime("%Y%m%d"))
        full_date = date_ts.strftime("%Y-%m-%d")
        year = int(date_ts.year)
        month = int(date_ts.month)
        month_name = date_ts.strftime("%B")
        day = int(date_ts.day)
        day_of_week = int(date_ts.weekday())
        day_name = date_ts.strftime("%A")
        is_weekend = 1 if day_of_week in [5, 6] else 0
        
        cursor.execute(
            """INSERT OR IGNORE INTO dim_date 
               (date_key, full_date, year, month, month_name, day, day_of_week, day_name, is_weekend) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (date_key, full_date, year, month, month_name, day, day_of_week, day_name, is_weekend)
        )
    print(f"  Loaded {len(unique_dates)} dates into dim_date.")
    
    # -------------------------------------------------------------------------
    # 4. Populate fact_video_performance
    # -------------------------------------------------------------------------
    cluster_col_present = "cluster_id" in df.columns
    fact_rows = []
    
    for _, row in df.iterrows():
        vid = str(row["video_id"])
        title = str(row["title"])
        c_title = str(row["channel_title"])
        c_key = channel_map.get(c_title, 1)
        cat_key = int(row["category_id"])
        
        date_ts = pd.Timestamp(row["publish_time"])
        d_key = int(date_ts.strftime("%Y%m%d"))
        
        p_hour = int(row.get("publish_hour", date_ts.hour))
        views = int(row["views"])
        likes = int(row["likes"])
        dislikes = int(row.get("dislikes", 0))
        comments = int(row["comment_count"])
        eng_rate = float(row["engagement_rate"])
        like_ratio = float(row.get("like_ratio", 0.0))
        comment_ratio = float(row.get("comment_ratio", 0.0))
        title_len = int(row.get("title_length", len(title)))
        tag_cnt = int(row.get("tag_count", 0))
        desc_len = int(row.get("desc_length", 0))
        high_eng = int(row.get("high_engagement", 0))
        cluster_id = int(row["cluster_id"]) if cluster_col_present else -1
        
        fact_rows.append((
            vid, title, c_key, cat_key, d_key, p_hour,
            views, likes, dislikes, comments, eng_rate,
            like_ratio, comment_ratio, title_len, tag_cnt,
            desc_len, high_eng, cluster_id
        ))
    
    cursor.executemany(
        """INSERT INTO fact_video_performance
           (video_id, title, channel_key, category_key, date_key, publish_hour,
            views, likes, dislikes, comment_count, engagement_rate,
            like_ratio, comment_ratio, title_length, tag_count,
            desc_length, high_engagement, cluster_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        fact_rows
    )
    
    conn.commit()
    conn.close()
    
    counts = {
        "fact_records": len(fact_rows),
        "dim_categories": len(unique_cats),
        "dim_channels": len(unique_channels),
        "dim_dates": len(unique_dates)
    }
    print(f"[SUCCESS] Data Warehouse ETL completed: {counts}")
    return counts


def run_analytical_queries(db_path: str = DB_PATH) -> Dict[str, Any]:
    """Execute analytical SQL queries representing standard OLAP reports."""
    conn = get_db_connection(db_path)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    queries = {
        "Q1_Top_10_Videos": """
            SELECT 
                f.title, c.channel_title, cat.category_name,
                f.views, f.likes, f.comment_count,
                ROUND(f.engagement_rate, 2) AS engagement_rate_pct
            FROM fact_video_performance f
            JOIN dim_channel c ON f.channel_key = c.channel_key
            JOIN dim_category cat ON f.category_key = cat.category_key
            ORDER BY f.views DESC
            LIMIT 10;
        """,
        "Q2_Category_Summary": """
            SELECT 
                cat.category_name,
                COUNT(f.video_key) AS total_videos,
                SUM(f.views) AS aggregate_views,
                ROUND(AVG(f.views), 0) AS avg_views,
                ROUND(AVG(f.likes), 0) AS avg_likes,
                ROUND(AVG(f.engagement_rate), 2) AS avg_engagement_rate_pct
            FROM fact_video_performance f
            JOIN dim_category cat ON f.category_key = cat.category_key
            GROUP BY cat.category_name
            ORDER BY avg_views DESC;
        """,
        "Q3_Top_10_Channels": """
            SELECT 
                c.channel_title,
                COUNT(f.video_key) AS video_count,
                SUM(f.views) AS total_views,
                ROUND(AVG(f.views), 0) AS avg_views,
                ROUND(AVG(f.engagement_rate), 2) AS avg_engagement_rate_pct
            FROM fact_video_performance f
            JOIN dim_channel c ON f.channel_key = c.channel_key
            GROUP BY c.channel_title
            HAVING video_count >= 3
            ORDER BY total_views DESC
            LIMIT 10;
        """,
        "Q4_Monthly_Performance": """
            SELECT 
                d.year, d.month, d.month_name,
                COUNT(f.video_key) AS videos_published,
                SUM(f.views) AS total_views,
                ROUND(AVG(f.views), 0) AS avg_views,
                ROUND(AVG(f.engagement_rate), 2) AS avg_engagement_pct
            FROM fact_video_performance f
            JOIN dim_date d ON f.date_key = d.date_key
            GROUP BY d.year, d.month, d.month_name
            ORDER BY d.year ASC, d.month ASC;
        """,
        "Q5_High_Engagement_Outliers": """
            SELECT 
                f.title, c.channel_title, cat.category_name,
                f.views,
                ROUND(f.engagement_rate, 2) AS video_engagement_rate,
                ROUND(cat_stats.cat_avg, 2) AS cat_avg_engagement,
                ROUND(f.engagement_rate - cat_stats.cat_avg, 2) AS engagement_lift
            FROM fact_video_performance f
            JOIN dim_channel c ON f.channel_key = c.channel_key
            JOIN dim_category cat ON f.category_key = cat.category_key
            JOIN (
                SELECT category_key, AVG(engagement_rate) AS cat_avg
                FROM fact_video_performance
                GROUP BY category_key
            ) cat_stats ON f.category_key = cat_stats.category_key
            WHERE f.views >= 500000 
              AND f.engagement_rate > (cat_stats.cat_avg * 1.5)
            ORDER BY f.engagement_rate DESC
            LIMIT 10;
        """
    }
    
    results = {}
    print("\n" + "=" * 60)
    print("EXECUTING ANALYTICAL SQL QUERIES (DATA WAREHOUSE)")
    print("=" * 60)
    
    for qname, qsql in queries.items():
        df_res = pd.read_sql_query(qsql, conn)
        results[qname] = df_res.to_dict(orient="records")
        print(f"\n--- {qname} (Top Records) ---")
        print(df_res.head(5).to_string(index=False))
    
    conn.close()
    
    with open(SQL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n[SUCCESS] Analytical query results saved to: {SQL_REPORT_PATH}")
    
    return results


if __name__ == "__main__":
    populate_warehouse()
    run_analytical_queries()

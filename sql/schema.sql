-- =====================================================================
-- Star Schema DDL for YouTube Video Analytics Data Warehouse
-- Course: Data Mining & Data Warehousing (BCI24505D) - AAT
-- Database Engine: SQLite 3
-- =====================================================================

-- Drop existing tables if they exist to support clean rebuilds
DROP TABLE IF EXISTS fact_video_performance;
DROP TABLE IF EXISTS dim_channel;
DROP TABLE IF EXISTS dim_category;
DROP TABLE IF EXISTS dim_date;

-- ---------------------------------------------------------------------
-- 1. Dimension Table: dim_category
-- Stores distinct YouTube video categories with descriptive names.
-- ---------------------------------------------------------------------
CREATE TABLE dim_category (
    category_key INTEGER PRIMARY KEY,
    category_id INTEGER NOT NULL UNIQUE,
    category_name TEXT NOT NULL
);

-- ---------------------------------------------------------------------
-- 2. Dimension Table: dim_channel
-- Stores unique channel titles and publisher metadata.
-- ---------------------------------------------------------------------
CREATE TABLE dim_channel (
    channel_key INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_title TEXT NOT NULL UNIQUE
);

-- ---------------------------------------------------------------------
-- 3. Dimension Table: dim_date
-- Date dimension supporting temporal rollups (Year, Month, Day, Weekday).
-- ---------------------------------------------------------------------
CREATE TABLE dim_date (
    date_key INTEGER PRIMARY KEY,       -- e.g., 20171114
    full_date TEXT NOT NULL,           -- e.g., '2017-11-14'
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    month_name TEXT NOT NULL,
    day INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL,      -- 0=Monday, 6=Sunday
    day_name TEXT NOT NULL,
    is_weekend INTEGER NOT NULL        -- 0=Weekday, 1=Weekend
);

-- ---------------------------------------------------------------------
-- 4. Fact Table: fact_video_performance
-- Contains quantitative measurements, metrics, and foreign keys.
-- ---------------------------------------------------------------------
CREATE TABLE fact_video_performance (
    video_key INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    title TEXT NOT NULL,
    channel_key INTEGER NOT NULL,
    category_key INTEGER NOT NULL,
    date_key INTEGER NOT NULL,
    publish_hour INTEGER NOT NULL,
    views INTEGER NOT NULL,
    likes INTEGER NOT NULL,
    dislikes INTEGER NOT NULL,
    comment_count INTEGER NOT NULL,
    engagement_rate REAL NOT NULL,
    like_ratio REAL NOT NULL,
    comment_ratio REAL NOT NULL,
    title_length INTEGER NOT NULL,
    tag_count INTEGER NOT NULL,
    desc_length INTEGER NOT NULL,
    high_engagement INTEGER NOT NULL,  -- 1 = High Engagement, 0 = Standard
    cluster_id INTEGER DEFAULT -1,     -- Assigned via K-Means clustering
    FOREIGN KEY (channel_key) REFERENCES dim_channel(channel_key),
    FOREIGN KEY (category_key) REFERENCES dim_category(category_key),
    FOREIGN KEY (date_key) REFERENCES dim_date(date_key)
);

-- Create performance indexes for analytical aggregation queries
CREATE INDEX idx_fact_channel ON fact_video_performance(channel_key);
CREATE INDEX idx_fact_category ON fact_video_performance(category_key);
CREATE INDEX idx_fact_date ON fact_video_performance(date_key);
CREATE INDEX idx_fact_views ON fact_video_performance(views DESC);
CREATE INDEX idx_fact_engagement ON fact_video_performance(engagement_rate DESC);

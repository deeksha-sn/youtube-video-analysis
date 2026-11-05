-- =====================================================================
-- Analytical SQL Queries for YouTube Data Warehouse
-- Course: Data Mining & Data Warehousing (BCI24505D)
-- Star Schema Query Demonstrations
-- =====================================================================

-- ---------------------------------------------------------------------
-- Query 1: Top 10 Videos by Total Views
-- Demonstrates multi-table join between Fact table and Dimension tables.
-- ---------------------------------------------------------------------
-- Research Question: Which individual videos achieved maximum viewership?
SELECT 
    f.video_id,
    f.title,
    c.channel_title,
    cat.category_name,
    f.views,
    f.likes,
    f.comment_count,
    ROUND(f.engagement_rate, 2) AS engagement_rate_pct
FROM fact_video_performance f
JOIN dim_channel c ON f.channel_key = c.channel_key
JOIN dim_category cat ON f.category_key = cat.category_key
ORDER BY f.views DESC
LIMIT 10;

-- ---------------------------------------------------------------------
-- Query 2: Category-Level Performance & Viewer Engagement Summary
-- Aggregates views, likes, comments, and average engagement by genre.
-- ---------------------------------------------------------------------
-- Research Question: Which categories receive higher average views and engagement?
SELECT 
    cat.category_name,
    COUNT(f.video_key) AS total_videos,
    SUM(f.views) AS aggregate_views,
    ROUND(AVG(f.views), 0) AS avg_views,
    ROUND(AVG(f.likes), 0) AS avg_likes,
    ROUND(AVG(f.comment_count), 0) AS avg_comments,
    ROUND(AVG(f.engagement_rate), 2) AS avg_engagement_rate_pct
FROM fact_video_performance f
JOIN dim_category cat ON f.category_key = cat.category_key
GROUP BY cat.category_name
ORDER BY avg_views DESC;

-- ---------------------------------------------------------------------
-- Query 3: Top 10 Channels by Total Views & Video Production
-- Demonstrates channel benchmarking and audience reach.
-- ---------------------------------------------------------------------
-- Research Question: Which channels command the largest cumulative viewership?
SELECT 
    c.channel_title,
    COUNT(f.video_key) AS trending_video_count,
    SUM(f.views) AS total_views,
    ROUND(AVG(f.views), 0) AS avg_views_per_video,
    SUM(f.likes) AS total_likes,
    ROUND(AVG(f.engagement_rate), 2) AS avg_engagement_rate_pct
FROM fact_video_performance f
JOIN dim_channel c ON f.channel_key = c.channel_key
GROUP BY c.channel_title
HAVING trending_video_count >= 3
ORDER BY total_views DESC
LIMIT 10;

-- ---------------------------------------------------------------------
-- Query 4: Monthly Video Performance & Seasonality Trends
-- Aggregates metrics using the Date dimension.
-- ---------------------------------------------------------------------
-- Research Question: How do video uploads and engagement fluctuate across calendar months?
SELECT 
    d.year,
    d.month,
    d.month_name,
    COUNT(f.video_key) AS videos_published,
    SUM(f.views) AS total_views,
    ROUND(AVG(f.views), 0) AS avg_views,
    ROUND(AVG(f.engagement_rate), 2) AS avg_engagement_rate_pct
FROM fact_video_performance f
JOIN dim_date d ON f.date_key = d.date_key
GROUP BY d.year, d.month, d.month_name
ORDER BY d.year ASC, d.month ASC;

-- ---------------------------------------------------------------------
-- Query 5: High-Engagement Video Identification (Above Category Average)
-- Uses window functions/subqueries to filter high-impact videos.
-- ---------------------------------------------------------------------
-- Research Question: Which videos generate disproportionately high engagement compared to category norms?
SELECT 
    f.video_id,
    f.title,
    c.channel_title,
    cat.category_name,
    f.views,
    ROUND(f.engagement_rate, 2) AS video_engagement_rate,
    ROUND(cat_stats.category_avg_engagement, 2) AS category_avg_engagement,
    ROUND(f.engagement_rate - cat_stats.category_avg_engagement, 2) AS engagement_lift
FROM fact_video_performance f
JOIN dim_channel c ON f.channel_key = c.channel_key
JOIN dim_category cat ON f.category_key = cat.category_key
JOIN (
    SELECT category_key, AVG(engagement_rate) AS category_avg_engagement
    FROM fact_video_performance
    GROUP BY category_key
) cat_stats ON f.category_key = cat_stats.category_key
WHERE f.views >= 500000 
  AND f.engagement_rate > (cat_stats.category_avg_engagement * 1.5)
ORDER BY f.engagement_rate DESC
LIMIT 15;

-- ---------------------------------------------------------------------
-- Query 6: Day-of-Week Publishing Impact on Virality
-- Evaluates weekday vs weekend upload performance.
-- ---------------------------------------------------------------------
SELECT 
    d.day_name,
    d.is_weekend,
    COUNT(f.video_key) AS upload_count,
    ROUND(AVG(f.views), 0) AS avg_views,
    ROUND(AVG(f.engagement_rate), 2) AS avg_engagement_rate_pct
FROM fact_video_performance f
JOIN dim_date d ON f.date_key = d.date_key
GROUP BY d.day_of_week, d.day_name, d.is_weekend
ORDER BY d.day_of_week ASC;

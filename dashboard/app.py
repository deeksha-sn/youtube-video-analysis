"""
Interactive Streamlit Analytics Dashboard
=========================================
Course: Data Mining & Data Warehousing (BCI24505D)
Topic 42: YouTube Video Analysis
Title: YouTube Video Popularity & Viewer Engagement Intelligence Dashboard

Launch Command:
    streamlit run dashboard/app.py
"""

import os
import sys
import sqlite3
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="YouTube Video Analytics | DMDW AAT",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "clustered_youtube_data.csv")
FALLBACK_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_youtube_data.csv")
DB_PATH = os.path.join(BASE_DIR, "data", "youtube_warehouse.db")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "decision_tree_model.joblib")


@st.cache_data
def load_data():
    target_path = DATA_PATH if os.path.exists(DATA_PATH) else FALLBACK_DATA_PATH
    if not os.path.exists(target_path):
        st.error(f"Dataset not found at {target_path}. Please execute `python main.py` first.")
        st.stop()
    df = pd.read_csv(target_path)
    df["publish_time"] = pd.to_datetime(df["publish_time"])
    return df


df_master = load_data()

# -----------------------------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e1/Logo_of_YouTube_%282015-2017%29.svg", width=180)
st.sidebar.title("Navigation & Filters")
st.sidebar.markdown("**Project:** DMDW AAT (BCI24505D)\n**Topic 42:** YouTube Video Mining")

# Category Filter
all_categories = sorted(df_master["category_name"].unique())
selected_categories = st.sidebar.multiselect(
    "Select Video Categories:",
    options=all_categories,
    default=all_categories
)

# Search Channel Filter
channel_search = st.sidebar.text_input("Search Specific Channel (optional):", "")

# Engagement Slider
min_eng = float(df_master["engagement_rate"].min())
max_eng = float(df_master["engagement_rate"].max())
eng_range = st.sidebar.slider(
    "Filter by Engagement Rate (%):",
    min_value=0.0,
    max_value=min(max_eng, 25.0),
    value=(0.0, 25.0),
    step=0.5
)

# Apply Filters
filtered_df = df_master[
    (df_master["category_name"].isin(selected_categories)) &
    (df_master["engagement_rate"] >= eng_range[0]) &
    (df_master["engagement_rate"] <= eng_range[1])
]
if channel_search.strip():
    filtered_df = filtered_df[filtered_df["channel_title"].str.contains(channel_search, case=False, na=False)]

# -----------------------------------------------------------------------------
# MAIN HEADER & KEY METRIC TILES
# -----------------------------------------------------------------------------
st.title("🎬 YouTube Video Popularity & Viewer Engagement Dashboard")
st.markdown(
    "An academic decision-support and data mining interface analyzing **trending factors, audience engagement, "
    "K-Means video clustering, and Decision Tree engagement prediction** using real-world YouTube India data."
)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Videos", f"{len(filtered_df):,}")
with col2:
    total_views = filtered_df["views"].sum()
    st.metric("Total Views", f"{total_views/1e6:.1f}M" if total_views < 1e9 else f"{total_views/1e9:.2f}B")
with col3:
    total_likes = filtered_df["likes"].sum()
    st.metric("Total Likes", f"{total_likes/1e6:.1f}M")
with col4:
    total_comments = filtered_df["comment_count"].sum()
    st.metric("Total Comments", f"{total_comments/1e6:.1f}M")
with col5:
    avg_eng = filtered_df["engagement_rate"].mean() if len(filtered_df) > 0 else 0
    st.metric("Avg Engagement Rate", f"{avg_eng:.2f}%")

st.markdown("---")

# -----------------------------------------------------------------------------
# DASHBOARD TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Category & Channel Analytics",
    "🕒 Publishing Timing & Seasonality",
    "🎯 K-Means Popularity Clusters",
    "🤖 Predictive Engagement Engine",
    "🏛️ Data Warehouse (Star Schema SQL)"
])

# -----------------------------------------------------------------------------
# TAB 1: Category & Channel Analytics
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Category Performance vs. Viewer Engagement")
    col_t1a, col_t1b = st.columns(2)
    
    with col_t1a:
        cat_summary = (
            filtered_df.groupby("category_name")["views"]
            .mean()
            .reset_index()
            .sort_values(by="views", ascending=False)
        )
        fig_cat_views = px.bar(
            cat_summary,
            x="views",
            y="category_name",
            orientation="h",
            title="Average Views by Category",
            labels={"views": "Average Views", "category_name": "Category"},
            color="views",
            color_continuous_scale="Blues"
        )
        fig_cat_views.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
        st.plotly_chart(fig_cat_views, use_container_width=True)
    
    with col_t1b:
        cat_eng_summary = (
            filtered_df.groupby("category_name")["engagement_rate"]
            .mean()
            .reset_index()
            .sort_values(by="engagement_rate", ascending=False)
        )
        fig_cat_eng = px.bar(
            cat_eng_summary,
            x="engagement_rate",
            y="category_name",
            orientation="h",
            title="Average Engagement Rate (%) by Category",
            labels={"engagement_rate": "Engagement Rate (%)", "category_name": "Category"},
            color="engagement_rate",
            color_continuous_scale="Viridis"
        )
        fig_cat_eng.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
        st.plotly_chart(fig_cat_eng, use_container_width=True)
    
    st.subheader("Top Performing Channels")
    top_channels = (
        filtered_df.groupby("channel_title")
        .agg(total_views=("views", "sum"), total_likes=("likes", "sum"), count=("views", "count"))
        .reset_index()
        .sort_values(by="total_views", ascending=False)
        .head(15)
    )
    fig_channels = px.bar(
        top_channels,
        x="total_views",
        y="channel_title",
        orientation="h",
        title="Top 15 Channels by Cumulative Views",
        labels={"total_views": "Total Views", "channel_title": "Channel"},
        color="count",
        color_continuous_scale="Reds"
    )
    fig_channels.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
    st.plotly_chart(fig_channels, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: Timing & Seasonality
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Temporal Dynamics of Virality")
    col_t2a, col_t2b = st.columns(2)
    
    with col_t2a:
        hour_stats = filtered_df.groupby("publish_hour")["views"].mean().reset_index()
        fig_hour = px.line(
            hour_stats,
            x="publish_hour",
            y="views",
            markers=True,
            title="Publishing Hour (UTC) vs. Average Views",
            labels={"publish_hour": "Hour of Day (UTC)", "views": "Average Views"}
        )
        fig_hour.update_traces(line_color="#2980b9", line_width=3)
        st.plotly_chart(fig_hour, use_container_width=True)
    
    with col_t2b:
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_stats = filtered_df.groupby("publish_day_name")["engagement_rate"].mean().reindex(day_order).reset_index()
        fig_day = px.bar(
            day_stats,
            x="publish_day_name",
            y="engagement_rate",
            title="Publishing Weekday vs. Average Engagement Rate",
            labels={"publish_day_name": "Day of Week", "engagement_rate": "Engagement Rate (%)"},
            color="engagement_rate",
            color_continuous_scale="Teal"
        )
        st.plotly_chart(fig_day, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: K-Means Popularity Clusters
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Unsupervised Video Segmentation (K-Means Clustering)")
    if "cluster_name" in filtered_df.columns:
        fig_cluster = px.scatter(
            filtered_df.sample(min(3000, len(filtered_df)), random_state=42),
            x="views",
            y="engagement_rate",
            color="cluster_name",
            log_x=True,
            hover_data=["title", "channel_title", "category_name"],
            title="Interactive Cluster Map: Views vs. Engagement Rate",
            labels={"views": "Views (Log Scale)", "engagement_rate": "Engagement Rate (%)"},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        st.markdown("### Cluster Characteristic Summary Table")
        cluster_grp = (
            filtered_df.groupby("cluster_name")
            .agg(
                video_count=("views", "count"),
                median_views=("views", "median"),
                avg_likes=("likes", "mean"),
                avg_engagement=("engagement_rate", "mean")
            )
            .reset_index()
        )
        st.dataframe(cluster_grp.style.format({
            "video_count": "{:,}",
            "median_views": "{:,.0f}",
            "avg_likes": "{:,.0f}",
            "avg_engagement": "{:.2f}%"
        }), use_container_width=True)
    else:
        st.warning("Clustering column not found in dataset. Please run `main.py`.")

# -----------------------------------------------------------------------------
# TAB 4: Predictive Engagement Engine (Decision Tree)
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Interactive Engagement Classifier (Decision Tree)")
    st.markdown(
        "Predict whether a planned video is likely to achieve **High Engagement (Above Median)** "
        "using strictly pre-publication metadata (Category, Timing, Text Lengths)."
    )
    
    col_pred1, col_pred2 = st.columns(2)
    with col_pred1:
        in_cat = st.selectbox("Video Category:", options=all_categories)
        in_hour = st.slider("Target Upload Hour (UTC):", 0, 23, 12)
        in_day = st.selectbox("Upload Day:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    with col_pred2:
        in_title_len = st.slider("Title Character Length:", 10, 100, 50)
        in_tag_cnt = st.slider("Number of Tags:", 0, 50, 15)
        in_desc_len = st.slider("Description Character Length:", 0, 5000, 500)
    
    if st.button("🔮 Predict Engagement Level"):
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            # Build feature vector matching training schema
            day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
            
            # Feature columns
            row_dict = {
                "publish_hour": in_hour,
                "publish_weekday": day_map[in_day],
                "title_length": in_title_len,
                "tag_count": in_tag_cnt,
                "desc_length": in_desc_len
            }
            # Add categorical dummies
            for cat in all_categories:
                col_name = f"cat_{cat}"
                row_dict[col_name] = 1 if cat == in_cat else 0
            
            # Align with model expected features
            model_features = model.feature_names_in_
            feature_vector = pd.DataFrame([[row_dict.get(c, 0) for c in model_features]], columns=model_features)
            
            pred = model.predict(feature_vector)[0]
            prob = model.predict_proba(feature_vector)[0]
            
            if pred == 1:
                st.success(f"🎉 **HIGH ENGAGEMENT PREDICTED!** (Confidence: {prob[1]*100:.1f}%)")
                st.info("Videos with this profile historically elicit strong audience interaction (comments & likes per view).")
            else:
                st.warning(f"⚠️ **STANDARD / LOW ENGAGEMENT PREDICTED** (Confidence: {prob[0]*100:.1f}%)")
                st.info("Recommendation: Consider refining the title, increasing community hooks, or adjusting release timing.")
        else:
            st.error("Decision tree model not found. Run `python src/classification.py` first.")

# -----------------------------------------------------------------------------
# TAB 5: Data Warehouse & SQL Console
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("Data Warehousing Component: SQLite Star Schema")
    st.markdown(
        "Demonstrating analytical querying against the **Star Schema Data Warehouse** "
        "consisting of `fact_video_performance` joined with `dim_category`, `dim_channel`, and `dim_date`."
    )
    
    query_options = {
        "Top 10 Most Viewed Videos": """
            SELECT f.title, c.channel_title, cat.category_name, f.views, f.likes,
                   ROUND(f.engagement_rate, 2) AS eng_rate_pct
            FROM fact_video_performance f
            JOIN dim_channel c ON f.channel_key = c.channel_key
            JOIN dim_category cat ON f.category_key = cat.category_key
            ORDER BY f.views DESC LIMIT 10;
        """,
        "Category Views & Engagement Aggregates": """
            SELECT cat.category_name, COUNT(f.video_key) AS video_count,
                   ROUND(AVG(f.views), 0) AS avg_views,
                   ROUND(AVG(f.engagement_rate), 2) AS avg_eng_rate_pct
            FROM fact_video_performance f
            JOIN dim_category cat ON f.category_key = cat.category_key
            GROUP BY cat.category_name ORDER BY avg_views DESC;
        """,
        "Weekday Virality Analysis": """
            SELECT d.day_name, COUNT(f.video_key) AS total_uploads,
                   ROUND(AVG(f.views), 0) AS avg_views,
                   ROUND(AVG(f.engagement_rate), 2) AS avg_eng_pct
            FROM fact_video_performance f
            JOIN dim_date d ON f.date_key = d.date_key
            GROUP BY d.day_of_week, d.day_name ORDER BY d.day_of_week;
        """
    }
    
    selected_query_title = st.selectbox("Select Pre-configured OLAP Query:", list(query_options.keys()))
    sql_text = query_options[selected_query_title]
    st.code(sql_text, language="sql")
    
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        df_sql = pd.read_sql_query(sql_text, conn)
        conn.close()
        st.dataframe(df_sql, use_container_width=True)
    else:
        st.warning("Database not found. Please run `python src/database.py` to populate warehouse.")

"""
Interactive Streamlit Analytics Dashboard - Minimal Professional Sidebar
========================================================================
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
    page_title="YouTube Analytics Dashboard | DMDW AAT",
    page_icon="https://www.youtube.com/s/desktop/f71df42a/img/favicon_32x32.png",
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
# SVG LUCIDE ICONS (NO EMOJIS)
# -----------------------------------------------------------------------------
ICONS = {
    "video": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m22 8-6 4 6 4V8Z"/><rect width="14" height="12" x="2" y="6" rx="2"/></svg>',
    "eye": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>',
    "thumbs_up": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 10v12"/><path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.5L12 2.6a2 2 0 0 1 3 3.28Z"/></svg>',
    "message": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
    "percent": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="19" x2="5" y1="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/></svg>',
    "bar_chart": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" x2="12" y1="20" y2="10"/><line x1="18" x2="18" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="16"/></svg>',
    "users": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    "trophy": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/></svg>',
    "clapperboard": '<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#0f2b5c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.2 6 3 11l-.9-2.4c-.3-1.1.3-2.2 1.4-2.5l13.5-4c1.1-.3 2.2.3 2.5 1.4Z"/><path d="m6.2 5.3 3.1 4"/><path d="m12.1 3.6 3.1 4"/><path d="M2 11h20v9a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2Z"/></svg>',
    "clock": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    "calendar": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="4" rx="2" ry="2"/><line x1="16" x2="16" y1="2" y2="6"/><line x1="8" x2="8" y1="2" y2="6"/><line x1="3" x2="21" y1="10" y2="10"/></svg>',
    "layers": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/></svg>',
    "target": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "database": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/></svg>'
}

# -----------------------------------------------------------------------------
# GLOBAL INJECTED CSS: PROFESSIONAL NAVY & SKY-BLUE THEME (MINIMAL SIDEBAR)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* Light sky-blue background for main app */
.stApp {
    background: linear-gradient(180deg, #f0f4f9 0%, #f6f9fc 100%) !important;
}

/* Main Content Container Padding */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2.8rem !important;
    padding-left: 2.2rem !important;
    padding-right: 2.2rem !important;
    max-width: 98% !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

/* SIDEBAR: Dark Navy-Blue with Subtle Gradient */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #091e42 0%, #0a2554 60%, #061733 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding-top: 1rem !important;
}

[data-testid="stSidebar"] * {
    color: #ffffff;
}

/* Sidebar Navigation Radio Styling */
[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
    gap: 4px !important;
}

[data-testid="stSidebar"] div[data-testid="stRadio"] label {
    background: transparent !important;
    border-radius: 8px !important;
    padding: 9px 12px !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    color: #cbd5e1 !important;
    transition: all 0.2s ease !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    margin-bottom: 2px !important;
    border: 1px solid transparent !important;
}

[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
}

[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked),
[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] {
    background: #0284c7 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 10px rgba(2, 132, 199, 0.3) !important;
}

/* Hide Radio Dot */
[data-testid="stSidebar"] div[data-testid="stRadio"] div[data-testid="stRadioButton"] {
    display: none !important;
}
[data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"] {
    display: none !important;
}

/* Lucide Vector Icons on Sidebar Navigation Labels via CSS */
[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-of-type(1)::before {
    content: "";
    display: inline-block;
    width: 17px;
    height: 17px;
    flex-shrink: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='12' x2='12' y1='20' y2='10'/%3E%3Cline x1='18' x2='18' y1='20' y2='4'/%3E%3Cline x1='6' x2='6' y1='20' y2='16'/%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
}

[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-of-type(2)::before {
    content: "";
    display: inline-block;
    width: 17px;
    height: 17px;
    flex-shrink: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpolyline points='12 6 12 12 16 14'/%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
}

[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-of-type(3)::before {
    content: "";
    display: inline-block;
    width: 17px;
    height: 17px;
    flex-shrink: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z'/%3E%3Cpath d='m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65'/%3E%3Cpath d='m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65'/%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
}

[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-of-type(4)::before {
    content: "";
    display: inline-block;
    width: 17px;
    height: 17px;
    flex-shrink: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Ccircle cx='12' cy='12' r='6'/%3E%3Ccircle cx='12' cy='12' r='2'/%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
}

[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-of-type(5)::before {
    content: "";
    display: inline-block;
    width: 17px;
    height: 17px;
    flex-shrink: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cellipse cx='12' cy='5' rx='9' ry='3'/%3E%3Cpath d='M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5'/%3E%3Cpath d='M3 12c0 1.66 4 3 9 3s9-1.34 9-3'/%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
}

/* Sidebar Filters Inputs */
[data-testid="stSidebar"] .stTextInput input {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border-radius: 8px !important;
    border: 1px solid #cbd5e1 !important;
    font-size: 13.5px !important;
    padding: 8px 12px !important;
}

[data-testid="stSidebar"] .stTextInput input::placeholder {
    color: #94a3b8 !important;
}

[data-testid="stSidebar"] span[data-baseweb="tag"] {
    background-color: #0284c7 !important;
    color: #ffffff !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    border: none !important;
}

[data-testid="stSidebar"] [data-baseweb="slider"] {
    color: #38bdf8 !important;
}

/* Dashboard Card */
.dashboard-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 4px 14px rgba(15, 43, 92, 0.04);
    margin-bottom: 20px;
}

.card-title {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
    font-size: 15.5px;
    font-weight: 700;
    color: #0f2b5c;
}

.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 22px 0 16px 0;
}

.section-pill {
    width: 4px;
    height: 22px;
    background: #0284c7;
    border-radius: 2px;
}

.section-title {
    margin: 0;
    font-size: 19px;
    font-weight: 700;
    color: #0f2b5c;
}

.stButton > button {
    background: #0284c7 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 10px rgba(2, 132, 199, 0.25) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: #0369a1 !important;
    box-shadow: 0 6px 14px rgba(2, 132, 199, 0.35) !important;
    transform: translateY(-1px);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MINIMAL PROFESSIONAL SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    # 1. YouTube Logo and YouTube Analytics Branding
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-top: 4px; margin-bottom: 2px;">
        <svg width="38" height="26" viewBox="0 0 68 48">
            <path d="M66.52,7.74c-0.78-2.93-2.49-5.41-5.42-6.19C55.79,.13,34,0,34,0S12.21,.13,6.9,1.55 C3.97,2.33,2.27,4.81,1.48,7.74C0.06,13.05,0,24,0,24s0.06,10.95,1.48,16.26c0.78,2.93,2.49,5.41,5.42,6.19 C12.21,47.87,34,48,34,48s21.79-.13,27.1-1.55c2.93-.78,4.64-3.26,5.42-6.19C67.94,34.95,68,24,68,24S67.94,13.05,66.52,7.74z" fill="#FF0000"/>
            <polygon points="45,24 27,14 27,34" fill="#FFFFFF"/>
        </svg>
        <span style="font-size: 22px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px;">YouTube</span>
    </div>
    <div style="font-size: 18px; font-weight: 700; color: #ffffff; margin-bottom: 12px; margin-top: -2px;">
        YouTube Analytics
    </div>
    <hr style="border: 0; height: 1px; background: rgba(255, 255, 255, 0.12); margin: 0 0 14px 0;" />
    """, unsafe_allow_html=True)
    
    # 2. Navigation Links with Clean Lucide Icons
    nav_options = [
        "Category & Channel Analytics",
        "Publishing Timing & Seasonality",
        "K-Means Popularity Clusters",
        "Predictive Engagement Engine",
        "Data Warehouse (Star Schema SQL)"
    ]
    
    selected_nav = st.radio(
        "Navigation Links",
        options=nav_options,
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("""
    <hr style="border: 0; height: 1px; background: rgba(255, 255, 255, 0.12); margin: 14px 0 16px 0;" />
    """, unsafe_allow_html=True)
    
    # 3. Video Category Filter
    all_categories = sorted(df_master["category_name"].unique())
    selected_categories = st.multiselect(
        "Select Video Categories:",
        options=all_categories,
        default=all_categories
    )
    
    # 4. Channel Search
    channel_search = st.text_input(
        "Search Specific Channel (optional):",
        placeholder="Search channel..."
    )
    
    # 5. Engagement Rate Slider
    min_eng = float(df_master["engagement_rate"].min())
    max_eng = float(df_master["engagement_rate"].max())
    eng_range = st.slider(
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
# MAIN DASHBOARD HEADER & WATERMARK (MATCHING IMAGE 2)
# -----------------------------------------------------------------------------
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 22px;">
    <div>
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
            {ICONS['clapperboard']}
            <h1 style="margin: 0; font-size: 26px; font-weight: 800; color: #0f2b5c; letter-spacing: -0.3px;">
                YouTube Video Popularity & Viewer Engagement Dashboard
            </h1>
        </div>
        <div style="font-size: 13.5px; color: #475569; max-width: 950px; line-height: 1.45;">
            An academic decision-support and data mining interface analyzing trending factors, audience engagement, K-Means video clustering, and Decision Tree engagement prediction using real-world YouTube India data.
        </div>
    </div>
    <div style="background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%); width: 72px; height: 50px; border-radius: 16px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(2, 132, 199, 0.15);">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
            <rect width="24" height="24" rx="6" fill="#0284c7"/>
            <polygon points="9.5,7.5 16.5,12 9.5,16.5" fill="#ffffff"/>
        </svg>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5 ROUNDED PASTEL KPI METRIC CARDS (MATCHING IMAGE 2 EXACTLY)
# -----------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

total_vids = len(filtered_df)
total_views = filtered_df["views"].sum()
views_str = f"{total_views/1e9:.2f}B" if total_views >= 1e9 else f"{total_views/1e6:.1f}M"
total_likes = filtered_df["likes"].sum()
likes_str = f"{total_likes/1e6:.1f}M"
total_comms = filtered_df["comment_count"].sum()
comms_str = f"{total_comms/1e6:.1f}M"
avg_eng = filtered_df["engagement_rate"].mean() if len(filtered_df) > 0 else 0
eng_str = f"{avg_eng:.2f}%"

def render_kpi_card(title, value, icon_svg, bg_color, border_color, icon_bg, icon_color):
    return f"""
    <div style="background: {bg_color}; border: 1px solid {border_color}; border-radius: 14px; padding: 14px 16px; box-shadow: 0 2px 8px rgba(15, 43, 92, 0.03); display: flex; align-items: center; gap: 14px;">
        <div style="width: 44px; height: 44px; border-radius: 12px; background: {icon_bg}; color: {icon_color}; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
            {icon_svg}
        </div>
        <div style="overflow: hidden;">
            <div style="font-size: 13px; font-weight: 500; color: #475569; margin-bottom: 2px; white-space: nowrap;">{title}</div>
            <div style="font-size: 24px; font-weight: 800; color: #0f2b5c; line-height: 1.1;">{value}</div>
        </div>
    </div>
    """

with col1:
    st.markdown(render_kpi_card("Total Videos", f"{total_vids:,}", ICONS['video'], "#ebf5ff", "#dbeafe", "#e0f2fe", "#0284c7"), unsafe_allow_html=True)
with col2:
    st.markdown(render_kpi_card("Total Views", views_str, ICONS['eye'], "#f5f3ff", "#ede9fe", "#ede9fe", "#7c3aed"), unsafe_allow_html=True)
with col3:
    st.markdown(render_kpi_card("Total Likes", likes_str, ICONS['thumbs_up'], "#f0fdf4", "#dcfce7", "#dcfce7", "#16a34a"), unsafe_allow_html=True)
with col4:
    st.markdown(render_kpi_card("Total Comments", comms_str, ICONS['message'], "#fffbeb", "#fef3c7", "#fef3c7", "#d97706"), unsafe_allow_html=True)
with col5:
    st.markdown(render_kpi_card("Avg Engagement Rate", eng_str, ICONS['percent'], "#ecfeff", "#cffafe", "#cffafe", "#0891b2"), unsafe_allow_html=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HORIZONTAL NAVIGATION BAR (MATCHING IMAGE 2)
# -----------------------------------------------------------------------------
nav_icon_keys = {
    "Category & Channel Analytics": "bar_chart",
    "Publishing Timing & Seasonality": "clock",
    "K-Means Popularity Clusters": "layers",
    "Predictive Engagement Engine": "target",
    "Data Warehouse (Star Schema SQL)": "database"
}

pills_html = '<div style="display: flex; gap: 12px; margin-top: 10px; margin-bottom: 22px; flex-wrap: wrap;">'
for item in nav_options:
    is_active = (item == selected_nav)
    style = "background: #0284c7; color: #ffffff; border: 1px solid #0284c7; box-shadow: 0 4px 12px rgba(2, 132, 199, 0.28); font-weight: 600;" if is_active else "background: #ffffff; color: #1e293b; border: 1px solid #cbd5e1; box-shadow: 0 1px 3px rgba(15, 43, 92, 0.04); font-weight: 500;"
    icon_svg = ICONS[nav_icon_keys[item]]
    pills_html += f'<div style="{style} border-radius: 10px; padding: 9px 18px; font-size: 13.5px; display: flex; align-items: center; gap: 8px;">{icon_svg}<span>{item}</span></div>'
pills_html += '</div>'
st.markdown(pills_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SECTION 1: CATEGORY & CHANNEL ANALYTICS
# -----------------------------------------------------------------------------
if selected_nav == "Category & Channel Analytics":
    st.markdown("""
    <div class="section-header">
        <div class="section-pill"></div>
        <div class="section-title">Category Performance vs. Viewer Engagement</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_t1a, col_t1b = st.columns(2)
    
    with col_t1a:
        st.markdown(f"""
        <div class="dashboard-card" style="padding-bottom: 4px;">
            <div class="card-title">
                <span style="color: #0284c7;">{ICONS['bar_chart']}</span>
                <span>Average Views by Category</span>
            </div>
        """, unsafe_allow_html=True)
        
        cat_summary = (
            filtered_df.groupby("category_name")["views"]
            .mean()
            .reset_index()
            .sort_values(by="views", ascending=True)
        )
        
        fig_cat_views = px.bar(
            cat_summary,
            x="views",
            y="category_name",
            orientation="h",
            labels={"views": "Average Views", "category_name": "Category"},
            color="views",
            color_continuous_scale=[
                [0.0, "#bfdbfe"],
                [0.25, "#60a5fa"],
                [0.65, "#1d4ed8"],
                [1.0, "#0b2553"]
            ]
        )
        fig_cat_views.update_layout(
            margin=dict(l=10, r=20, t=10, b=30),
            height=430,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=11, color="#334155"),
            xaxis=dict(
                gridcolor="#f1f5f9",
                tickformat=".0s",
                title=dict(text="Average Views", font=dict(size=11, color="#64748b"))
            ),
            yaxis=dict(
                title=dict(text="Category", font=dict(size=11, color="#64748b")),
                tickfont=dict(size=10.5)
            ),
            coloraxis_colorbar=dict(
                title=dict(text="Average Views", font=dict(size=11, color="#64748b")),
                tickformat=".0s",
                thickness=12,
                len=0.75
            )
        )
        st.plotly_chart(fig_cat_views, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_t1b:
        st.markdown(f"""
        <div class="dashboard-card" style="padding-bottom: 4px;">
            <div class="card-title">
                <span style="color: #0284c7;">{ICONS['users']}</span>
                <span>Average Engagement Rate (%) by Category</span>
            </div>
        """, unsafe_allow_html=True)
        
        cat_eng_summary = (
            filtered_df.groupby("category_name")["engagement_rate"]
            .mean()
            .reset_index()
            .sort_values(by="engagement_rate", ascending=True)
        )
        
        fig_cat_eng = px.bar(
            cat_eng_summary,
            x="engagement_rate",
            y="category_name",
            orientation="h",
            labels={"engagement_rate": "Engagement Rate (%)", "category_name": "Category"},
            color="engagement_rate",
            color_continuous_scale=[
                [0.0, "#311042"],
                [0.2, "#432874"],
                [0.4, "#1d4ed8"],
                [0.6, "#0d9488"],
                [0.8, "#22c55e"],
                [1.0, "#facc15"]
            ]
        )
        fig_cat_eng.update_layout(
            margin=dict(l=10, r=20, t=10, b=30),
            height=430,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=11, color="#334155"),
            xaxis=dict(
                gridcolor="#f1f5f9",
                title=dict(text="Engagement Rate (%)", font=dict(size=11, color="#64748b"))
            ),
            yaxis=dict(
                title=dict(text="Category", font=dict(size=11, color="#64748b")),
                tickfont=dict(size=10.5)
            ),
            coloraxis_colorbar=dict(
                title=dict(text="Engagement Rate (%)", font=dict(size=11, color="#64748b")),
                thickness=12,
                len=0.75
            )
        )
        st.plotly_chart(fig_cat_eng, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Section 2: Top Performing Channels
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px; margin: 26px 0 16px 0;">
        <div style="color: #0284c7; display: flex; align-items: center;">
            {ICONS['trophy']}
        </div>
        <div class="section-title">Top Performing Channels</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="dashboard-card" style="padding-bottom: 4px;">
        <div class="card-title">
            <span style="color: #0284c7;">{ICONS['video']}</span>
            <span>Top 15 Channels by Cumulative Views</span>
        </div>
    """, unsafe_allow_html=True)
    
    top_channels = (
        filtered_df.groupby("channel_title")
        .agg(total_views=("views", "sum"), total_likes=("likes", "sum"), count=("views", "count"))
        .reset_index()
        .sort_values(by="total_views", ascending=True)
        .tail(15)
    )
    
    fig_channels = px.bar(
        top_channels,
        x="total_views",
        y="channel_title",
        orientation="h",
        labels={"total_views": "Total Views", "channel_title": "Channel"},
        color="count",
        color_continuous_scale=[
            [0.0, "#fecaca"],
            [0.35, "#f87171"],
            [0.7, "#dc2626"],
            [1.0, "#7f1d1d"]
        ]
    )
    fig_channels.update_layout(
        margin=dict(l=10, r=20, t=10, b=30),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11, color="#334155"),
        xaxis=dict(
            gridcolor="#f1f5f9",
            tickformat=".0s",
            title=dict(text="Total Views", font=dict(size=11, color="#64748b"))
        ),
        yaxis=dict(
            title=dict(text="Channel", font=dict(size=11, color="#64748b")),
            tickfont=dict(size=10.5)
        ),
        coloraxis_colorbar=dict(
            title=dict(text="count", font=dict(size=11, color="#64748b")),
            thickness=12,
            len=0.75
        )
    )
    st.plotly_chart(fig_channels, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SECTION 2: PUBLISHING TIMING & SEASONALITY
# -----------------------------------------------------------------------------
elif selected_nav == "Publishing Timing & Seasonality":
    st.markdown("""
    <div class="section-header">
        <div class="section-pill"></div>
        <div class="section-title">Temporal Dynamics of Virality & Upload Scheduling</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_t2a, col_t2b = st.columns(2)
    
    with col_t2a:
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="card-title">
                <span style="color: #0284c7;">{ICONS['clock']}</span>
                <span>Publishing Hour (UTC) vs. Average Views</span>
            </div>
        """, unsafe_allow_html=True)
        
        hour_stats = filtered_df.groupby("publish_hour")["views"].mean().reset_index()
        fig_hour = px.line(
            hour_stats,
            x="publish_hour",
            y="views",
            markers=True,
            labels={"publish_hour": "Hour of Day (UTC)", "views": "Average Views"}
        )
        fig_hour.update_traces(
            line=dict(color="#0284c7", width=3),
            marker=dict(size=7, color="#0b2553")
        )
        fig_hour.update_layout(
            margin=dict(l=10, r=20, t=10, b=30),
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=11, color="#334155"),
            xaxis=dict(gridcolor="#f1f5f9", dtick=2),
            yaxis=dict(gridcolor="#f1f5f9", tickformat=".0s")
        )
        st.plotly_chart(fig_hour, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_t2b:
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="card-title">
                <span style="color: #0284c7;">{ICONS['calendar']}</span>
                <span>Publishing Weekday vs. Average Engagement Rate</span>
            </div>
        """, unsafe_allow_html=True)
        
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_stats = filtered_df.groupby("publish_day_name")["engagement_rate"].mean().reindex(day_order).reset_index()
        
        fig_day = px.bar(
            day_stats,
            x="publish_day_name",
            y="engagement_rate",
            labels={"publish_day_name": "Day of Week", "engagement_rate": "Engagement Rate (%)"},
            color="engagement_rate",
            color_continuous_scale=[
                [0.0, "#cffafe"],
                [0.5, "#0891b2"],
                [1.0, "#0e7490"]
            ]
        )
        fig_day.update_layout(
            margin=dict(l=10, r=20, t=10, b=30),
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=11, color="#334155"),
            xaxis=dict(gridcolor="#f1f5f9"),
            yaxis=dict(gridcolor="#f1f5f9"),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_day, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SECTION 3: K-MEANS POPULARITY CLUSTERS
# -----------------------------------------------------------------------------
elif selected_nav == "K-Means Popularity Clusters":
    st.markdown("""
    <div class="section-header">
        <div class="section-pill"></div>
        <div class="section-title">Unsupervised Video Segmentation (K-Means Clustering)</div>
    </div>
    """, unsafe_allow_html=True)
    
    if "cluster_name" in filtered_df.columns:
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="card-title">
                <span style="color: #0284c7;">{ICONS['layers']}</span>
                <span>Interactive Cluster Map: Views vs. Engagement Rate</span>
            </div>
        """, unsafe_allow_html=True)
        
        cluster_color_map = {
            "Moderate Reach / Low Engagement": "#64748b",
            "High Engagement / Enthusiast Viral": "#0d9488",
            "Mega Blockbusters / Mass Reach": "#1d4ed8"
        }
        
        fig_cluster = px.scatter(
            filtered_df.sample(min(3000, len(filtered_df)), random_state=42),
            x="views",
            y="engagement_rate",
            color="cluster_name",
            log_x=True,
            hover_data=["title", "channel_title", "category_name"],
            labels={"views": "Views (Log Scale)", "engagement_rate": "Engagement Rate (%)", "cluster_name": "Cluster"},
            color_discrete_map=cluster_color_map
        )
        fig_cluster.update_traces(marker=dict(size=6, opacity=0.65))
        fig_cluster.update_layout(
            margin=dict(l=10, r=20, t=10, b=30),
            height=440,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=11, color="#334155"),
            xaxis=dict(gridcolor="#f1f5f9"),
            yaxis=dict(gridcolor="#f1f5f9"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.8)"
            )
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="card-title">
                <span style="color: #0284c7;">{ICONS['bar_chart']}</span>
                <span>Cluster Characteristic Summary Profile</span>
            </div>
        """, unsafe_allow_html=True)
        
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
        st.dataframe(
            cluster_grp.style.format({
                "video_count": "{:,}",
                "median_views": "{:,.0f}",
                "avg_likes": "{:,.0f}",
                "avg_engagement": "{:.2f}%"
            }),
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Clustering column not found in dataset. Please run `main.py` first.")

# -----------------------------------------------------------------------------
# SECTION 4: PREDICTIVE ENGAGEMENT ENGINE (DECISION TREE)
# -----------------------------------------------------------------------------
elif selected_nav == "Predictive Engagement Engine":
    st.markdown("""
    <div class="section-header">
        <div class="section-pill"></div>
        <div class="section-title">Interactive Engagement Classifier (Decision Tree)</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="dashboard-card">
        <div class="card-title">
            <span style="color: #0284c7;">{ICONS['target']}</span>
            <span>Predict Engagement Level from Pre-Publication Metadata</span>
        </div>
        <div style="font-size: 13px; color: #64748b; margin-top: -8px; margin-bottom: 20px;">
            Simulate whether a planned video is likely to achieve <b>High Engagement (Above Median)</b> using strictly pre-upload metadata (Zero Data Leakage).
        </div>
    """, unsafe_allow_html=True)
    
    col_pred1, col_pred2 = st.columns(2)
    with col_pred1:
        in_cat = st.selectbox("Video Category:", options=all_categories)
        in_hour = st.slider("Target Upload Hour (UTC):", 0, 23, 12)
        in_day = st.selectbox("Upload Day:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    with col_pred2:
        in_title_len = st.slider("Title Character Length:", 10, 100, 50)
        in_tag_cnt = st.slider("Number of Tags:", 0, 50, 15)
        in_desc_len = st.slider("Description Character Length:", 0, 5000, 500)
    
    predict_clicked = st.button("Predict Engagement Level")
    
    if predict_clicked:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
            
            row_dict = {
                "publish_hour": in_hour,
                "publish_weekday": day_map[in_day],
                "title_length": in_title_len,
                "tag_count": in_tag_cnt,
                "desc_length": in_desc_len
            }
            for cat in all_categories:
                row_dict[f"cat_{cat}"] = 1 if cat == in_cat else 0
            
            model_features = model.feature_names_in_
            feature_vector = pd.DataFrame([[row_dict.get(c, 0) for c in model_features]], columns=model_features)
            
            pred = model.predict(feature_vector)[0]
            prob = model.predict_proba(feature_vector)[0]
            
            if pred == 1:
                st.markdown(f"""
                <div style="background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 12px; padding: 18px 20px; margin-top: 18px;">
                    <div style="font-size: 17px; font-weight: 700; color: #065f46; margin-bottom: 4px;">
                        High Engagement Predicted (Confidence: {prob[1]*100:.1f}%)
                    </div>
                    <div style="font-size: 13.5px; color: #047857;">
                        Videos with this profile historically elicit strong audience interaction (high like and comment ratio relative to view count).
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 12px; padding: 18px 20px; margin-top: 18px;">
                    <div style="font-size: 17px; font-weight: 700; color: #92400e; margin-bottom: 4px;">
                        Standard / Low Engagement Predicted (Confidence: {prob[0]*100:.1f}%)
                    </div>
                    <div style="font-size: 13.5px; color: #b45309;">
                        Recommendation: Consider refining the title length (40-65 chars), adding a more detailed description, or targeting high-engagement categories.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Decision Tree model file not found. Run `python src/classification.py` first.")
            
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SECTION 5: DATA WAREHOUSE & SQL CONSOLE
# -----------------------------------------------------------------------------
elif selected_nav == "Data Warehouse (Star Schema SQL)":
    st.markdown("""
    <div class="section-header">
        <div class="section-pill"></div>
        <div class="section-title">Data Warehousing Component: SQLite Star Schema</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="dashboard-card">
        <div class="card-title">
            <span style="color: #0284c7;">{ICONS['database']}</span>
            <span>Analytical OLAP Query Explorer</span>
        </div>
        <div style="font-size: 13px; color: #64748b; margin-top: -8px; margin-bottom: 16px;">
            Demonstrating analytical queries against the Star Schema Data Warehouse (<code>fact_video_performance</code> joined with dimension tables).
        </div>
    """, unsafe_allow_html=True)
    
    query_options = {
        "Top 10 Most Viewed Videos": """SELECT f.title, c.channel_title, cat.category_name, f.views, f.likes,
       ROUND(f.engagement_rate, 2) AS eng_rate_pct
FROM fact_video_performance f
JOIN dim_channel c ON f.channel_key = c.channel_key
JOIN dim_category cat ON f.category_key = cat.category_key
ORDER BY f.views DESC LIMIT 10;""",
        "Category Views & Engagement Aggregates": """SELECT cat.category_name, COUNT(f.video_key) AS video_count,
       ROUND(AVG(f.views), 0) AS avg_views,
       ROUND(AVG(f.engagement_rate), 2) AS avg_eng_rate_pct
FROM fact_video_performance f
JOIN dim_category cat ON f.category_key = cat.category_key
GROUP BY cat.category_name ORDER BY avg_views DESC;""",
        "Weekday Virality Analysis": """SELECT d.day_name, COUNT(f.video_key) AS total_uploads,
       ROUND(AVG(f.views), 0) AS avg_views,
       ROUND(AVG(f.engagement_rate), 2) AS avg_eng_pct
FROM fact_video_performance f
JOIN dim_date d ON f.date_key = d.date_key
GROUP BY d.day_of_week, d.day_name ORDER BY d.day_of_week;"""
    }
    
    selected_query_title = st.selectbox("Select Analytical SQL Query:", list(query_options.keys()))
    sql_text = query_options[selected_query_title]
    st.code(sql_text, language="sql")
    
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        df_sql = pd.read_sql_query(sql_text, conn)
        conn.close()
        st.dataframe(df_sql, use_container_width=True)
    else:
        st.warning("Database not found. Please run `python src/database.py` to populate warehouse.")
        
    st.markdown("</div>", unsafe_allow_html=True)

# PowerPoint Presentation Slide Deck & Demonstration Guide

**Course:** Data Mining & Data Warehousing (BCI24505D)  
**Topic 42:** YouTube Video Analysis  
**Project Title:** YouTube Video Analysis: Identifying Factors Influencing Video Popularity and Viewer Engagement  
**Format:** 12 Structured Slides with Speaker Script & Demonstration Flow

---

## Slide 1: Title & Introduction
- **Slide Title:** YouTube Video Popularity & Viewer Engagement Mining
- **Subtitle:** Identifying Critical Virality Determinants Using Data Mining & Data Warehousing
- **Student Details:** 3rd Year B.E. Computer Science & Engineering (AI & ML)
- **Course & Code:** Data Mining & Data Warehousing (BCI24505D) - AAT
- **Speaker Notes:**
  > "Good morning, respected faculty and evaluators. Today, I am presenting our Data Mining & Data Warehousing Activity-Based Assessment on Topic 42: YouTube Video Analysis. In this project, we analyze the Kaggle YouTube Trending India dataset to separate the factors that drive massive view counts from those that foster deep viewer engagement."

---

## Slide 2: Problem Statement & Academic Objectives
- **Key Points:**
  - **The Dilemma:** Creators and media brands often confuse sheer view volume with genuine community engagement.
  - **The Risk:** Clickbait and erratic upload schedules fail to yield sustainable audience interaction.
  - **Academic Objectives:**
    1. Preprocess & deduplicate real-world trending data.
    2. Implement unsupervised K-Means clustering to discover audience tiers.
    3. Train a zero-leakage Decision Tree classifier using pre-upload metadata.
    4. Build an analytical Star Schema Data Warehouse in SQLite.
    5. Deploy an interactive executive dashboard in Streamlit.
- **Speaker Notes:**
  > "Our primary research question is: Can we use pre-upload metadata—such as category, upload timing, title length, and description length—to accurately predict viewer engagement before a video goes live, without falling into the trap of data leakage?"

---

## Slide 3: Dataset Ingestion & Preprocessing Pipeline
- **Key Points:**
  - **Source:** Kaggle YouTube Trending Video Statistics (India - `INvideos.csv`).
  - **Raw Scale:** 37,352 daily trending snapshots across 16 attributes.
  - **Deduplication:** Videos trend over multi-day windows; keeping the peak snapshot yielded **16,306 unique creative works** (21,045 duplicate snapshots removed).
  - **Metric Formulation:**
    - Engagement Rate: $((\text{Likes} + \text{Comments}) / \text{Views}) \times 100$ (Median = $0.826\%$)
    - Title, Tag, and Description character lengths extracted.
    - Log transforms ($\log(1+x)$) applied to compress heavy right-tail skewness.
- **Speaker Notes:**
  > "Deduplication was essential. If we hadn't removed repeat daily snapshots, viral videos would have been counted 10 to 30 times, completely distorting correlation metrics and biasing our models."

---

## Slide 4: System Architecture
- **Architecture Diagram:**
  - Raw Data $\rightarrow$ Preprocessing & Feature Engineering $\rightarrow$ 
    - Branch 1: Exploratory Data Analysis (12 Plots)
    - Branch 2: K-Means Clustering ($k=3$)
    - Branch 3: Decision Tree Classifier (Zero-Leakage)
    - Branch 4: SQLite Star Schema ETL (1 Fact + 3 Dimensions)
  - All components feed into the **Streamlit Executive Dashboard**.
- **Speaker Notes:**
  > "Our architecture cleanly bridges Data Mining and Data Warehousing. Cleaned features simultaneously feed unsupervised clustering, supervised classification, and the dimensional Star Schema warehouse."

---

## Slide 5: Exploratory Data Analysis – Views vs. Engagement Paradox
- **Key Points:**
  - **Strong Scale Correlation:** $r(\text{views}, \text{likes}) = +0.85$, $r(\text{views}, \text{comments}) = +0.70$.
  - **The Engagement Paradox:** $r(\text{views}, \text{engagement\_rate}) = -0.05$.
  - **Finding:** Massive view counts do not increase interaction rates. As an audience widens into the millions, passive viewing increases, causing the like-to-view and comment-to-view ratios to drop.
- **Speaker Notes:**
  > "This heatmap illustrates our key empirical finding: While absolute likes scale with views, engagement rate does not. Mass audiences are largely passive consumers, whereas niche audiences actively engage."

---

## Slide 6: Category Dynamics & Publishing Timing
- **Key Points:**
  - **Highest View Volume:** Movies ($3.86\text{M}$ avg) and Music ($2.02\text{M}$ avg).
  - **Highest Engagement Rate:** Pets & Animals ($8.34\%$), Gaming ($4.90\%$), and Science & Technology ($3.82\%$).
  - **Optimal Upload Window:** 04:00 - 11:00 UTC ($09:30 - 16:30$ IST).
  - **Seasonality:** Weekend uploads show higher engagement rates ($\approx 2.05\%$) than mid-week uploads.
- **Speaker Notes:**
  > "Categories with enthusiastic fanbases like Gaming and Tech generate more than double the engagement rate of general entertainment, even with smaller view counts."

---

## Slide 7: Algorithm 1 – K-Means Clustering (Unsupervised Mining)
- **Key Points:**
  - **Features:** Scaled $[\log(\text{views}), \log(\text{likes}), \log(\text{comments}), \text{engagement\_rate}]$.
  - **Optimization:** Elbow Curve evaluated for $k \in [2, 7]$; validated at $k=3$ with peak Silhouette score ($0.3879$).
  - **Cluster Profiles:**
    - **Cluster 0 (47.8%):** *Moderate Reach / Low Engagement* (Median views $\approx 85\text{k}$, Eng Rate $\approx 0.52\%$).
    - **Cluster 1 (38.6%):** *High Engagement / Community Viral* (Median views $\approx 215\text{k}$, Eng Rate $\approx 3.48\%$).
    - **Cluster 2 (13.6%):** *Mega Blockbusters* (Median views $\approx 1.42\text{M}$, Eng Rate $\approx 1.85\%$).
- **Speaker Notes:**
  > "Rather than an arbitrary split, K-Means naturally divides the dataset into three distinct behavioral segments, confirming that high view count and high engagement represent separate clusters."

---

## Slide 8: Algorithm 2 – Decision Tree Classification (Supervised Mining)
- **Key Points:**
  - **Objective:** Predict if a video will achieve High Engagement ($\ge$ median $0.826\%$).
  - **Zero Data Leakage:** Excluded all post-upload response metrics. Used only pre-upload metadata: Category, Upload Hour, Weekday, Title Length, Tag Count, Description Length.
  - **Hyperparameters:** Depth pruned to 4; minimum leaf samples set to 25.
  - **Validation Metrics (Test Set $n=3,262$):**
    - **Accuracy:** **71.67%**
    - **Precision:** **75.78%**
    - **Recall:** **63.70%**
    - **F1-Score:** **69.22%**
- **Speaker Notes:**
  > "We achieved 71.7% accuracy without data leakage. If we had included likes or views, the model would have had 99% accuracy but zero practical utility for a creator planning a video."

---

## Slide 9: Feature Importance & Tree Interpretation
- **Key Points:**
  - **Top Feature Drivers:**
    1. Content Category ($\approx 42.1\%$ importance)
    2. Description Length ($\approx 24.3\%$ importance)
    3. Title Length ($\approx 15.6\%$ importance)
    4. Publishing Hour ($\approx 10.8\%$ importance)
  - **Actionable Takeaways:**
    - Informative descriptions provide context that drives comments and likes.
    - Titles between 40-65 characters out-engage overly long titles.
- **Speaker Notes:**
  > "Category selection sets the foundation for engagement style, while description depth and title length represent the most impactful levers creators control directly."

---

## Slide 10: Data Warehousing – Star Schema Implementation
- **Key Points:**
  - Built an authentic Star Schema in **SQLite 3**:
    - Central Fact: `fact_video_performance` (16,306 rows)
    - Dimensions: `dim_category` (17 rows), `dim_channel` (1,421 rows), `dim_date` (219 rows)
  - Indexes created on foreign keys and metric measures.
  - Executed 5 OLAP analytical queries (Top videos, category rollups, channel benchmarking, monthly trends, engagement outliers).
- **Speaker Notes:**
  > "Our dimensional Star Schema simplifies reporting by decoupling dimensions like Channel and Date from the numeric measures in the Fact table, allowing rapid analytical slicing and dicing."

---

## Slide 11: Streamlit Dashboard Demonstration
- **Key Points:**
  - **Live URL / Port:** Localhost via `streamlit run dashboard/app.py`.
  - **Interactive Features:**
    - Dynamic Category & Channel multi-select filters.
    - Plotly interactive charts for category performance.
    - 2D interactive K-Means cluster scatter plot.
    - **What-If Predictive Simulator:** Allows users to input category, timing, and title length to get real-time Decision Tree engagement predictions.
    - In-browser SQL query execution console.
- **Speaker Notes:**
  > "Now, let me transition to demonstrating the live interactive dashboard built in Streamlit..."

---

## Slide 12: Conclusion & Future Scope
- **Key Points:**
  - **Conclusion:** All five mandatory AAT criteria successfully implemented and validated on authentic real-world data.
  - **Core Lesson:** Video popularity and viewer engagement are distinct dimensions requiring tailored content strategies.
  - **Future Enhancements:**
    - Computer Vision embeddings of thumbnails.
    - NLP sentiment analysis of comment threads.
    - Time-series tracking of virality velocity curves.
- **Speaker Notes:**
  > "Thank you for your time. I am now ready for questions and viva voce."

---

## Live Demonstration Sequence for Evaluators

1. **Step 1 (Source Code & Folder Structure):** Open the project in VS Code, showing `src/`, `data/`, `sql/`, `outputs/figures/`, and `notebooks/`.
2. **Step 2 (Data Pipeline Execution):** Run `python main.py` in the terminal to show all 7 pipeline steps executing smoothly in under 15 seconds.
3. **Step 3 (Notebook Walkthrough):** Open `notebooks/YouTube_Video_Analysis.ipynb`, highlighting markdown explanations, preprocessing outputs, and inline plots.
4. **Step 4 (SQL Data Warehouse):** Open `sql/schema.sql` and run a sample query from `sql/analysis_queries.sql` against `data/youtube_warehouse.db`.
5. **Step 5 (Interactive Streamlit Dashboard):** Run `streamlit run dashboard/app.py` in terminal:
   - Filter by category (e.g. Comedy vs News).
   - Show the K-Means cluster tab.
   - Use the **What-If Predictive Engagement Engine** to test an upcoming video idea live!

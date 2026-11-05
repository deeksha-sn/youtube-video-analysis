# Empirical Results and In-Depth Interpretation

**Course:** Data Mining & Data Warehousing (BCI24505D)  
**Topic 42:** YouTube Video Analysis  
**Dataset:** Real-world YouTube Trending Statistics (India Region - `INvideos.csv`)

---

## 1. Summary of Dataset Preprocessing Pipeline

The preprocessing and deduplication pipeline executed on the raw dataset yielded the following audited metrics:

| Processing Stage | Quantitative Value | Academic Rationale |
| :--- | :--- | :--- |
| **Initial Raw Records** | 37,352 | Unfiltered daily snapshots of trending entries |
| **Raw Feature Dimensions** | 16 | Original Kaggle schema attributes |
| **Missing Descriptions Imputed** | 561 | Empty string replacement preserves video identity |
| **Duplicate Snapshots Removed** | 21,045 | Videos trend over multi-day spans; deduplication avoids multi-counting |
| **Invalid / Zero-View Filtered** | 1 | Ensures denominator $> 0$ for ratio computations |
| **Final Cleaned Unique Videos** | **16,306** | Cross-sectional sample of distinct creative works |
| **Median Engagement Threshold** | **0.826%** | Serves as objective boundary for High vs Normal engagement target |

### Outlier Detection Analysis (Interquartile Range - IQR)
Because online virality follows a heavy-tailed power-law distribution, extreme values are natural phenomena rather than recording errors. The table below documents the calculated IQR boundaries:

| Metric | Q1 (25th %) | Median (50th %) | Q3 (75th %) | IQR | Upper Bound (Q3 + 1.5×IQR) | Outliers Count (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Views** | 43,846 | 129,566 | 451,215 | 407,369 | 1,062,268 | 2,058 (12.62%) |
| **Likes** | 465 | 1,482 | 6,368 | 5,903 | 15,222 | 2,126 (13.04%) |
| **Comments** | 41 | 148 | 612 | 571 | 1,468 | 2,130 (13.06%) |
| **Engagement Rate (%)** | 0.41% | 0.83% | 2.05% | 1.64% | 4.51% | 2,028 (12.44%) |

---

## 2. Exploratory Findings & Empirical Observations

### 2.1 Viewing Volume vs. Engagement Dichotomy
A central empirical finding is the negative to neutral Pearson correlation between view count and engagement rate:
$$r(\text{views}, \text{likes}) = +0.85 \quad \text{vs} \quad r(\text{views}, \text{engagement\_rate}) = -0.05$$

**Interpretation:** While raw likes and comments scale proportionally with total view count due to sheer audience size, the proportion of viewers who actively engage (likes + comments divided by views) actually diminishes or plateaus for mega-viral blockbusters. Mass audiences consist predominantly of passive viewers, whereas smaller, niche audiences (in gaming, science, or comedy) exhibit deep community participation.

### 2.2 Category Disparities
1. **Highest View Volume:** **Entertainment**, **Music**, and **Film & Animation** command over 78% of all cumulative views in the Indian dataset.
2. **Highest Engagement Rates:** **Pets & Animals** (8.34%), **Gaming** (4.90%), and **Science & Technology** (3.82%) generate significantly higher engagement rates than mass entertainment (1.98%).
3. **Leading Channels:** **T-Series** (492M+ views across 62 trending videos), **5-Minute Crafts** (207M+ views), and **Marvel Entertainment** (199M+ views) lead aggregate reach.

### 2.3 Timing Dynamics
- **Peak Upload Hour:** 18:00 UTC (23:30 IST) represents the peak trending view volume hour, while uploads between 04:00 - 11:00 UTC (09:30 - 16:30 IST) gather steady accumulation prior to prime-time evening consumption.
- **Weekday Virality:** Friday and Saturday releases exhibit higher engagement rates (1.95% - 2.10%) compared to mid-week uploads, consistent with leisure browsing schedules.

---

## 3. Data Mining Algorithm 1: K-Means Clustering

### Cluster Optimization: Elbow Method & Silhouette Analysis
The Elbow Curve was computed for $k \in [2, 7]$ on scaled features:
$$X = [\log(1+\text{views}), \log(1+\text{likes}), \log(1+\text{comments}), \text{engagement\_rate}]$$

- $k=2$: Inertia = 39,650.5 | Silhouette = 0.3707
- **$k=3$:** **Inertia = 30,252.6 | Silhouette = 0.3879 (Optimal Cohesion)**
- $k=4$: Inertia = 23,514.5 | Silhouette = 0.3438
- $k=5$: Inertia = 20,323.6 | Silhouette = 0.3133

The elbow inflection point and peak silhouette score quantitatively validate $k=3$ as the optimal partition.

### Resulting Cluster Characteristics

| Cluster ID | Semantic Label | Video Share (%) | Median Views | Median Likes | Avg Engagement Rate (%) | Dominant Genre |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cluster 0** | **Moderate Reach / Low Engagement** | 47.8% | 85,420 | 620 | 0.52% | News & Politics / Entertainment |
| **Cluster 1** | **High Engagement / Enthusiast Viral** | 38.6% | 215,800 | 4,890 | 3.48% | Comedy / Science & Tech |
| **Cluster 2** | **Mega Blockbusters / Mass Reach** | 13.6% | 1,420,500 | 48,200 | 1.85% | Music / Film Trailers |

**Insight:** YouTube content does not follow a simple linear popularity curve. Rather, it partitions into three functional strata: routine baseline content (Cluster 0), dedicated community-driven viral content (Cluster 1), and commercial mass-market tentpoles (Cluster 2).

---

## 4. Data Mining Algorithm 2: Decision Tree Classification

### Objective & Zero Data Leakage Enforcement
To predict whether an upcoming video will achieve **High Engagement ($y=1$)** or **Normal Engagement ($y=0$)** relative to the median threshold (0.826%), only metadata known prior to publication was supplied:
- Predictor Matrix: `publish_hour`, `publish_weekday`, `title_length`, `tag_count`, `desc_length`, and One-Hot Encoded `category_name` dummies ($D = 21$ features).
- Sample Partition: Stratified 80% Train ($n=13,044$) and 20% Test ($n=3,262$).

### Model Performance Metrics

| Evaluation Metric | Realized Value | Academic Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **71.67%** | Substantially outperforms a random baseline (50%) |
| **Precision** | **75.78%** | When predicting High Engagement, the model is correct $\approx 76\%$ of the time |
| **Recall** | **63.70%** | Captures nearly two-thirds of all genuinely high-engagement videos |
| **F1-Score** | **69.22%** | Balanced harmonic mean indicating reliable predictive generalization |

### Confusion Matrix on Test Set ($n=3,262$)

| | Predicted Normal ($y=0$) | Predicted High ($y=1$) |
| :--- | :--- | :--- |
| **Actual Normal ($y=0$)** | **1,299 (TN)** | **332 (FP)** |
| **Actual High ($y=1$)** | **592 (FN)** | **1,039 (TP)** |

### Top Metadata Feature Importances (Gini Impurity Reduction)
1. **Category (Entertainment / Music / News):** $\approx 42.1\%$ total relative importance. Choosing content genre creates the strongest structural baseline for audience interaction style.
2. **Description Length (`desc_length`):** $\approx 24.3\%$ importance. Videos with comprehensive, informative descriptions provide context and links that stimulate discussion.
3. **Title Length (`title_length`):** $\approx 15.6\%$ importance. Concise, provocative titles out-engage overly long, keyword-stuffed titles.
4. **Publish Hour (`publish_hour`):** $\approx 10.8\%$ importance. Timing release to catch peak active audience windows.
5. **Tag Count (`tag_count`):** $\approx 7.2\%$ importance.

**Honest Modeling Discussion:** Achieving $\approx 72\%$ accuracy purely from pre-upload metadata is a realistic and honest result. Video engagement is ultimately driven by content quality, storytelling, and creator charisma—factors that cannot be captured by text character counts or category labels alone. Any model claiming 95%+ accuracy using metadata alone would suffer from severe data leakage.

---

## 5. Formal Synthesis of the 8 Research Questions

### RQ1: Which factors are associated with higher video views?
**Answer:** Sheer view count correlates most strongly with Likes ($r = 0.85$) and Comments ($r = 0.70$). However, views exhibit a near-zero or slightly negative correlation with Engagement Rate ($r = -0.05$). High viewership is driven by broad topical appeal, established channel subscriber bases, and algorithmic recommendation push, rather than high interaction rates per viewer.

### RQ2: Which categories receive higher average views?
**Answer:** **Entertainment**, **Music**, and **Film & Animation** command the highest average views, led by movie teasers and major label music videos.

### RQ3: Does a higher number of likes indicate higher engagement?
**Answer:** No. Absolute likes reflect audience scale, not engagement depth. A video with 10M views and 100k likes has an engagement rate of only 1.0%, whereas a tutorial with 50k views and 5k likes has a 10.0% engagement rate. Engagement must always be normalized against viewership.

### RQ4: Which publishing hours show higher average views?
**Answer:** Uploads between 04:00 and 11:00 UTC (09:30 - 16:30 IST) gather the greatest cumulative views by allowing YouTube's recommendation index to index and distribute the video ahead of evening peak leisure hours.

### RQ5: What patterns can be observed in video comments?
**Answer:** Commenting represents a significantly higher friction action than liking. On average, videos receive 5 to 10 times more likes than comments. Videos that stimulate questions, controversies, or giveaways show disproportionately high comment-to-view ratios.

### RQ6: What are the characteristics of high-engagement videos?
**Answer:** High-engagement videos typically feature:
- Niche, enthusiast communities (Gaming, Tech, Comedy, Education).
- Optimized title lengths (40-65 characters).
- Well-structured descriptions with calls to action.
- Content formats that encourage active debate or viewer participation.

### RQ7: Can videos be grouped based on popularity?
**Answer:** Yes. Unsupervised K-Means clustering ($k=3$) successfully segments videos into:
1. *Moderate Reach / Baseline Content* (47.8%)
2. *High Engagement / Community Viral* (38.6%)
3. *Mega Blockbusters / Mass Reach* (13.6%)

### RQ8: Which metadata features help classify engagement levels?
**Answer:** Content Category is the single strongest pre-upload predictor, followed by Description Length and Title Length. Upload timing provides secondary optimization.

---

## 6. Correlation vs. Causation Warning
While strong correlations exist between certain metadata features and engagement, these associations do **not** prove direct causation:
- Writing a longer description or changing an upload hour will not automatically turn mediocre content into a viral hit.
- Correlation indicates that creators who invest time in comprehensive metadata descriptions also tend to invest higher effort into video production, narrative quality, and audience retention.

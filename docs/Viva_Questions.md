# Academic Viva Voce Preparation Guide (25 Questions & Answers)

**Course:** Data Mining & Data Warehousing (BCI24505D)  
**Topic 42:** YouTube Video Popularity & Engagement Analysis  
**Degree:** B.E. Computer Science & Engineering (AI & ML)

---

## Part I: Data Mining & Machine Learning Concepts

### Q1: What is K-Means clustering, and how does it work?
**Answer:**
K-Means is an unsupervised partitioning algorithm that divides $n$ data points into $k$ non-overlapping clusters. It works iteratively:
1. Randomly initializes $k$ centroids (using the K-Means++ technique for smarter initial placement).
2. Assigns each data point to its closest centroid based on Euclidean distance.
3. Recomputes each centroid as the mathematical mean of all points assigned to that cluster.
4. Repeats steps 2 and 3 until centroids stabilize (convergence) or the maximum iterations are reached.

### Q2: Why did you apply log transformations before running K-Means?
**Answer:**
YouTube metrics like views, likes, and comments follow extreme right-skewed power-law distributions—a few viral videos have tens of millions of views, while the vast majority have a few thousand. Because K-Means uses standard Euclidean distance, raw view counts would completely dominate the distance metric, making other features like engagement rate irrelevant. Applying $\log(1 + x)$ compresses the numerical scale across orders of magnitude while preserving relative order, allowing all features to contribute fairly.

### Q3: How did you select the optimal number of clusters ($k$)?
**Answer:**
We used a dual validation approach:
1. **The Elbow Method:** We plotted the Within-Cluster Sum of Squares (Inertia) against $k \in [2, 7]$. The point where the rate of inertia reduction distinctly slows down indicates the "elbow" ($k=3$).
2. **Silhouette Coefficient:** We computed the average silhouette score, which measures how cohesive points are within their own cluster compared to neighboring clusters. $k=3$ achieved the highest silhouette score ($0.3879$).

### Q4: What is the Silhouette Score, and what do its values signify?
**Answer:**
The Silhouette Score ranges from $-1$ to $+1$:
- Close to $+1$: Indicates well-separated, highly cohesive clusters.
- Near $0$: Indicates overlapping clusters on decision boundaries.
- Negative (towards $-1$): Indicates points have been assigned to the wrong cluster.

### Q5: How does a Decision Tree Classifier make decisions?
**Answer:**
A Decision Tree recursively splits the dataset into subsets using the feature and cutoff value that produces the greatest purity in target classes. At each split, it evaluates the **Gini Impurity**:
$$I_G = 1 - \sum p_i^2$$
A Gini of $0$ represents a completely pure node (all samples belong to the same class). The algorithm selects splits that minimize child Gini impurity (maximizing Information Gain).

### Q6: What is "Data Leakage," and how did you prevent it in your Decision Tree?
**Answer:**
Data leakage occurs when information from the target variable or post-event data is inadvertently included in the training features, leading to unrealistically high test accuracy that fails in production.
In our project, our target was `high_engagement` (derived from likes, comments, and views). If we had included views, likes, or comments as input features, the model would have had direct access to the target's components. We prevented data leakage by strictly using **pre-publication metadata** available *before* the video goes live: `publish_hour`, `publish_weekday`, `title_length`, `tag_count`, `desc_length`, and `category_name`.

### Q7: Why is your Decision Tree accuracy $\approx 72\%$ rather than $99\%$?
**Answer:**
An accuracy of $71.67\%$ with balanced classes is realistic and methodologically sound. A video's ultimate engagement depends heavily on subjective human factors: content quality, audio-video editing, thumbnail appeal, and creator personality—none of which are captured purely by title length or category labels. Models reporting $99\%$ accuracy on this task almost certainly suffer from data leakage.

### Q8: Explain the difference between Precision and Recall in this project.
**Answer:**
- **Precision (75.78%):** Out of all videos our model predicted as "High Engagement," $75.78\%$ actually achieved high engagement. It measures the reliability of positive predictions.
- **Recall (63.70%):** Out of all truly "High Engagement" videos in the dataset, our model successfully identified $63.70\%$. It measures how thoroughly the model catches target instances.
- **F1-Score (69.22%):** The harmonic mean of precision and recall, balancing false positives and false negatives.

### Q9: What is overfitting in Decision Trees, and how did you mitigate it?
**Answer:**
An unconstrained decision tree can grow until every training point is in its own leaf, capturing noise and memorizing the training set. We controlled overfitting by:
1. Setting `max_depth = 4` to limit hierarchical complexity.
2. Setting `min_samples_leaf = 25` to require statistical significance before creating a split.

---

## Part II: Data Warehousing & SQL Concepts

### Q10: What is a Data Warehouse, and how does it differ from an OLTP database?
**Answer:**
- **OLTP (Online Transaction Processing):** Optimized for high-frequency, low-latency insert/update transactions in normalized schemas (3NF) to minimize redundancy (e.g., recording individual video clicks).
- **Data Warehouse (OLAP - Online Analytical Processing):** Optimized for complex, read-heavy analytical aggregations across historical data, typically organized in multidimensional schemas (Star or Snowflake) with pre-aggregated measures.

### Q11: Describe the Star Schema implemented in your project.
**Answer:**
Our Star Schema consists of:
- **One Fact Table (`fact_video_performance`):** Stores numeric additive and semi-additive business measures (`views`, `likes`, `dislikes`, `comment_count`, `engagement_rate`) along with integer foreign keys referencing dimension tables.
- **Three Dimension Tables:**
  - `dim_category`: Stores category IDs and names.
  - `dim_channel`: Stores distinct channel publishers.
  - `dim_date`: Stores temporal attributes (Year, Month, Day, Day-of-Week, Weekend flag).

### Q12: Why is a Star Schema preferred over a 3NF normalized schema for analytics?
**Answer:**
1. **Fewer Joins:** Dimension tables are denormalized, meaning an analytical query requires only 1-3 simple joins between the fact table and dimensions, compared to 8-10 joins in 3NF.
2. **Intuitive for Business Reporting:** Queries map directly to business dimensions (e.g., views by channel, by month, by category).
3. **Optimized Aggregations:** Indexing on surrogate foreign keys accelerates `GROUP BY` and `ORDER BY` operations.

### Q13: What is a Surrogate Key, and why did you use one?
**Answer:**
A surrogate key is a system-generated integer identifier (such as `video_key` or `channel_key` with `AUTOINCREMENT`) that has no intrinsic business meaning. It decouples the warehouse from operational system key changes, provides uniform indexing, and saves storage space over text-based natural keys.

### Q14: Explain the difference between additive, semi-additive, and non-additive facts.
**Answer:**
- **Additive Facts:** Can be summed across all dimensions (e.g., `views`, `likes`, `comment_count`).
- **Semi-Additive Facts:** Can be summed across some dimensions but not others (e.g., bank account balances across branches, but not across dates).
- **Non-Additive Facts:** Cannot be summed across any dimension (e.g., ratios and percentages like `engagement_rate`). These must be recalculated via division of sums: $(\sum \text{likes} + \sum \text{comments}) / \sum \text{views}$.

---

## Part III: Data Preprocessing & Statistics

### Q15: Why did you deduplicate the raw dataset by `video_id`?
**Answer:**
In YouTube Trending data, popular videos trend for multiple days in a row (appearing as separate daily snapshots). If analyzed without deduplication, viral videos would be counted 10 to 30 times, severely biasing averages, inflating view distributions, and creating artificial correlation. We kept the latest, highest-view snapshot per video to obtain an accurate cross-sectional view of unique creative works.

### Q16: How did you calculate Engagement Rate?
**Answer:**
$$\text{Engagement Rate (\%)} = \left( \frac{\text{Likes} + \text{Comment Count}}{\text{Views}} \right) \times 100$$
We ensured views were strictly greater than zero to prevent division-by-zero runtime exceptions, and clipped values at $100\%$ to handle rare live-stream reporting latency artifacts.

### Q17: What is the Interquartile Range (IQR) method for outlier detection?
**Answer:**
IQR is a robust non-parametric measure of statistical dispersion:
$$\text{IQR} = Q_3 - Q_1$$
- Lower Bound: $Q_1 - 1.5 \times \text{IQR}$
- Upper Bound: $Q_3 + 1.5 \times \text{IQR}$
Any point outside these bounds is an outlier. Because online views follow heavy tails, we documented outliers but did not blindly drop them, as genuine viral hits represent legitimate consumer phenomena.

### Q18: What is the difference between correlation and causation?
**Answer:**
Correlation measures statistical association between two variables (e.g., $r(\text{views}, \text{likes}) = 0.85$), meaning they tend to increase together. Causation means a change in one variable directly causes a change in the other.
*Example:* A high tag count correlates modestly with higher views, but simply adding 50 random tags will not cause an uninteresting video to go viral.

---

## Part IV: Python & Technology Architecture

### Q19: Why use Pandas DataFrames instead of native Python lists/dictionaries?
**Answer:**
Pandas is built on NumPy's C-level contiguous memory arrays. Vectorized column operations (e.g., `df['likes'] / df['views']`) run orders of magnitude faster than Python `for` loops, and provide optimized handling of missing values, filtering, grouping, and indexing.

### Q20: What is the purpose of `StandardScaler` in Scikit-Learn?
**Answer:**
`StandardScaler` transforms features to have a mean of $0$ and a standard deviation of $1$ ($z = (x - \mu)/\sigma$). This ensures features with large numerical magnitudes (e.g., view counts in millions) do not disproportionately dominate distance-based algorithms like K-Means over smaller features like engagement rate ($0-10\%$).

### Q21: What role does SQLite play in this project?
**Answer:**
SQLite provides a zero-configuration, self-contained, serverless relational database engine. It allows us to build and query an authentic Star Schema Data Warehouse locally within the project without requiring heavy database server installations (like PostgreSQL or Oracle), while supporting standard ANSI SQL.

### Q22: What does Streamlit do in this architecture?
**Answer:**
Streamlit turns our Python analytical scripts into an interactive web dashboard. It allows non-technical stakeholders to explore category trends, adjust engagement thresholds, inspect K-Means clusters interactively via Plotly, test real-time Decision Tree predictions, and execute SQL queries against the Star Schema.

### Q23: Why did you set `random_state=42` across train-test splits and models?
**Answer:**
Setting a fixed pseudo-random number generator seed (`random_state=42`) ensures computational reproducibility. Every time the code is executed by faculty or reviewers, it produces the exact same train/test splits, centroid initializations, and evaluation metrics.

### Q24: What is the significance of the Confusion Matrix?
**Answer:**
A confusion matrix visualizes classifier performance across four quadrants: True Negatives (TN), False Positives (FP), False Negatives (FN), and True Positives (TP). It exposes whether a model has asymmetric error tendencies (e.g., being overly optimistic or conservative in predicting High Engagement).

### Q25: If you had 6 more months on this project, what would you enhance?
**Answer:**
1. **Multimodal Analysis:** Extract thumbnail visual embeddings using Convolutional Neural Networks (ResNet/CLIP) to evaluate visual clickability.
2. **NLP on Titles & Comments:** Perform Transformer-based sentiment analysis (BERT/RoBERTa) on top user comments to measure positive vs negative sentiment ratios.
3. **Time-Series Tracking:** Transition from static snapshot deduplication to longitudinal panel data to model video virality half-life and decay curves over time.

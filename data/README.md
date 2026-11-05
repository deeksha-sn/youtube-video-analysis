# Dataset Documentation: YouTube Trending Video Statistics

## 1. Overview and Source
This project utilizes the real-world **YouTube Trending Video Statistics Dataset** (sourced from Kaggle: [datasnaek/youtube-new](https://www.kaggle.com/datasets/datasnaek/youtube-new)). The dataset includes daily records for top trending YouTube videos across various regions, specifically focusing on the India region (`INvideos.csv`) alongside the category mapping metadata (`IN_category_id.json`).

## 2. Directory Structure
```
data/
├── raw/
│   ├── INvideos.csv            # Raw dataset containing Indian trending video metadata
│   └── IN_category_id.json     # JSON mapping category IDs to human-readable category titles
└── processed/
    └── cleaned_youtube_data.csv # Preprocessed, cleaned, and feature-engineered dataset
```

## 3. Dataset Attributes
| Attribute Name | Data Type | Description |
| :--- | :--- | :--- |
| `video_id` | String | Unique identifier for each YouTube video |
| `trending_date` | String | Date on which the video trended (format: `yy.dd.mm`) |
| `title` | String | Title of the video |
| `channel_title` | String | Title / name of the YouTube channel |
| `category_id` | Integer | ID mapping to category genre (e.g., Entertainment, Music) |
| `publish_time` | ISO Datetime | Timestamp when the video was published |
| `tags` | String | Video tags separated by pipe (`\|`) characters |
| `views` | Integer | Cumulative view count at the trending timestamp |
| `likes` | Integer | Number of likes received |
| `dislikes` | Integer | Number of dislikes received |
| `comment_count` | Integer | Total count of comments |
| `thumbnail_link` | String | URL to video thumbnail image |
| `comments_disabled` | Boolean | Flag indicating whether comments were turned off |
| `ratings_disabled` | Boolean | Flag indicating whether likes/dislikes were turned off |
| `video_error_or_removed` | Boolean | Flag indicating if video had errors or was removed |
| `description` | String | Video description text |

## 4. Derived & Engineered Attributes
During data preprocessing, the following attributes are generated:
- `category_name`: Clean text name mapped from `IN_category_id.json`.
- `publish_year`, `publish_month`, `publish_day`, `publish_hour`, `publish_weekday`: Temporal decomposition from `publish_time`.
- `engagement_rate`: Calculated as `((likes + comments) / views) * 100`.
- `like_ratio`: Calculated as `(likes / views) * 100`.
- `comment_ratio`: Calculated as `(comments / views) * 100`.
- `title_length`: Character length of the video title.
- `tag_count`: Number of individual tags assigned.
- `desc_length`: Character length of the description.
- `high_engagement`: Binary target variable (1 if engagement_rate >= median engagement rate, else 0).

## 5. Dataset Limitations
1. **Selection Bias**: The dataset comprises exclusively videos that reached YouTube's "Trending" section. Consequently, it represents viral or algorithmically favored content rather than the global distribution of all YouTube uploads.
2. **Snapshot Nature**: Metrics like views, likes, and comments reflect snapshots on specific trending days rather than lifetime totals.
3. **Regional Dynamics**: The Indian market exhibits distinct language, music, film, and entertainment consumption habits that may differ from Western or East Asian demographics.

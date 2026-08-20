# SentimentAnalysis

Sentiment classification of IMDB movie reviews.

## Data

`data/IMDB-movie-reviews.csv` — a 100-row sample of the IMDB movie review dataset.

| Property | Value |
| --- | --- |
| Rows | 100 |
| Delimiter | `;` (semicolon) |
| Columns | `review`, `sentiment` |
| Labels | `negative` (58), `positive` (42) |

Review text contains raw HTML line breaks (`<br />`) that should be stripped during preprocessing.

Loading it:

```python
import pandas as pd

df = pd.read_csv("data/IMDB-movie-reviews.csv", sep=";")
df["review"] = df["review"].str.replace(r"<br\s*/?>", " ", regex=True)
```

## Status

Early stage — the dataset is in place, modelling code is not written yet.

## Layout

```
.
├── data/
│   └── IMDB-movie-reviews.csv
└── README.md
```

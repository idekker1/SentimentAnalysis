# SentimentAnalysis

Compares four sentiment models — DistilBERT SST-2, VADER, GPT-2 and T5 — on the same
100 IMDB movie reviews, and scores them against the labels that came with the data.

The dataset is in `data/IMDB-movie-reviews.csv`: 100 rows, semicolon-delimited, a
`review` column and a `sentiment` column, split 58 negative / 42 positive.

## Setup

```bash
pip install -r requirements.txt
```

Use Python 3.11 — the pinned `torch==2.3.1` has no wheels for 3.13 and up. The first
run downloads the model weights, which takes a while; after that they are cached.

## Usage

Run the three steps in order: `check_data()` reads the CSV and reports what is in it,
`format_data()` cleans the reviews and normalises the labels, `predict()` scores them.

```python
from src.sentiment_analyzer import SentimentAnalyzer, SentimentModel

analyzer = SentimentAnalyzer("data/IMDB-movie-reviews.csv")
analyzer.check_data()
analyzer.format_data()

results = analyzer.predict(SentimentModel.T5)
```

`predict()` returns a DataFrame with `Review`, `y_target`, `y_pred` and `confidence`.
Pass `export_csv=True` to also write it to `notebook_outputs/`.

To score the result:

```python
from src.results_analyzer import ResultsAnalyzer

ResultsAnalyzer.calculate_accuracy(results["y_target"], results["y_pred"])
```

`ResultsAnalyzer` also has `plot_confusion_matrix()` for one model and
`compare_models()` for several at once.

## Results

Accuracy over all 100 reviews:

| Model | Accuracy |
| --- | --- |
| T5 | 0.95 |
| DistilBERT SST-2 | 0.89 |
| GPT-2 | 0.78 |
| VADER | 0.64 |

The per-review predictions behind these numbers are in `notebook_outputs/`.

## Tests

```bash
pytest
```

## Layout

```
data/              the review CSV
notebooks/         the original exploration this code came from
notebook_outputs/  predictions, one CSV per model
src/               SentimentAnalyzer and ResultsAnalyzer
tests/             the test suite
```

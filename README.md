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

`main.py` runs the whole comparison end to end — all four models over the dataset,
scored, with everything written to `outputs/`:

```bash
python main.py
```

It writes one predictions CSV and one confusion matrix per model, plus
`model_comparison.csv` and `model_comparison.png` across all of them. Point it at
another file or narrow it to the models you want:

```bash
python main.py --data data/IMDB-movie-reviews.csv --models vader t5 --output-dir outputs
```

An input with no `sentiment` column still gets predictions; the scoring step is
skipped, as there is nothing to score against.

### From Python

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

The per-review predictions behind these numbers are in `notebook_outputs/`, which the
notebook produced from the raw review text.

`main.py` scores slightly differently — DistilBERT 0.88 and GPT-2 0.76, VADER and T5
unchanged — because `format_data()` strips the `<br />` tags that 68 of the 100 reviews
carry and the notebook left in. Every prediction that differs is on a review whose text
that cleanup changed; nothing else moves between the two.

### As an API

`src/api.py` puts a single prediction behind HTTP. Start the server with:

```bash
uvicorn src.api:app --reload
```

`GET /health` answers `{"status": "ok"}` without loading anything, so it stays cheap
to poll. `POST /predict` scores one review:

```bash
curl -X POST localhost:8000/predict \
     -H 'Content-Type: application/json' \
     -d '{"text": "A brilliant, moving film.", "model": "vader"}'
```

```json
{"text": "A brilliant, moving film.", "model": "vader", "sentiment": "positive", "confidence": 0.5859}
```

`model` is optional and defaults to `distilbert`; `text` comes back as the model saw
it, cleaned the same way `format_data()` cleans a CSV column.

An unknown `model` or a missing `text` comes back as a `422` from FastAPI's own
validation. Nothing else is checked: whatever `text` holds goes to the model, which
will label it either way. Interactive docs are at `/docs`.

Both handlers are plain `def` rather than `async def`: FastAPI runs a sync handler in
its threadpool, so a request that spends seconds inside a model does not block the
event loop for everyone else. Each request loads its model — VADER answers
immediately, the transformers pay a load every time and a download the first time.

## Tests

```bash
pytest
```

## Layout

```
main.py            the runner: dataset in, analysis out
data/              the review CSV
notebooks/         the original exploration this code came from
notebook_outputs/  predictions, one CSV per model
outputs/           where main.py writes its results (git-ignored)
src/               SentimentAnalyzer, ResultsAnalyzer and the API
tests/             the test suite
```

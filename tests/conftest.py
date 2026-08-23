"""Shared fixtures for the test suite.

pytest imports this file automatically before collecting tests; anything defined
here is available to every test module in ``tests/`` without being imported.
"""

from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

# Select a non-interactive backend before any test imports pyplot, so figures are
# rendered into memory instead of trying to open a window on a machine that has
# no display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

from src.sentiment_analyzer import SentimentAnalyzer, SentimentModel

# Review text for the fixture CSVs. Two constraints come from the file format
# SentimentAnalyzer assumes: no semicolons (the delimiter) and nothing outside
# cp1252. The HTML breaks and padding are deliberate — format_data is supposed to
# strip them, so they need to be here to prove it does.
#
# The wording is otherwise unimportant: no test asserts what any model predicts
# for these reviews, only that it returns something well formed.
LABELLED_ROWS: list[tuple[str, str]] = [
    ("A wonderful little production.<br /><br />The filming technique is charming.", "positive"),
    ("   Basically a family where a little boy thinks there is a zombie in his closet.   ", "negative"),
    ("I absolutely loved this movie. Brilliant, moving and beautifully acted.", "positive"),
    ("A dull, lifeless mess. I regret every minute I wasted on it.", "negative"),
]


def write_csv(path, rows, header=("review", "sentiment")):
    """Write rows in the semicolon-separated cp1252 format the class expects."""
    lines = [";".join(header)] + [";".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="cp1252")
    return path


@pytest.fixture
def make_csv(tmp_path):
    """Return a factory that writes an input CSV into this test's temp directory.

    ``tmp_path`` is a pytest built-in: a fresh empty directory per test, cleaned up
    afterwards, so tests never touch the real ``data/`` or ``notebook_outputs/``.

    Usage::

        path = make_csv([("great film", "positive")])
        path = make_csv([("great film",)], header=("review",))  # unlabelled
    """

    def _make_csv(rows, *, name="reviews.csv", header=("review", "sentiment")):
        return write_csv(tmp_path / name, rows, header)

    return _make_csv


@pytest.fixture
def unlabelled_csv(make_csv):
    """The same reviews with the sentiment column absent — prediction-only input."""
    return make_csv(
        [(review,) for review, _ in LABELLED_ROWS],
        name="reviews_unlabelled.csv",
        header=("review",),
    )


@pytest.fixture
def analyzer(make_csv, tmp_path):
    """A freshly constructed analyzer over a labelled CSV, before any step has run.

    Exports are pointed at ``tmp_path`` so a test that forgets to pass an explicit
    output path still cannot write into the repository.
    """
    return SentimentAnalyzer(make_csv(LABELLED_ROWS), output_dir=tmp_path / "out")


@pytest.fixture
def ready_analyzer(analyzer):
    """An analyzer with check_data() and format_data() already run."""
    analyzer.check_data()
    analyzer.format_data()
    return analyzer


@pytest.fixture(scope="session")
def scored(tmp_path_factory):
    """Run one model for real and return its results frame: ``scored(model)``.

    Calling this *is* the integration test — it exercises a whole code path end to
    end: tokenizer, weights, device selection and frame assembly.

    Two properties matter, and they pull in opposite directions:

    - ``scope="session"`` plus the cache means each model is loaded and run at
      most once for the entire suite, however many tests ask for it. Function
      scope would reload the weights per test and take the suite from seconds to
      minutes.
    - Loading is *lazy*, one model per call, so a framework that fails to load
      takes down only its own tests. Eagerly building all four up front would let
      one broken model error every prediction test and hide the health of the
      other three — the opposite of what this suite is for.

    Weights come from the local Hugging Face cache; the first ever run downloads
    roughly 1.6 GB for distilbert, gpt2 and t5-base.

    Returns:
        A callable taking a :class:`SentimentModel` and returning the frame
        :meth:`SentimentAnalyzer.predict` produced for it.
    """
    path = write_csv(tmp_path_factory.mktemp("scored") / "reviews.csv", LABELLED_ROWS)

    analyzer = SentimentAnalyzer(path)
    analyzer.check_data()
    analyzer.format_data()

    cache: dict[SentimentModel, pd.DataFrame] = {}

    def _scored(model: SentimentModel):
        if model not in cache:
            cache[model] = analyzer.predict(model)
        return cache[model]

    return _scored


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure a test opened.

    ``autouse=True`` applies this to all tests without them asking. Matplotlib
    keeps figures alive globally, so without this the plotting tests leak them
    into each other and eventually warn about too many open figures.
    """
    yield
    plt.close("all")

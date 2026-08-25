"""Sentiment analysis over a CSV of reviews.

Wraps the workflow from ``notebooks/Analysis.ipynb`` in a single class: inspect an
input CSV, format it into features and (optionally) targets, and score it with one
of four sentiment models.
"""

from __future__ import annotations

import os

# transformers probes for TensorFlow on import; the installed TF 2.13 is not
# compatible with protobuf 6.x. This is a torch-only project, so skip the probe.
# Must be set before any transformers import, which is why it sits at module top.
os.environ["USE_TF"] = "0"

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

# Pinned class order, shared with results_analyzer. sklearn orders the cells of a
# confusion matrix by this sequence, so it must be defined in exactly one place.
LABELS: tuple[str, str] = ("negative", "positive")


def _select_device() -> str:
    """Pick the torch device the notebook would have used.

    The notebook hard-codes ``"mps"``; this keeps that choice where it is available
    and degrades to CUDA or CPU elsewhere, so the class is not Apple-only.

    Returns:
        A device string accepted by ``transformers`` and :meth:`torch.Tensor.to`.
    """
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SentimentModel(str, Enum):
    """The four models compared in the notebook.

    The ``str`` mixin means a plain string works wherever a member is expected
    (``predict("vader")``) and the value drops straight into an output filename.
    """

    DISTILBERT = "distilbert"  # distilbert-base-uncased-finetuned-sst-2-english
    VADER = "vader"  # vaderSentiment lexicon, no learned parameters
    GPT2 = "gpt2"  # gpt2 124M, zero-shot via a two-token prompt
    T5 = "t5"  # t5-base, "sst2 sentence: " text-to-text prefix


@dataclass(frozen=True)
class DatasetSchema:
    """What :meth:`SentimentAnalyzer.check_data` read off an input file.

    The file's layout is fixed (see :class:`SentimentAnalyzer`), so this carries
    only what varies between files: the shape of the data and whether it is
    labelled.

    Attributes:
        path: The CSV that was inspected.
        columns: Column names as they appear in the file.
        n_rows: Number of data rows.
        n_missing: Count of missing values per column.
        n_duplicates: Number of fully duplicated rows.
        has_labels: Whether the file carries ground-truth sentiment alongside the
            reviews. This is the switch that decides whether the dataset can be
            scored or only predicted on.
    """

    path: Path
    columns: list[str]
    n_rows: int
    n_missing: dict[str, int]
    n_duplicates: int
    has_labels: bool


class SentimentAnalyzer:
    """Loads a review CSV and scores it with one of four sentiment models.

    The input is expected in the format of the bundled IMDB sample: semicolon
    separated, cp1252 encoded, with the review text under :attr:`REVIEW_COLUMN` and
    its label, when present, under :attr:`LABEL_COLUMN`. Point the class at another
    layout by overriding those four attributes on a subclass.

    Intended call order is :meth:`check_data`, :meth:`format_data`, then
    :meth:`predict`; each step stores its result on the instance and the next step
    raises :class:`RuntimeError` if the previous one has not run. Reviews that are
    already in memory rather than in a file skip all three for
    :meth:`predict_texts`.

    Attributes:
        data_path: The CSV to analyse, or ``None`` when the reviews are supplied
            directly to :meth:`predict_texts` instead of read off disk.
        output_dir: Where :meth:`predict` writes CSV exports by default.
        schema: Result of :meth:`check_data`, ``None`` until it runs.
        raw_data: The frame as read from disk, ``None`` until :meth:`check_data` runs.
        X: Cleaned review text, ``None`` until :meth:`format_data` runs.
        y_target: Normalised ground-truth labels, ``None`` when the input is
            unlabelled or :meth:`format_data` has not run.
    """

    LABELS: tuple[str, str] = LABELS

    # The input format, taken as given rather than detected: the same semicolon
    # separator, cp1252 encoding and column names the notebook hard-codes in cell 1.
    DELIMITER: str = ";"
    ENCODING: str = "cp1252"
    REVIEW_COLUMN: str = "review"
    LABEL_COLUMN: str = "sentiment"

    DEFAULT_OUTPUT_DIR: Path = Path("default_outputs")

    # Review text in the IMDB sample contains raw HTML line breaks.
    HTML_BREAK_PATTERN: re.Pattern[str] = re.compile(r"<br\s*/?>")

    def __init__(
        self,
        data_path: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        """Set up the analyzer without touching the filesystem.

        Reading is deliberately left to :meth:`check_data`, so constructing an
        analyzer for a missing or malformed file never raises.

        Args:
            data_path: Path to the input CSV. Optional: callers that go through
                :meth:`predict_texts` bring their own text and never read a file.
            output_dir: Directory for CSV exports. Defaults to
                :attr:`DEFAULT_OUTPUT_DIR`.
        """
        self.data_path = Path(data_path) if data_path is not None else None
        self.output_dir = Path(output_dir) if output_dir else self.DEFAULT_OUTPUT_DIR

        self.schema: DatasetSchema | None = None
        self.raw_data: pd.DataFrame | None = None
        self.X: pd.Series | None = None
        self.y_target: pd.Series | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_data(self) -> DatasetSchema:
        """Read the input CSV and check it is the dataset this class expects.

        Reads the file with the assumed format, confirms the review column is
        there, notes whether labels came with it, and counts rows, missing values
        and duplicates. Stores the frame on :attr:`raw_data` and the schema on
        :attr:`schema`.

        Returns:
            The schema describing the file.

        Raises:
            RuntimeError: If the analyzer was built without a ``data_path``.
            FileNotFoundError: If :attr:`data_path` does not exist.
            ValueError: If the file parses but has no :attr:`REVIEW_COLUMN`.
        """
        if self.data_path is None:
            raise RuntimeError("No data_path was given; there is no file to check.")
        if not self.data_path.is_file():
            raise FileNotFoundError(f"No such file: {self.data_path}")

        data = pd.read_csv(self.data_path, sep=self.DELIMITER, encoding=self.ENCODING)
        if self.REVIEW_COLUMN not in data.columns:
            raise ValueError(
                f"No {self.REVIEW_COLUMN!r} column in {list(data.columns)}."
            )

        self.raw_data = data
        self.schema = DatasetSchema(
            path=self.data_path,
            columns=list(data.columns),
            n_rows=len(data),
            n_missing={column: int(n) for column, n in data.isna().sum().items()},
            n_duplicates=int(data.duplicated().sum()),
            has_labels=self.LABEL_COLUMN in data.columns,
        )
        return self.schema

    def format_data(self) -> tuple[pd.Series, pd.Series | None]:
        """Format the checked data into features and, when present, targets.

        Cleans the review column into :attr:`X` by stripping HTML breaks and
        surrounding whitespace. When :attr:`schema` reports a label column, also
        normalises it into :attr:`y_target`; when the input is reviews only,
        :attr:`y_target` stays ``None`` and the dataset is prediction-only.

        Returns:
            ``(X, y_target)``, where ``y_target`` is ``None`` for unlabelled input.

        Raises:
            RuntimeError: If :meth:`check_data` has not run.
            ValueError: If the label column holds values outside :attr:`LABELS`.
        """
        if self.schema is None or self.raw_data is None:
            raise RuntimeError("check_data() must run before format_data().")

        self.X = self._clean_reviews(self.raw_data[self.REVIEW_COLUMN])
        self.y_target = (
            self._normalise_labels(self.raw_data[self.LABEL_COLUMN])
            if self.schema.has_labels
            else None
        )
        return self.X, self.y_target

    def predict(
        self,
        model: SentimentModel,
        export_csv: bool = False,
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Score the formatted reviews with one of the four models.

        Dispatches to the private runner for ``model`` and assembles its output
        into the same frame the notebook exports, so results from different models
        are directly comparable and can be passed straight to
        :meth:`results_analyzer.ResultsAnalyzer.compare_models`.

        Args:
            model: Which model to run.
            export_csv: Whether to also write the results to disk.
            output_path: Destination for the export. Defaults to
                ``{output_dir}/{model.value}_predictions.csv``. Ignored when
                ``export_csv`` is ``False``.

        Returns:
            A frame with columns ``Review``, ``y_target``, ``y_pred`` and
            ``confidence``. ``y_target`` is filled with ``NA`` when the input was
            unlabelled.

        Raises:
            RuntimeError: If :meth:`format_data` has not run.
            ValueError: If ``model`` is not a :class:`SentimentModel`.
        """
        if self.X is None:
            raise RuntimeError("format_data() must run before predict().")

        # Accepts a member or its plain string value; anything else is a ValueError.
        model = SentimentModel(model)

        labels, confidences = self._build_dispatch()[model](self.X.tolist())
        results = self._build_results_frame(labels, confidences)

        if export_csv:
            destination = (
                Path(output_path)
                if output_path
                else self.output_dir / f"{model.value}_predictions.csv"
            )
            self._export_results(results, destination)

        return results

    def predict_texts(
        self,
        reviews: Sequence[str],
        model: SentimentModel,
    ) -> pd.DataFrame:
        """Score review text held in memory, without reading a file.

        The in-memory counterpart to the ``check_data`` / ``format_data`` /
        ``predict`` sequence, for reviews that never were a CSV — a request body
        arriving at :mod:`src.api`, say. The text is cleaned exactly as
        :meth:`format_data` cleans the review column, so the same review scored
        through either route reaches the model as the same string.

        Text supplied this way carries no ground truth, so :attr:`y_target` is
        cleared and the ``y_target`` column of the result comes back as ``NA``.

        Args:
            reviews: Review text, uncleaned.
            model: Which model to run.

        Returns:
            The same frame :meth:`predict` returns, one row per review.

        Raises:
            ValueError: If ``reviews`` is empty, or ``model`` is not a
                :class:`SentimentModel`.
        """
        if len(reviews) == 0:
            raise ValueError("No reviews to score.")

        self.X = self._clean_reviews(pd.Series(reviews, dtype="object"))
        self.y_target = None
        return self.predict(model)

    # ------------------------------------------------------------------
    # format_data helpers
    # ------------------------------------------------------------------

    def _clean_reviews(self, reviews: pd.Series) -> pd.Series:
        """Strip HTML line breaks and surrounding whitespace from review text.

        Args:
            reviews: Raw review column.

        Returns:
            The cleaned text, index preserved.
        """
        return (
            reviews.astype(str)
            .str.replace(self.HTML_BREAK_PATTERN, " ", regex=True)
            .str.strip()
        )

    def _normalise_labels(self, labels: pd.Series) -> pd.Series:
        """Lower-case and validate ground-truth labels against :attr:`LABELS`.

        Args:
            labels: Raw label column.

        Returns:
            Labels as lower-case strings.

        Raises:
            ValueError: If any value falls outside :attr:`LABELS`.
        """
        normalised = labels.astype(str).str.strip().str.lower()

        unexpected = set(normalised.unique()) - set(self.LABELS)
        if unexpected:
            raise ValueError(
                f"Labels outside {list(self.LABELS)}: {sorted(unexpected)}."
            )
        return normalised

    # ------------------------------------------------------------------
    # Model runners
    #
    # All four share one signature so predict() stays a dispatch table and never
    # needs to know which model it called. torch, transformers and vaderSentiment
    # are imported inside these methods rather than at module level, so importing
    # this class does not pull in the heavy dependencies.
    # ------------------------------------------------------------------

    def _predict_distilbert(self, reviews: list[str]) -> tuple[list[str], list[float]]:
        """Score reviews with DistilBERT fine-tuned on SST-2.

        A ``transformers`` sentiment-analysis pipeline batches the whole dataset in
        one pass. The pipeline's labels come back upper-case and need lowering to
        line up with :attr:`LABELS`. Truncation at 512 tokens is required and does
        cut text: 15 of the 100 bundled reviews exceed the window.

        Args:
            reviews: Cleaned review text.

        Returns:
            ``(labels, confidences)``, where each confidence is the probability the
            model assigned to the label it picked.
        """
        from transformers import pipeline

        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=_select_device(),
        )
        predictions = classifier(reviews, truncation=True)

        return (
            [p["label"].lower() for p in predictions],
            [p["score"] for p in predictions],
        )

    def _predict_vader(self, reviews: list[str]) -> tuple[list[str], list[float]]:
        """Score reviews with the VADER lexicon.

        No model download and no device selection: this is a dictionary lookup plus
        a handful of grammatical rules, and it reads reviews of any length in full.
        VADER's documented rule calls ``compound >= 0.05`` positive; this dataset
        has no neutral class, so everything below that threshold is called negative.

        Args:
            reviews: Cleaned review text.

        Returns:
            ``(labels, confidences)``. The confidence is ``abs(compound)`` — a
            measure of how polarised the wording is, *not* a probability, so it is
            not strictly comparable to the other three models' numbers.
        """
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        # polarity_scores returns {'neg', 'neu', 'pos', 'compound'}; 'compound' is
        # the normalised [-1, 1] summary score.
        scores = [analyzer.polarity_scores(review) for review in reviews]

        return (
            ["positive" if s["compound"] >= 0.05 else "negative" for s in scores],
            [abs(s["compound"]) for s in scores],
        )

    def _predict_gpt2(self, reviews: list[str]) -> tuple[list[str], list[float]]:
        """Score reviews zero-shot with GPT-2.

        GPT-2 has no classification head, so it is scored as a language model: each
        review is wrapped in a prompt that stops where the answer goes
        (``"Review: ...\\nSentiment (positive or negative):"``) and the next-token
        logits for the two single-token label words are compared. One forward pass
        per review, no text generated. The 1024-token window fits every bundled
        review, but the cap is kept so a prompt can never overflow it.

        Args:
            reviews: Cleaned review text.

        Returns:
            ``(labels, confidences)``, the confidence being a softmax over just the
            two label tokens: 0.5 means the model could not separate them.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        prefix = "Review: "
        suffix = "\nSentiment (positive or negative):"

        device = _select_device()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()

        # Written with a leading space both label words are a single token, so one
        # forward pass per review is enough and no text has to be generated.
        label_ids = [tokenizer.encode(f" {label}")[0] for label in self.LABELS]
        max_review_tokens = (
            1024 - len(tokenizer.encode(prefix)) - len(tokenizer.encode(suffix))
        )

        labels: list[str] = []
        confidences: list[float] = []
        for review in reviews:
            prompt_ids = (
                tokenizer.encode(prefix)
                + tokenizer.encode(review)[:max_review_tokens]
                + tokenizer.encode(suffix)
            )
            with torch.no_grad():
                logits = model(torch.tensor([prompt_ids], device=device)).logits[0, -1]

            # Softmax over just the two label tokens: this is the model's choice
            # between them, ignoring every other continuation it could have written.
            probs = torch.softmax(torch.stack([logits[i] for i in label_ids]), dim=0)
            labels.append(self.LABELS[int(probs.argmax())])
            confidences.append(float(probs.max()))

        return labels, confidences

    def _predict_t5(self, reviews: list[str]) -> tuple[list[str], list[float]]:
        """Score reviews with T5 as a text-to-text task.

        T5 answers every task by generating text, so the prediction is the first
        word the decoder writes. The ``"sst2 sentence: "`` prefix is the exact
        string t5-base was pretrained to answer for SST-2, not an invented prompt.
        One decoder step is enough; input is truncated at T5's 512-token training
        length, which does cut 21 of the 100 bundled reviews.

        Args:
            reviews: Cleaned review text.

        Returns:
            ``(labels, confidences)``, the confidence being a softmax over just the
            two label tokens, on the same scale as :meth:`_predict_gpt2`.
        """
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        prefix = "sst2 sentence: "

        device = _select_device()
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-base").to(device).eval()

        # Both label words are a single sentencepiece token, so one decoder step is
        # enough and no text has to be generated.
        label_ids = [tokenizer(label).input_ids[0] for label in self.LABELS]
        decoder_start = torch.tensor(
            [[model.config.decoder_start_token_id]], device=device
        )

        labels: list[str] = []
        confidences: list[float] = []
        for review in reviews:
            encoded = tokenizer(
                prefix + review, return_tensors="pt", truncation=True, max_length=512
            ).to(device)

            # Feed the decoder its start token and read the logits for the first
            # output word.
            with torch.no_grad():
                logits = model(**encoded, decoder_input_ids=decoder_start).logits[0, -1]

            probs = torch.softmax(torch.stack([logits[i] for i in label_ids]), dim=0)
            labels.append(self.LABELS[int(probs.argmax())])
            confidences.append(float(probs.max()))

        return labels, confidences

    # ------------------------------------------------------------------
    # predict helpers
    # ------------------------------------------------------------------

    def _build_dispatch(self) -> dict[SentimentModel, Callable[[list[str]], tuple[list[str], list[float]]]]:
        """Map each model to its runner, so :meth:`predict` needs no branching.

        Returns:
            Model to bound runner method.
        """
        return {
            SentimentModel.DISTILBERT: self._predict_distilbert,
            SentimentModel.VADER: self._predict_vader,
            SentimentModel.GPT2: self._predict_gpt2,
            SentimentModel.T5: self._predict_t5,
        }

    def _build_results_frame(
        self,
        labels: list[str],
        confidences: list[float],
    ) -> pd.DataFrame:
        """Assemble a model's output into the standard results frame.

        Combines the reviews and (when available) targets held on the instance with
        the predictions just produced.

        Args:
            labels: Predicted label per review.
            confidences: Confidence per prediction.

        Returns:
            A frame with columns ``Review``, ``y_target``, ``y_pred``,
            ``confidence``.

        Raises:
            ValueError: If ``labels`` and ``confidences`` do not match the number
                of reviews.
        """
        if len(labels) != len(self.X) or len(confidences) != len(self.X):
            raise ValueError(
                f"Expected {len(self.X)} predictions, got {len(labels)} labels "
                f"and {len(confidences)} confidences."
            )

        return pd.DataFrame(
            {
                "Review": self.X.reset_index(drop=True),
                "y_target": (
                    self.y_target.reset_index(drop=True)
                    if self.y_target is not None
                    else pd.NA
                ),
                "y_pred": labels,
                "confidence": confidences,
            }
        )

    def _export_results(self, results: pd.DataFrame, path: Path) -> Path:
        """Write a results frame to CSV, creating the parent directory if needed.

        Args:
            results: Frame from :meth:`_build_results_frame`.
            path: Destination file.

        Returns:
            The path written to.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(path, index=False)
        return path

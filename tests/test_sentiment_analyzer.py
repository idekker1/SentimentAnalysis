"""Tests for :class:`src.sentiment_analyzer.SentimentAnalyzer`.

One class per public method, in the order the methods are meant to be called:
check_data, format_data, predict.

Every prediction test runs the real model. The four are loaded once for the whole
suite by the session-scoped ``scored`` fixture in ``conftest.py``, so the cost is
paid a single time rather than per test.

Nothing here asserts that a prediction is *correct*. Accuracy is a property of the
published weights rather than of this code — it would move under a transformers
upgrade or a re-uploaded checkpoint without anything here being broken — and it is
measured properly by the notebook comparison over the full dataset. What these
tests pin down is that each framework loads, runs, and hands back a well-formed
results frame.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.sentiment_analyzer import LABELS, SentimentAnalyzer, SentimentModel


class TestCheckData:
    """Reading the input file and describing what is in it."""

    def test_reads_schema_from_a_labelled_file(self, analyzer):
        schema = analyzer.check_data()

        assert schema.columns == ["review", "sentiment"]
        assert schema.n_rows == 4
        assert schema.has_labels is True
        assert schema.n_duplicates == 0
        assert schema.n_missing == {"review": 0, "sentiment": 0}

    def test_file_without_a_sentiment_column_is_unlabelled(self, unlabelled_csv):
        assert SentimentAnalyzer(unlabelled_csv).check_data().has_labels is False

    def test_an_analyzer_with_no_path_has_nothing_to_check(self):
        # The in-memory entry point leaves data_path unset; reaching check_data()
        # on such an analyzer is a caller mistake rather than a missing file.
        with pytest.raises(RuntimeError, match="No data_path"):
            SentimentAnalyzer().check_data()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SentimentAnalyzer(tmp_path / "does_not_exist.csv").check_data()

    def test_file_without_a_review_column_raises(self, make_csv):
        path = make_csv([("positive",)], header=("sentiment",))

        # `match` is a regex run against the message, so the test pins down *which*
        # error was raised, not merely that something went wrong.
        with pytest.raises(ValueError, match="No 'review' column"):
            SentimentAnalyzer(path).check_data()


class TestFormatData:
    """Turning the raw frame into X and, when labelled, y_target."""

    def test_requires_check_data_first(self, analyzer):
        with pytest.raises(RuntimeError, match="check_data"):
            analyzer.format_data()

    def test_strips_html_breaks_and_surrounding_whitespace(self, analyzer):
        analyzer.check_data()
        X, y_target = analyzer.format_data()

        # Each <br /> becomes a single space, so the doubled break leaves two.
        assert X.iloc[0] == (
            "A wonderful little production.  The filming technique is charming."
        )
        assert X.iloc[1].startswith("Basically") and X.iloc[1].endswith("closet.")
        assert list(y_target) == ["positive", "negative", "positive", "negative"]

    def test_lower_cases_label_text(self, make_csv):
        analyzer = SentimentAnalyzer(make_csv([("A fine film.", "POSITIVE")]))
        analyzer.check_data()

        assert list(analyzer.format_data()[1]) == ["positive"]

    def test_rejects_labels_outside_the_known_classes(self, make_csv):
        analyzer = SentimentAnalyzer(make_csv([("An unremarkable film.", "neutral")]))
        analyzer.check_data()

        with pytest.raises(ValueError, match="neutral"):
            analyzer.format_data()

    def test_unlabelled_input_leaves_targets_as_none(self, unlabelled_csv):
        analyzer = SentimentAnalyzer(unlabelled_csv)
        analyzer.check_data()
        X, y_target = analyzer.format_data()

        assert y_target is None
        assert len(X) == 4


class TestPredict:
    """The four models, run for real.

    ``@pytest.mark.parametrize`` runs one test body once per model and reports
    them as four separate results, so a failure names the framework that broke.
    """

    @pytest.mark.parametrize("model", list(SentimentModel))
    def test_returns_the_standard_results_frame(self, model, scored):
        results = scored(model)

        assert list(results.columns) == ["Review", "y_target", "y_pred", "confidence"]
        assert len(results) == 4

    @pytest.mark.parametrize("model", list(SentimentModel))
    def test_every_review_gets_a_known_label(self, model, scored):
        y_pred = scored(model)["y_pred"]

        # Which label is not the point; that it is one of the two, for every row,
        # is. A model that silently returned None or "LABEL_1" would land here.
        assert y_pred.notna().all()
        assert set(y_pred) <= set(LABELS)

    @pytest.mark.parametrize("model", list(SentimentModel))
    def test_every_prediction_carries_a_usable_confidence(self, model, scored):
        confidence = scored(model)["confidence"]

        assert pd.api.types.is_float_dtype(confidence)
        assert confidence.notna().all()
        assert confidence.between(0, 1).all()

    @pytest.mark.parametrize("model", list(SentimentModel))
    def test_reviews_and_targets_survive_the_round_trip(self, model, scored):
        results = scored(model)

        # Guards the frame assembly rather than the model: predictions must be
        # aligned with the rows they came from, not reordered or reindexed.
        assert results["Review"].iloc[0].startswith("A wonderful little production.")
        assert "<br" not in results["Review"].iloc[0]
        assert list(results["y_target"]) == ["positive", "negative", "positive", "negative"]

    def test_requires_format_data_first(self, analyzer):
        analyzer.check_data()

        with pytest.raises(RuntimeError, match="format_data"):
            analyzer.predict(SentimentModel.VADER)

    def test_unknown_model_raises(self, ready_analyzer):
        with pytest.raises(ValueError):
            ready_analyzer.predict("bag-of-words")

    def test_accepts_a_plain_string_model_name(self, ready_analyzer):
        assert len(ready_analyzer.predict("vader")) == 4

    def test_unlabelled_input_fills_targets_with_na(self, unlabelled_csv):
        analyzer = SentimentAnalyzer(unlabelled_csv)
        analyzer.check_data()
        analyzer.format_data()

        assert analyzer.predict(SentimentModel.VADER)["y_target"].isna().all()


class TestPredictTexts:
    """Scoring review text that never was a file.

    VADER throughout: the route into the models is what is under test here, and
    the lexicon exercises all of it without loading a transformer.
    """

    def test_scores_text_passed_straight_in(self):
        results = SentimentAnalyzer().predict_texts(["A fine film."], SentimentModel.VADER)

        assert list(results.columns) == ["Review", "y_target", "y_pred", "confidence"]
        assert len(results) == 1
        assert results["y_pred"].iloc[0] in LABELS

    def test_cleans_text_the_way_format_data_does(self):
        results = SentimentAnalyzer().predict_texts(
            ["  Wonderful.<br /><br />The filming is charming.  "], SentimentModel.VADER
        )

        # Identical treatment to the CSV route, so the same review scored either
        # way reaches the model as the same string.
        assert results["Review"].iloc[0] == "Wonderful.  The filming is charming."

    def test_scores_several_reviews_in_order(self):
        results = SentimentAnalyzer().predict_texts(
            ["A fine film.", "A dull mess.", "Loved every minute."],
            SentimentModel.VADER,
        )

        assert len(results) == 3
        assert results["Review"].iloc[1] == "A dull mess."

    def test_has_no_targets_to_score_against(self):
        results = SentimentAnalyzer().predict_texts(["A fine film."], SentimentModel.VADER)

        assert results["y_target"].isna().all()

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="No reviews"):
            SentimentAnalyzer().predict_texts([], SentimentModel.VADER)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            SentimentAnalyzer().predict_texts(["A fine film."], "bag-of-words")


class TestExport:
    """Writing results to CSV.

    VADER throughout — this is testing the file handling, and the lexicon gives a
    real results frame without loading a transformer.
    """

    def test_writes_the_frame_to_the_given_path(self, ready_analyzer, tmp_path):
        # A directory that does not exist yet — export is supposed to create it.
        destination = tmp_path / "nested" / "vader.csv"

        results = ready_analyzer.predict(
            SentimentModel.VADER, export_csv=True, output_path=destination
        )

        written = pd.read_csv(destination)
        assert list(written.columns) == list(results.columns)
        assert len(written) == 4

    def test_defaults_to_the_output_dir_and_model_name(self, ready_analyzer):
        ready_analyzer.predict(SentimentModel.VADER, export_csv=True)

        assert (ready_analyzer.output_dir / "vader_predictions.csv").is_file()

    def test_nothing_is_written_unless_export_is_asked_for(self, ready_analyzer):
        ready_analyzer.predict(SentimentModel.VADER)

        assert not ready_analyzer.output_dir.exists()

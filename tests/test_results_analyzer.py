"""Tests for :class:`src.results_analyzer.ResultsAnalyzer`.

Every method here is static or a classmethod over plain label sequences, so these
tests need no fixtures beyond hand-written lists — which also makes the expected
confusion counts checkable by eye.
"""

from __future__ import annotations

import pandas as pd
import pytest
from matplotlib.axes import Axes

from src.results_analyzer import ResultsAnalyzer

# Six rows, three of each class, with one mistake in each direction:
#   true negative 2, false positive 1, false negative 1, true positive 2
# which makes accuracy 4/6.
Y_TARGET = ["negative", "negative", "negative", "positive", "positive", "positive"]
Y_PRED = ["negative", "negative", "positive", "negative", "positive", "positive"]


def results_frame(y_target, y_pred):
    """Build a minimal stand-in for what SentimentAnalyzer.predict() returns."""
    return pd.DataFrame({"y_target": y_target, "y_pred": y_pred})


class TestCalculateAccuracy:
    def test_counts_correct_predictions(self):
        assert ResultsAnalyzer.calculate_accuracy(Y_TARGET, Y_PRED) == pytest.approx(4 / 6)
        assert ResultsAnalyzer.calculate_accuracy(Y_TARGET, Y_TARGET) == 1.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="y_target has 6 rows"):
            ResultsAnalyzer.calculate_accuracy(Y_TARGET, Y_PRED[:5])

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty dataset"):
            ResultsAnalyzer.calculate_accuracy([], [])

    def test_report_is_printed_only_when_asked(self, capsys):
        # capsys is a pytest built-in that captures anything written to stdout.
        ResultsAnalyzer.calculate_accuracy(Y_TARGET, Y_PRED)
        assert capsys.readouterr().out == ""

        ResultsAnalyzer.calculate_accuracy(Y_TARGET, Y_PRED, print_report=True)
        assert "precision" in capsys.readouterr().out


class TestConfusionCells:
    def test_splits_predictions_into_the_four_cells(self):
        cells = ResultsAnalyzer._confusion_cells(Y_TARGET, Y_PRED)

        assert cells["true negative"] == 2
        assert cells["false positive"] == 1
        assert cells["false negative"] == 1
        assert cells["true positive"] == 2
        assert list(cells.index) == list(ResultsAnalyzer.CELL_NAMES)


class TestCompareModels:
    def test_builds_a_column_per_model(self):
        comparison = ResultsAnalyzer.compare_models(
            {
                "vader": results_frame(Y_TARGET, Y_PRED),
                "distilbert": results_frame(Y_TARGET, Y_TARGET),
            },
            plot=False,
        )

        assert list(comparison.columns) == ["vader", "distilbert"]
        assert list(comparison.index) == list(ResultsAnalyzer.CELL_NAMES) + ["accuracy"]
        assert comparison.loc["accuracy", "distilbert"] == 1.0
        assert comparison.loc["false positive", "vader"] == 1

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="at least one results frame"):
            ResultsAnalyzer.compare_models({}, plot=False)

    def test_frame_missing_a_required_column_raises(self):
        frame = pd.DataFrame({"y_target": Y_TARGET})

        with pytest.raises(ValueError, match="missing column"):
            ResultsAnalyzer.compare_models({"vader": frame}, plot=False)

    def test_models_scored_on_different_data_cannot_be_compared(self):
        other_target = ["positive"] * 6

        with pytest.raises(ValueError, match="different y_target"):
            ResultsAnalyzer.compare_models(
                {
                    "vader": results_frame(Y_TARGET, Y_PRED),
                    "gpt2": results_frame(other_target, Y_PRED),
                },
                plot=False,
            )

    def test_drawing_the_grid_does_not_change_the_numbers(self):
        # The Agg backend set in conftest.py means this renders without a window.
        without_plot = ResultsAnalyzer.compare_models(
            {"vader": results_frame(Y_TARGET, Y_PRED)}, plot=False
        )
        with_plot = ResultsAnalyzer.compare_models(
            {"vader": results_frame(Y_TARGET, Y_PRED)}, plot=True
        )

        pd.testing.assert_frame_equal(without_plot, with_plot)


class TestPlotConfusionMatrix:
    """Plotting is checked for structure only.

    Asserting on pixels would break on any matplotlib upgrade; what matters is that
    the call succeeds, honours the pinned label order and returns usable axes.
    """

    def test_returns_axes_with_the_pinned_label_order(self):
        ax = ResultsAnalyzer.plot_confusion_matrix(Y_TARGET, Y_PRED, title="vader")

        assert isinstance(ax, Axes)
        assert ax.get_title() == "vader"
        assert ax.get_ylabel() == "Actual"
        assert [t.get_text() for t in ax.get_xticklabels()] == list(ResultsAnalyzer.LABELS)

    def test_draws_into_a_supplied_axes(self):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()

        assert ResultsAnalyzer.plot_confusion_matrix(Y_TARGET, Y_PRED, ax=ax) is ax

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="y_target has 6 rows"):
            ResultsAnalyzer.plot_confusion_matrix(Y_TARGET, Y_PRED[:5])

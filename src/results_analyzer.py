"""Scoring and comparison of sentiment predictions.

The evaluation half of ``notebooks/Analysis.ipynb``: accuracy, confusion matrices,
and the cell-by-cell comparison across all four models.

Every public method is a static method taking ``y_target`` and ``y_pred``
explicitly, so nothing has to be constructed to use it.
"""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.sentiment_analyzer import LABELS


class ResultsAnalyzer:
    """Evaluates and compares the output of :class:`SentimentAnalyzer.predict`.

    Stateless by design — the methods are a namespace of related functions rather
    than the behaviour of an object, which keeps them callable directly from a
    notebook.
    """

    LABELS: tuple[str, str] = LABELS

    # The order sklearn's 2x2 confusion matrix ravels into, given LABELS.
    CELL_NAMES: tuple[str, str, str, str] = (
        "true negative",
        "false positive",
        "false negative",
        "true positive",
    )

    # Colour encodes correctness, not model — the model name is already on the axis.
    CORRECT_COLOR: str = "#2b9459"
    ERROR_COLOR: str = "#d9534f"

    # Plain cells rather than heatmap shading, so matrices for different models can
    # be read against each other instead of each against its own colour scale.
    PLAIN_CMAP: ListedColormap = ListedColormap(["white"])

    @staticmethod
    def plot_confusion_matrix(
        y_target: pd.Series | list[str],
        y_pred: pd.Series | list[str],
        *,
        labels: tuple[str, ...] = LABELS,
        ax: Axes | None = None,
        title: str | None = None,
    ) -> Axes:
        """Draw the confusion matrix for one model's predictions.

        Renders unshaded with no colorbar and a pinned label order, so the matrix
        reads in the same order as a classification report and several models'
        matrices stay comparable side by side.

        Args:
            y_target: Ground-truth labels.
            y_pred: Predicted labels.
            labels: Class order for the rows and columns.
            ax: Axes to draw into. A new figure is created when ``None``.
            title: Optional title, typically the model name.

        Returns:
            The axes the matrix was drawn into.

        Raises:
            ValueError: If ``y_target`` and ``y_pred`` differ in length.
        """
        if len(y_target) != len(y_pred):
            raise ValueError(
                f"y_target has {len(y_target)} rows, y_pred has {len(y_pred)}."
            )

        if ax is None:
            _, ax = plt.subplots(figsize=(3.2, 3.2))

        ConfusionMatrixDisplay.from_predictions(
            y_target,
            y_pred,
            labels=list(labels),
            ax=ax,
            cmap=ResultsAnalyzer.PLAIN_CMAP,
            colorbar=False,
            # The shading-based text colour would be white on white.
            text_kw={"color": "black"},
        )
        ax.set_ylabel("Actual")
        if title:
            ax.set_title(title)

        return ax

    @staticmethod
    def calculate_accuracy(
        y_target: pd.Series | list[str],
        y_pred: pd.Series | list[str],
        *,
        print_report: bool = False,
    ) -> float:
        """Compute the accuracy of one model's predictions on a dataset.

        Args:
            y_target: Ground-truth labels.
            y_pred: Predicted labels.
            print_report: Whether to also print the per-class precision/recall
                breakdown shown under each model in the notebook.

        Returns:
            Accuracy as a fraction in ``[0, 1]``.

        Raises:
            ValueError: If ``y_target`` and ``y_pred`` differ in length, or either
                is empty.
        """
        if len(y_target) != len(y_pred):
            raise ValueError(
                f"y_target has {len(y_target)} rows, y_pred has {len(y_pred)}."
            )
        if len(y_target) == 0:
            raise ValueError("Cannot score an empty dataset.")

        if print_report:
            print(classification_report(y_target, y_pred, digits=3))

        return float(accuracy_score(y_target, y_pred))

    @classmethod
    def compare_models(
        cls,
        results: Mapping[str, pd.DataFrame],
        *,
        plot: bool = True,
    ) -> pd.DataFrame:
        """Compare several models scored on the same dataset.

        Takes the frames produced by :meth:`SentimentAnalyzer.predict` and pulls
        each model's confusion matrix apart into its four cells, so "which model
        wins" splits into four separate questions — who finds the negative reviews,
        who over-calls positive, who misses positives, who finds them.

        Args:
            results: Model name to its results frame; each must carry ``y_target``
                and ``y_pred`` columns.
            plot: Whether to also draw the comparison grid.

        Returns:
            A frame indexed by :attr:`CELL_NAMES` plus ``accuracy``, with one
            column per model.

        Raises:
            ValueError: If ``results`` is empty, a frame is missing the required
                columns, or the frames do not share an identical ``y_target`` —
                the comparison is only meaningful against a common ground truth.
        """
        if not results:
            raise ValueError("compare_models() needs at least one results frame.")

        y_target: pd.Series | None = None
        for name, frame in results.items():
            missing = {"y_target", "y_pred"} - set(frame.columns)
            if missing:
                raise ValueError(f"{name!r} is missing column(s) {sorted(missing)}.")

            targets = frame["y_target"].reset_index(drop=True)
            if y_target is None:
                y_target = targets
            elif not targets.equals(y_target):
                raise ValueError(
                    f"{name!r} was scored against a different y_target; models can "
                    "only be compared on a common ground truth."
                )

        cell_counts = pd.DataFrame(
            {
                name: cls._confusion_cells(frame["y_target"], frame["y_pred"])
                for name, frame in results.items()
            }
        )
        accuracy = pd.DataFrame(
            {
                name: [cls.calculate_accuracy(frame["y_target"], frame["y_pred"])]
                for name, frame in results.items()
            },
            index=["accuracy"],
        )

        if plot:
            cls._plot_comparison_grid(
                cell_counts,
                n_negative=int((y_target == cls.LABELS[0]).sum()),
                n_positive=int((y_target == cls.LABELS[1]).sum()),
            )

        return pd.concat([cell_counts, accuracy])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _confusion_cells(
        y_target: pd.Series | list[str],
        y_pred: pd.Series | list[str],
        labels: tuple[str, ...] = LABELS,
    ) -> pd.Series:
        """Reduce one model's predictions to the four confusion matrix counts.

        Args:
            y_target: Ground-truth labels.
            y_pred: Predicted labels.
            labels: Class order, which fixes which cell is which.

        Returns:
            The counts indexed by :attr:`CELL_NAMES`.
        """
        # confusion_matrix returns [[TN, FP], [FN, TP]] with negative first.
        return pd.Series(
            confusion_matrix(y_target, y_pred, labels=list(labels)).ravel(),
            index=list(ResultsAnalyzer.CELL_NAMES),
        )

    @classmethod
    def _plot_comparison_grid(
        cls,
        cell_counts: pd.DataFrame,
        n_negative: int,
        n_positive: int,
    ) -> Figure:
        """Draw one bar chart per confusion cell, with one bar per model.

        Tiles are laid out like the matrix itself — actual-negative on the top row,
        actual-positive on the bottom, correct calls on the diagonal — and share a
        y-axis so bar heights compare between tiles and not only within one. That
        leaves the error tiles with short bars, which is the honest picture; the
        value is labelled on every bar to keep them readable.

        Args:
            cell_counts: Confusion cell counts, models as columns.
            n_negative: Total negative reviews in the ground truth, for the tile
                subtitles and a shared y-limit.
            n_positive: Total positive reviews in the ground truth.

        Returns:
            The completed figure.
        """
        cell_color = {
            "true negative": cls.CORRECT_COLOR,
            "false positive": cls.ERROR_COLOR,
            "false negative": cls.ERROR_COLOR,
            "true positive": cls.CORRECT_COLOR,
        }
        titles = {
            "true negative": f"True negatives (of {n_negative} negative)",
            "false positive": f"False positives (of {n_negative} negative)",
            "false negative": f"False negatives (of {n_positive} positive)",
            "true positive": f"True positives (of {n_positive} positive)",
        }

        fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey=True)

        for ax, cell in zip(axes.ravel(), cls.CELL_NAMES):
            counts = cell_counts.loc[cell]
            bars = ax.bar(counts.index, counts.values, width=0.6, color=cell_color[cell])
            ax.bar_label(bars, padding=3, fontsize=9)

            ax.set_title(titles[cell], fontsize=11, loc="left")
            ax.set_ylim(0, max(n_negative, n_positive) * 1.15)
            ax.tick_params(axis="both", labelsize=9, length=0)
            ax.grid(axis="y", color="#e6e5e2", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#c3c2b7")

        axes[0, 0].set_ylabel("Reviews")
        axes[1, 0].set_ylabel("Reviews")

        fig.suptitle("Confusion matrix cell by cell", fontsize=13, x=0.06, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        return fig

"""Run the sentiment models over a dataset and write the analysis to a folder.

The command-line front end to :mod:`src`: it walks the same three steps the README
describes — ``check_data``, ``format_data``, ``predict`` — once per model, then hands
the predictions to :class:`ResultsAnalyzer` for scoring. Everything it produces goes
into a single output directory, leaving the input data and ``notebook_outputs/``
untouched.

    python main.py
    python main.py --models vader t5 --output-dir outputs/quick-check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# Nothing here opens a window — every figure is written to disk — so pin the
# non-interactive backend. Must happen before pyplot is imported, directly or by
# way of results_analyzer, which is why it sits above those imports.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src.results_analyzer import ResultsAnalyzer
from src.sentiment_analyzer import DatasetSchema, SentimentAnalyzer, SentimentModel

DEFAULT_DATA_PATH = Path("data/IMDB-movie-reviews.csv")

# Kept out of notebook_outputs/, which holds the predictions the notebook produced.
# Already covered by .gitignore, so a run does not leave the working tree dirty.
DEFAULT_OUTPUT_DIR = Path("outputs")

FIGURE_DPI = 150


def main() -> None:
    """Run the models named on the command line and write out the analysis."""
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = SentimentAnalyzer(args.data, output_dir=args.output_dir)
    report_schema(analyzer.check_data())
    analyzer.format_data()

    results = run_models(analyzer, args.models)

    if analyzer.y_target is None:
        # No ground truth came with the file, so the predictions are all there is
        # to write; scoring them is not possible.
        print("\nInput is unlabelled — predictions written, scoring skipped.")
    else:
        score_models(results, args.output_dir)
        compare(results, args.output_dir)

    print(f"\nDone. Results are in {args.output_dir}/")


def parse_args() -> argparse.Namespace:
    """Read the dataset, output directory and model list off the command line.

    Returns:
        The parsed arguments, with ``models`` already converted to
        :class:`SentimentModel` members.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Input review CSV (default: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write the results (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=SentimentModel,
        choices=list(SentimentModel),
        default=list(SentimentModel),
        metavar="MODEL",
        help=(
            "Models to run, from "
            f"{', '.join(model.value for model in SentimentModel)} "
            "(default: all four)."
        ),
    )
    return parser.parse_args()


def report_schema(schema: DatasetSchema) -> None:
    """Print what :meth:`SentimentAnalyzer.check_data` found in the input file.

    Args:
        schema: The :class:`DatasetSchema` returned by ``check_data()``.
    """
    missing = {column: n for column, n in schema.n_missing.items() if n}

    print(f"Dataset:    {schema.path}")
    print(f"Rows:       {schema.n_rows}")
    print(f"Columns:    {', '.join(schema.columns)}")
    print(f"Missing:    {missing or 'none'}")
    print(f"Duplicates: {schema.n_duplicates}")
    print(f"Labelled:   {'yes' if schema.has_labels else 'no'}")


def run_models(
    analyzer: SentimentAnalyzer,
    models: list[SentimentModel],
) -> dict[str, pd.DataFrame]:
    """Score the formatted reviews with each model, exporting as it goes.

    Args:
        analyzer: An analyzer that has already run ``format_data()``.
        models: The models to run.

    Returns:
        Model name to its results frame, ready for :meth:`compare`.
    """
    results: dict[str, pd.DataFrame] = {}

    for model in models:
        # The first run of a transformer model downloads its weights, which is
        # slow enough to be worth announcing before it starts rather than after.
        # Flushed because a redirected stdout is block-buffered, which would hold
        # the line back until the run it is announcing had already finished.
        print(f"\nRunning {model.value}...", flush=True)
        results[model.value] = analyzer.predict(model, export_csv=True)
        # Where predict() puts it, given the output_dir the analyzer was built with.
        print(f"  predictions -> {analyzer.output_dir}/{model.value}_predictions.csv")

    return results


def score_models(results: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Print each model's accuracy and save its confusion matrix.

    Args:
        results: Model name to results frame, from :func:`run_models`.
        output_dir: Directory to write the figures into.
    """
    print("\nAccuracy")
    for name, frame in results.items():
        accuracy = ResultsAnalyzer.calculate_accuracy(
            frame["y_target"], frame["y_pred"]
        )
        print(f"  {name:<12} {accuracy:.2f}")

        ax = ResultsAnalyzer.plot_confusion_matrix(
            frame["y_target"], frame["y_pred"], title=name
        )
        save_figure(ax.figure, output_dir / f"{name}_confusion_matrix.png")


def compare(results: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Write the cross-model comparison table and its plot.

    Args:
        results: Model name to results frame, from :func:`run_models`.
        output_dir: Directory to write the table and figure into.
    """
    comparison = ResultsAnalyzer.compare_models(results, plot=True)

    table_path = output_dir / "model_comparison.csv"
    comparison.to_csv(table_path)

    # compare_models() draws the grid but keeps the figure to itself, so take the
    # one it just created — nothing else has drawn since.
    save_figure(plt.gcf(), output_dir / "model_comparison.png")

    print(f"\n{comparison}")
    print(f"\n  comparison -> {table_path}")


def save_figure(figure: Figure, path: Path) -> None:
    """Write a figure to disk and close it.

    Closing matters here in a way it does not in a notebook: a run over four models
    opens five figures, and matplotlib warns once twenty are left open.

    Args:
        figure: The figure to save.
        path: Destination PNG.
    """
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()

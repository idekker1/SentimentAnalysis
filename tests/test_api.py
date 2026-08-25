"""Tests for the HTTP front end in :mod:`src.api`.

Scoring a review loads a model, so the endpoint is exercised for real only a
handful of times, and always through VADER, whose lexicon needs no download and
no device. The other three models are covered where they belong, against the
analyzer itself in ``test_sentiment_analyzer.py``; what is left for here is the
HTTP layer — the routes, the request body FastAPI accepts, and the shape of what
comes back.

Nothing here asserts that a prediction is *correct*, for the same reason that file
gives: accuracy belongs to the published weights, not to this code.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import PredictionRequest, app
from src.sentiment_analyzer import LABELS, SentimentModel


@pytest.fixture(scope="module")
def client():
    """A test client over the app, built once for the whole module.

    ``TestClient`` calls the app in-process rather than over a socket, so there is
    no server to start and no port to pick.
    """
    return TestClient(app)


class TestHealth:
    """The liveness endpoint."""

    def test_reports_ok(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestPredict:
    """A review going in and a prediction coming back, VADER running for real."""

    def test_returns_a_known_label_and_a_usable_confidence(self, client):
        response = client.post(
            "/predict",
            json={"text": "A brilliant, moving film.", "model": "vader"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["sentiment"] in LABELS
        assert 0 <= body["confidence"] <= 1

    def test_echoes_the_model_that_ran(self, client):
        response = client.post(
            "/predict",
            json={"text": "A dull, lifeless mess.", "model": "vader"},
        )

        assert response.json()["model"] == "vader"

    def test_returns_the_text_the_model_saw_not_the_text_that_was_sent(self, client):
        response = client.post(
            "/predict",
            json={
                "text": "  Wonderful.<br /><br />The filming is charming.  ",
                "model": "vader",
            },
        )

        # The same cleanup format_data() does to a CSV column, so a caller can see
        # what was actually scored. Each <br /> becomes a space, hence the two.
        assert response.json()["text"] == "Wonderful.  The filming is charming."


class TestRequestBody:
    """What the endpoint accepts, checked without running anything."""

    def test_model_defaults_to_distilbert(self):
        assert PredictionRequest(text="A fine film.").model is SentimentModel.DISTILBERT

    @pytest.mark.parametrize("model", list(SentimentModel))
    def test_every_model_can_be_asked_for_by_name(self, model):
        assert PredictionRequest(text="A fine film.", model=model.value).model is model

    def test_unknown_model_is_rejected(self, client):
        response = client.post(
            "/predict",
            json={"text": "A fine film.", "model": "bag-of-words"},
        )

        assert response.status_code == 422

    def test_missing_text_is_rejected(self, client):
        assert client.post("/predict", json={"model": "vader"}).status_code == 422

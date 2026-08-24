r"""HTTP front end to :meth:`SentimentAnalyzer.predict_texts`.

Two endpoints, no more: ``GET /health`` to say the service is up, and
``POST /predict`` to score one review. Serve it with::

    uvicorn src.api:app --reload

Then::

    curl -X POST localhost:8000/predict \
         -H 'Content-Type: application/json' \
         -d '{"text": "A brilliant, moving film.", "model": "vader"}'

Scoring a review loads a model, which takes seconds rather than milliseconds, so
both handlers are plain ``def`` rather than ``async def``. FastAPI runs a sync
handler in its threadpool, leaving the event loop free to accept other requests
while the model works; an ``async def`` doing the same work would block the loop
and stall the whole process for the duration.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.sentiment_analyzer import SentimentAnalyzer, SentimentModel

# Fine-tuned on SST-2 and the fastest of the three transformers, so it is the
# sensible thing to hand a caller who expressed no preference. Any member of
# SentimentModel can be asked for by name in the request.
DEFAULT_MODEL = SentimentModel.DISTILBERT

app = FastAPI(
    title="Sentiment Analysis API",
    description="Scores a movie review as positive or negative.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    """The body of a ``POST /predict`` call.

    Attributes:
        text: The review to score.
        model: Which of the four models to run. Anything outside
            :class:`SentimentModel` is rejected by FastAPI with a 422 naming the
            valid values.
    """

    text: str = Field(description="The review text to score.")
    model: SentimentModel = Field(
        default=DEFAULT_MODEL,
        description="Which model to run.",
    )


class PredictionResponse(BaseModel):
    """The body of a successful ``POST /predict`` call.

    Attributes:
        text: The review as the model saw it — HTML breaks and surrounding
            whitespace stripped — rather than as it was sent.
        model: The model that produced the prediction.
        sentiment: ``"positive"`` or ``"negative"``.
        confidence: How strongly the model picked that label, in ``[0, 1]``. It
            is a probability for every model except VADER, where it is the
            strength of the lexicon score; see the runners in
            :mod:`src.sentiment_analyzer` for what each number means.
    """

    text: str
    model: SentimentModel
    sentiment: str
    confidence: float


class HealthResponse(BaseModel):
    """The body of a ``GET /health`` call.

    Attributes:
        status: Always ``"ok"``; the response arriving at all is the signal.
    """

    status: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report that the service is up.

    Deliberately cheap: it loads no model and touches no disk, so a load balancer
    can poll it while a prediction is in flight.

    Returns:
        The fixed ``{"status": "ok"}`` body.
    """
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Score one review with one model.

    Args:
        request: The review text and the model to run it through.

    Returns:
        The predicted sentiment and the model's confidence in it.

    Raises:
        HTTPException: 422, raised by FastAPI itself, for a malformed body or a
            model name outside :class:`SentimentModel`.
    """
    # A fresh analyzer per request: it holds the text of the review being scored,
    # so sharing one across the threadpool would let concurrent requests overwrite
    # each other's input. It is a cheap object — the model weights it loads are
    # cached by transformers, not by this.
    results = SentimentAnalyzer().predict_texts([request.text], request.model)
    prediction = results.iloc[0]

    return PredictionResponse(
        text=prediction["Review"],
        model=request.model,
        sentiment=prediction["y_pred"],
        confidence=float(prediction["confidence"]),
    )

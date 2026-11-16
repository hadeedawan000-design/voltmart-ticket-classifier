"""
VoltMart Electronics - support ticket classifier API (FastAPI).

Run locally:   uvicorn api_app:app --reload
Docs page:     http://127.0.0.1:8000/docs

What it does:
  * loads best_model.joblib once, when the server starts
  * POST /predict takes a JSON ticket description and returns category + confidence
  * errors are written to api_errors.log, every request is written to api_requests.log
"""
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.joblib"
ERROR_LOG = BASE_DIR / "api_errors.log"
REQUEST_LOG = BASE_DIR / "api_requests.log"

# ---------------------------------------------------------------- logging
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ticket_api")
logger.setLevel(logging.INFO)
logger.propagate = False

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Log files are a bonus: if the folder is read-only the API still works (console logging stays on).
for path, level in ((ERROR_LOG, logging.ERROR), (REQUEST_LOG, logging.INFO)):
    try:
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning("Could not open log file %s: %s", path.name, exc)

# ------------------------------------------------------------------ model
state = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup: load the saved model."""
    try:
        state["model"] = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s. Categories: %s",
                    MODEL_PATH.name, list(state["model"].classes_))
    except Exception:
        logger.exception("Could not load model from %s", MODEL_PATH)
    yield


app = FastAPI(
    title="VoltMart Ticket Classifier",
    description="Send a support ticket description and get the predicted category and a confidence score.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------- request/response
class TicketRequest(BaseModel):
    description: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The text of the customer support ticket.",
        validation_alias=AliasChoices("description", "text", "ticket", "ticket_text"),
        examples=["veb saytım çox yavaş yüklənir və bunun səbəbini bilmirəm"],
    )

    @field_validator("description")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("description must not be empty or only spaces")
        return value


class PredictResponse(BaseModel):
    category: str
    confidence: float
    scores: dict[str, float]


# ---------------------------------------------------------------- helpers
def predict_with_confidence(model, text: str) -> PredictResponse:
    # The model was trained on lower-case text (Task 1 cleaning), so do the same here.
    cleaned = text.strip().lower()
    classes = [str(c) for c in model.classes_]

    if hasattr(model, "predict_proba"):
        # e.g. LogisticRegression: real probabilities
        probabilities = model.predict_proba([cleaned])[0]
    else:
        # e.g. LinearSVC has no probabilities: turn its decision scores into 0-1 values (softmax).
        scores = np.ravel(model.decision_function([cleaned]))
        if scores.size == 1:                       # only two categories
            scores = np.array([-scores[0], scores[0]])
        exp = np.exp(scores - scores.max())
        probabilities = exp / exp.sum()

    best = int(np.argmax(probabilities))
    return PredictResponse(
        category=classes[best],
        confidence=round(float(probabilities[best]), 4),
        scores={c: round(float(p), 4) for c, p in zip(classes, probabilities)},
    )


# --------------------------------------------------------------- handlers
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, response.status_code, ms)
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    problems = [
        {"field": ".".join(str(p) for p in err["loc"] if p != "body"), "message": err["msg"]}
        for err in exc.errors()
    ]
    logger.warning("Bad request to %s: %s", request.url.path, problems)
    return JSONResponse(status_code=422, content={"error": "Invalid request", "details": problems})


# -------------------------------------------------------------- endpoints
@app.get("/")
def home():
    return {
        "service": "VoltMart Ticket Classifier",
        "status": "ok" if state["model"] is not None else "model not loaded",
        "usage": "POST /predict with JSON {\"description\": \"your ticket text\"}",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok" if state["model"] is not None else "model not loaded",
            "model_loaded": state["model"] is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(ticket: TicketRequest):
    model = state["model"]
    if model is None:
        logger.error("Prediction requested but the model is not loaded")
        raise HTTPException(status_code=503, detail="Model is not loaded. Check the server logs.")

    try:
        result = predict_with_confidence(model, ticket.description)
    except Exception:
        logger.exception("Prediction failed (text length %d)", len(ticket.description))
        raise HTTPException(status_code=500, detail="Prediction failed. The error was logged.")

    # Log only the length, not the ticket text itself (tickets can contain private information).
    logger.info("Predicted '%s' (confidence %.3f) for a ticket of %d characters",
                result.category, result.confidence, len(ticket.description))
    return result

# Support Ticket Classification for VoltMart Electronics

VoltMart Electronics is an online store that receives dozens of customer support tickets every day.
This project builds a small machine-learning classifier that reads a ticket and suggests its category,
and serves it through a **FastAPI** web service so other programs can use it.

| Link | URL |
|------|-----|
| Running API | Runs locally with uvicorn at `http://127.0.0.1:8000` (see "Deployment status" below) |
| Interactive docs (try it in the browser) | `http://127.0.0.1:8000/docs` once the server is running |
| Demo video | [demo_prediction.mp4](demo_prediction.mp4) |
| Source code | https://github.com/hadeedawan000-design/voltmart-ticket-classifier |

## Project parts

| Task | What | Files |
|------|------|-------|
| 1 | Load and clean the support tickets | `load_and_clean_tickets.ipynb`, `cleaned_tickets.csv` |
| 2 | Baseline and improved models with 5-fold cross-validation | `baseline_and_improved.ipynb`, `best_model.joblib` |
| 3 | FastAPI service and deployment | `api_app.py`, `requirements.txt`, `Dockerfile` (optional, for Docker hosts), `test_api.py` |

## Run it on your computer

You need Python 3.9 or newer.

```bash
# 1. go to the project folder
cd voltmart-ticket-classifier

# 2. install the packages
pip install -r requirements.txt

# 3. start the server
uvicorn api_app:app --reload
```

The server starts at http://127.0.0.1:8000. Open http://127.0.0.1:8000/docs to try it in your browser.
`best_model.joblib` must be in the same folder as `api_app.py`.

## Test it

**With curl** (Windows Command Prompt):

```
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"description\": \"veb saytım çox yavaş yüklənir\"}"
```

**With curl** (Mac / Linux):

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"description": "veb saytım çox yavaş yüklənir"}'
```

**With the test script** (uses httpx, checks several tickets and bad inputs):

```bash
python test_api.py                              # local server
python test_api.py https://YOUR-LIVE-URL        # a deployed copy, if you host one
```

## API reference

### `POST /predict`

Request (JSON):

```json
{"description": "veb saytım çox yavaş yüklənir və bunun səbəbini bilmirəm"}
```

`text`, `ticket` and `ticket_text` are also accepted as the field name.

Response (JSON):

```json
{
  "category": "website",
  "confidence": 0.93,
  "scores": {"email": 0.04, "website": 0.93}
}
```

- `category`: the predicted ticket category.
- `confidence`: the model's score for that category (0 to 1).
- `scores`: the score for every category.

Other endpoints: `GET /` (service info) and `GET /health` (is the model loaded?).

### Errors

| Status | When | Example |
|--------|------|---------|
| 422 | Missing, empty or blank `description` (or more than 5000 characters) | `{"error": "Invalid request", "details": [...]}` |
| 503 | `best_model.joblib` could not be loaded | `{"detail": "Model is not loaded..."}` |
| 500 | Unexpected error while predicting | `{"detail": "Prediction failed..."}` |

### Logging

- `api_errors.log`: errors (for example a model that fails to load or a failed prediction), with details.
- `api_requests.log`: one line per request (method, path, status, time) and each prediction (category,
  confidence and ticket length). The ticket text itself is not logged because it can contain private information.

## Deployment status

The brief asks for a free Hugging Face Space. At the time of writing (November 2026), the Space creation page for a
free account marks **Docker** and **Gradio** Spaces as **Paid** ("Gradio and Docker Spaces require a paid plan"),
and only **Static** Spaces are free. A Static Space cannot run a Python API, so the service was **not** hosted on a public URL.

Instead, the service was run and tested with **uvicorn on a laptop**, and `demo_prediction.mp4` shows it running and
answering a curl test. Everything needed to host it later is included:

- `Dockerfile`, which runs the app on port 7860 (the port Hugging Face Spaces uses)
- `requirements.txt`

### To host it on Hugging Face Spaces (needs a plan that allows Docker Spaces)
1. Create a new **Docker** Space and upload `api_app.py`, `best_model.joblib`, `requirements.txt` and `Dockerfile`.
2. Add this block at the very top of the Space's `README.md`:

```
---
title: VoltMart Ticket Classifier
sdk: docker
app_port: 7860
---
```

3. Wait until the Space shows **Running**, then test it with `python test_api.py https://YOUR-SPACE.hf.space`.

---

# Model card

## Purpose
Suggest the category of a customer support ticket so support staff can route it faster.
It is a helper for humans. It is not meant to decide anything on its own.

## Model
- **Type:** scikit-learn `Pipeline` saved with joblib (`best_model.joblib`).
- **Chosen model:** Character 2-5 grams TF-IDF + LinearSVC (called "Char 2-5 grams + LinearSVC" in the notebook).>
- **Baseline for comparison:** TF-IDF + Logistic Regression.
- **Input:** ticket text, which is converted to lower case before prediction (the training data was lower-cased too).
- **Output:** predicted category, a confidence score, and a score for every category.
- **Confidence:** if the chosen model is Logistic Regression, this is its predicted probability. If it is LinearSVC,
  the model has no probabilities, so the API converts its decision scores to 0-1 values with a softmax.
  In both cases, treat the score as a rough guide. It is not a calibrated probability.

## Training data
- **Source:** the public Hugging Face dataset `interneuronai/customer_support_ticket_classification_pegasus_dataset`.
- **Size:** 3,633 raw rows reduced to 2,839 after cleaning (lower-casing, removing missing values and duplicates).
- **Columns:** `text` (the ticket) and `label` (the category).
- **Categories (2):** `email` (1,594 tickets) and `website` (1,245 tickets).
- **Language:** the ticket text is **Azerbaijani**. Some tickets use the full Azerbaijani letters and some use a
  simplified spelling without them (for example "veb saydim ishlemir").
- **Cleaning:** see `load_and_clean_tickets.ipynb`.

## Evaluation
5-fold stratified cross-validation on `cleaned_tickets.csv` (see `baseline_and_improved.ipynb`).

| Model | Accuracy (mean) | F1 macro (mean) |
|-------|-----------------|-----------------|
| Baseline: TF-IDF + Logistic Regression | 0.9824 | 0.9821 |
| Improved: Char 2-5 grams + LinearSVC | 0.9968 | 0.9968 |

The improved model's mean F1 is 0.0147 higher than the baseline's. Three improved models were tried
(word 1-2 grams + Logistic Regression, word 1-2 grams + LinearSVC, character 2-5 grams + LinearSVC)
and the best one was saved as `best_model.joblib`.

A leakage check confirmed that no identical ticket text appears in both the training and test part of any fold.

## Limitations
- **Language:** it was trained on Azerbaijani tickets only. English or other languages will give unreliable results.
- **Small dataset:** about 2,800 tickets. Rare topics and new product names may be misclassified.
- **Only two categories:** it can only answer `email` or `website`. It has no "unknown" option,
  so a ticket about something else (for example payment or delivery) is still forced into one of the two.
- **Scores are probably too optimistic:** the tickets in the dataset are very similar to each other (many repeat the
  same few problems in different words), so a score near 99% probably does not carry over to real, varied tickets.
- **Not calibrated:** a high confidence does not guarantee a correct answer.
- **Plain text only:** it uses word and character patterns. It does not understand meaning, sarcasm or context.
- **Data source:** the dataset is public and was not collected from VoltMart. Real VoltMart tickets may look different.
- **Human review:** use it to suggest a category and always let a person check important or sensitive tickets.

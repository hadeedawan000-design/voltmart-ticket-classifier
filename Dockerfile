# Used by Hugging Face Spaces (Docker Space) to run the FastAPI app.
# Spaces expects the app to listen on port 7860.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "7860"]

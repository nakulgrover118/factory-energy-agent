FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY ui ./ui
COPY tests ./tests

RUN useradd --create-home app && mkdir -p /app/store && chown -R app /app
USER app

EXPOSE 8000 8501
CMD ["sh", "-c", "python -m energy.bootstrap && uvicorn energy.api:app --host 0.0.0.0 --port 8000"]

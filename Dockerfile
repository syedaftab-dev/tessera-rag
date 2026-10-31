FROM python:3.11-slim-bookworm

# Patch OS-level CVEs, then install system deps required by onnxruntime and native packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so pip install is a cached layer.
# Re-runs only when requirements.txt changes, not on every code change.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy only the app package — everything else (evals/, ui/, DATA/, DOCS/) stays out
COPY app/ ./app/

# Render provides dynamic PORT (defaults to 8080 if not set)
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

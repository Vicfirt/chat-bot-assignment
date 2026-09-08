FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app HF_HOME=/home/app/.cache/huggingface

RUN useradd -m app
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY eval ./eval
COPY loadtest ./loadtest

RUN mkdir -p /app/data/chroma /app/data/raw_pdfs && chown -R app:app /app
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

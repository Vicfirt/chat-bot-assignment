FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app \
    HF_HOME=/opt/hf-cache SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache

RUN useradd -m app
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so retrieval needs no runtime download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# After the bake: never touch huggingface.co at runtime, load straight from cache.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY app ./app
COPY eval ./eval
COPY loadtest ./loadtest

RUN mkdir -p /app/data/chroma /app/data/raw_pdfs \
    && chown -R app:app /app /opt/hf-cache
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

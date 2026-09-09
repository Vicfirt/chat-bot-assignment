FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app \
    HF_HOME=/opt/hf-cache SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache

RUN useradd -m app
WORKDIR /app

# requirements.lock is the fully-pinned transitive resolution (uv pip compile);
# requirements.txt is the human-maintained top level.
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# Bake the embedding + reranker models into the image so retrieval needs no
# runtime download.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# After the bake: don't hit huggingface.co at runtime (models are cached).
# Only HF_HUB_OFFLINE — TRANSFORMERS_OFFLINE pushes transformers into an
# offline-load branch that meta-inits the cross-encoder and fails to materialize.
ENV HF_HUB_OFFLINE=1

COPY app ./app
COPY eval ./eval
COPY loadtest ./loadtest

RUN mkdir -p /app/data/chroma /app/data/raw_pdfs \
    && chown -R app:app /app /opt/hf-cache
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

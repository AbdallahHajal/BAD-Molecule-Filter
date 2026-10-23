# Container for the BAD Molecule Filter scoring service.
#
#   docker build -t bad-filter .
#   docker run -p 7860:7860 bad-filter
#
# Python is pinned to 3.11 because mordred 1.2.0, the version the model was built
# with, does not install on 3.12 or later.
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs the container as uid 1000.
RUN useradd -m -u 1000 app
USER app
ENV PATH=/home/app/.local/bin:$PATH \
    HOME=/home/app \
    PYTHONUNBUFFERED=1 \
    BAD_MODEL_PATH=/home/app/model/BAD_MM_model.onnx

WORKDIR /home/app/filter

COPY --chown=app:app requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=app:app bad_filter.py server.py columns_MM.json scaler_MM.npz ./
COPY --chown=app:app docs ./docs

# Fetch the 423 MB model at build time so the first visitor does not wait for it.
RUN mkdir -p /home/app/model \
 && python -c "import bad_filter; bad_filter.download_model()"

EXPOSE 7860
CMD ["python", "server.py", "--preload"]

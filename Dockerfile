FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG OMNIVOICE_COMMIT=5ba967c4d5b0f08244ae856b033eea583d1e4517
ARG APP_VERSION=0.1.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app \
    OMNIVOICE_EXPECTED_VERSION=0.2.1 \
    OMNIVOICE_EXPECTED_TORCH=2.8.0 \
    OMNIVOICE_EXPECTED_TORCHAUDIO=2.8.0 \
    OMNIVOICE_EXPECTED_TRANSFORMERS=5.3.0 \
    OMNIVOICE_EXPECTED_CUDA=12.8 \
    OMNIVOICE_MODEL_DIR=/opt/models/omnivoice \
    OMNIVOICE_MANIFEST=/app/MODEL_MANIFEST.json

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl ffmpeg git libsndfile1 python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/bootstrap \
    && /opt/bootstrap/bin/pip install --no-cache-dir --upgrade pip uv

RUN git clone https://github.com/k2-fsa/OmniVoice.git /opt/omnivoice-src \
    && cd /opt/omnivoice-src \
    && git checkout --detach "$OMNIVOICE_COMMIT" \
    && test "$(git rev-parse HEAD)" = "$OMNIVOICE_COMMIT" \
    && /opt/bootstrap/bin/uv sync --frozen --no-dev

WORKDIR /app
COPY MODEL_MANIFEST.json /app/MODEL_MANIFEST.json
COPY scripts/fetch_model.py /app/scripts/fetch_model.py
COPY scripts/verify_model.py /app/scripts/verify_model.py
COPY app/model_manifest.py /app/app/model_manifest.py
COPY app/__init__.py /app/app/__init__.py

RUN mkdir -p /opt/models/omnivoice \
    && /opt/venv/bin/python /app/scripts/fetch_model.py \
         --manifest /app/MODEL_MANIFEST.json \
         --output /opt/models/omnivoice \
    && /opt/venv/bin/python /app/scripts/verify_model.py \
         --manifest /app/MODEL_MANIFEST.json \
         --model-dir /opt/models/omnivoice \
    && rm -rf /root/.cache/huggingface /opt/models/omnivoice/.cache

COPY app /app/app
COPY scripts /app/scripts
COPY LICENSES /app/LICENSES

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    OMNIVOICE_STARTUP_SMOKE=1

RUN /opt/venv/bin/python - <<'PY'
import importlib.metadata as md
import torch
assert md.version("omnivoice") == "0.2.1", md.version("omnivoice")
assert torch.__version__.startswith("2.8.0"), torch.__version__
assert md.version("torchaudio").startswith("2.8.0"), md.version("torchaudio")
assert md.version("transformers") == "5.3.0", md.version("transformers")
assert str(torch.version.cuda).startswith("12.8"), torch.version.cuda
print("omnivoice", md.version("omnivoice"))
print("torch", torch.__version__)
print("torchaudio", md.version("torchaudio"))
print("transformers", md.version("transformers"))
print("torch_cuda", torch.version.cuda)
PY

EXPOSE 8000

HEALTHCHECK NONE

LABEL org.opencontainers.image.title="Voice Studio OmniVoice Runtime" \
      org.opencontainers.image.version="$APP_VERSION" \
      org.opencontainers.image.source="https://github.com/dilshadshahsyed883-cmd/voice-studio-omnivoice-runtime" \
      ai.omnivoice.upstream.commit="$OMNIVOICE_COMMIT" \
      ai.omnivoice.model.revision="18db15024ce4b7e15638be6ef0e283d99d282f39"

CMD ["/opt/venv/bin/python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

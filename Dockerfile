FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/app/.cache/torch \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN mkdir -p /app/results /app/runs /app/figs /app/data /app/.cache/torch /app/.cache/huggingface

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY . /app

ENV PYTHONPATH=/app

RUN python - <<'PY'
try:
    from torchmetrics.image.inception import InceptionV3
except (ImportError, AttributeError):
    try:
        from torchmetrics.image.fid import InceptionV3
    except (ImportError, AttributeError):
        from torchmetrics.image.fid import FIDInceptionV3 as InceptionV3
InceptionV3(output_blocks=[3], normalize_input=False)
PY

ENTRYPOINT ["python"]
CMD []

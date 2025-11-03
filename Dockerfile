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
from pytorch_fid.inception import InceptionV3
model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]])
PY

ENTRYPOINT ["python"]
CMD []

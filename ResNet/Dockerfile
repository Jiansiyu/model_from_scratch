FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install torchvision --upgrade && \
    python3.10 -m pip install -r requirements.txt


COPY *.py .
COPY config.yaml .

ENV DATA_DIR=/data
ENV OUTPUT_DIR=/output

ENTRYPOINT ["python3.10", "train.py"]
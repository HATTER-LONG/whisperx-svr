FROM nvidia/cuda:11.7.0-base-ubuntu22.04

ENV PYTHON_VERSION=3.10
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    ffmpeg \
    git \
    curl\
    && rm -rf /var/lib/apt/lists/*

RUN ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -s -f /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip setuptools \
    && pip install gunicorn \
    && pip install "fastapi[all]" \
    && pip install git+https://github.com/m-bain/whisperx.git 
# pip install faster-whisper

WORKDIR /app
COPY . /app

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8888", "--workers", "1", "--timeout", "0", "whisperx-app:app", "-k", "uvicorn.workers.UvicornWorker"]

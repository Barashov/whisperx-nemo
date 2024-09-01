FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=8345

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install setuptools wheel
RUN pip3 install Cython packaging

RUN pip3 install \
    pydub \
    torchvision \
    torchaudio \
    fastapi \
    uvicorn \
    python-multipart \
    confluent-kafka \
    omegaconf \
    minio \
    ollama \
    requests

RUN pip3 install transformers==4.39.3 huggingface-hub==0.23.4

RUN pip3 install \
    git+https://github.com/m-bain/whisperx.git

RUN pip3 install faster-whisper==1.0.2

RUN pip3 install \
    "nemo_toolkit[all]" \
    deepmultilingualpunctuation

RUN mkdir -p /app/uploads /app/models

COPY service_diar.py /app/


CMD ["python3", "service_diar.py"]
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
    chatterbox-tts \
    fastapi \
    uvicorn \
    requests \
    torchaudio \
    pydub

COPY handler.py .

CMD ["uvicorn", "handler:app", "--host", "0.0.0.0", "--port", "8080"]

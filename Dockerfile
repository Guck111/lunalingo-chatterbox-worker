FROM runpod/base:0.6.2-cuda12.1.0

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install chatterbox-tts runpod requests torchaudio pydub

COPY handler.py .

CMD ["python3", "-u", "handler.py"]

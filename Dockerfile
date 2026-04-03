FROM runpod/base:0.6.2-cuda12.1.0

RUN pip install chatterbox-tts runpod requests torchaudio

COPY handler.py .

CMD ["python", "-u", "handler.py"]

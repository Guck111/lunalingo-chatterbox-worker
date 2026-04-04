import re
import torch
import torchaudio as ta
import base64
import tempfile
import os
import io
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from pydub import AudioSegment
from chatterbox.tts import ChatterboxTTS

app = FastAPI()
model = ChatterboxTTS.from_pretrained(device="cuda")


def chunk_text(text: str, max_chars: int = 150) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = (current + " " + sentence).strip() if current else sentence
    if current:
        chunks.append(current)
    return chunks


class TTSRequest(BaseModel):
    text: str
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    seed: int = 0
    reference_audio_url: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: TTSRequest):
    if req.seed > 0:
        torch.manual_seed(req.seed)

    audio_prompt_path = None
    if req.reference_audio_url:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(requests.get(req.reference_audio_url).content)
        tmp.close()
        audio_prompt_path = tmp.name

    chunks = chunk_text(req.text)
    combined = AudioSegment.empty()

    for chunk in chunks:
        wav = model.generate(
            chunk,
            audio_prompt_path=audio_prompt_path,
            exaggeration=req.exaggeration,
            cfg_weight=req.cfg_weight,
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            ta.save(wav_file.name, wav, model.sr)
            segment = AudioSegment.from_wav(wav_file.name)
            os.unlink(wav_file.name)
        combined += segment

    if audio_prompt_path:
        os.unlink(audio_prompt_path)

    mp3_buffer = io.BytesIO()
    combined.export(mp3_buffer, format="mp3", bitrate="128k")
    audio_base64 = base64.b64encode(mp3_buffer.getvalue()).decode()

    return {"audio_base64": audio_base64}

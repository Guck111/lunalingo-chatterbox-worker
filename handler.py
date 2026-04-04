import re
import json
import torch
import torchaudio as ta
import base64
import tempfile
import os
import io
import requests
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
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


class BatchTTSRequest(BaseModel):
    requests: list[TTSRequest]


def _generate_single(req: TTSRequest, ref_cache: dict[str, str]) -> str:
    """Generate audio for one request. ref_cache maps URL → local file path."""
    if req.seed > 0:
        torch.manual_seed(req.seed)

    audio_prompt_path = None
    if req.reference_audio_url:
        if req.reference_audio_url not in ref_cache:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(requests.get(req.reference_audio_url).content)
            tmp.close()
            ref_cache[req.reference_audio_url] = tmp.name
        audio_prompt_path = ref_cache[req.reference_audio_url]

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

    mp3_buffer = io.BytesIO()
    combined.export(mp3_buffer, format="mp3", bitrate="128k")
    return base64.b64encode(mp3_buffer.getvalue()).decode()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: TTSRequest):
    ref_cache: dict[str, str] = {}
    try:
        audio_base64 = _generate_single(req, ref_cache)
    finally:
        for path in ref_cache.values():
            try:
                os.unlink(path)
            except OSError:
                pass
    return {"audio_base64": audio_base64}


@app.post("/batch")
def batch_generate(req: BatchTTSRequest):
    """Process multiple TTS requests sequentially on one GPU.
    Streams NDJSON — one JSON line per completed item — so the connection
    stays alive and the client receives results as they are ready."""
    if not req.requests:
        return {"results": []}

    def stream():
        ref_cache: dict[str, str] = {}
        try:
            for item in req.requests:
                audio_base64 = _generate_single(item, ref_cache)
                yield json.dumps({"audio_base64": audio_base64}) + "\n"
        finally:
            for path in ref_cache.values():
                try:
                    os.unlink(path)
                except OSError:
                    pass

    return StreamingResponse(stream(), media_type="application/x-ndjson")

import os

# Redirect HuggingFace model cache to RunPod volume so weights are downloaded
# only on the first start and reused on subsequent restarts.
os.environ["HF_HOME"] = "/runpod-volume/hf_cache"

import re
import json
import queue
import threading
import torch
import torchaudio as ta
import base64
import tempfile
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

# Reference audio is pre-loaded from env at startup into the RunPod volume.
# Path: /runpod-volume/reference.wav — persists across restarts.
VOLUME_REF_PATH = "/runpod-volume/reference.wav"
_startup_ref_path: Optional[str] = None


@app.on_event("startup")
def preload_reference_audio():
    global _startup_ref_path
    url = os.environ.get("REFERENCE_AUDIO_URL")
    if not url:
        return
    if os.path.exists(VOLUME_REF_PATH):
        print(f"[startup] reference audio already on volume: {VOLUME_REF_PATH}")
        _startup_ref_path = VOLUME_REF_PATH
        return
    print(f"[startup] downloading reference audio from {url}")
    os.makedirs("/runpod-volume", exist_ok=True)
    data = requests.get(url, timeout=30).content
    with open(VOLUME_REF_PATH, "wb") as f:
        f.write(data)
    print(f"[startup] reference audio saved to {VOLUME_REF_PATH} ({len(data)} bytes)")
    _startup_ref_path = VOLUME_REF_PATH


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


def _resolve_ref_path(url: Optional[str], ref_cache: dict[str, str]) -> Optional[str]:
    """Return local path for reference audio. Uses volume cache when available."""
    if not url:
        return None
    # If the URL matches the pre-loaded startup reference, use it directly
    if _startup_ref_path and url == os.environ.get("REFERENCE_AUDIO_URL"):
        return _startup_ref_path
    if url not in ref_cache:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(requests.get(url, timeout=30).content)
        tmp.close()
        ref_cache[url] = tmp.name
    return ref_cache[url]


def _generate_single(req: TTSRequest, ref_cache: dict[str, str]) -> str:
    """Generate audio for one request. Returns base64-encoded MP3."""
    if req.seed > 0:
        torch.manual_seed(req.seed)

    audio_prompt_path = _resolve_ref_path(req.reference_audio_url, ref_cache)

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
        for url, path in ref_cache.items():
            if path != _startup_ref_path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
    return {"audio_base64": audio_base64}


@app.post("/batch")
def batch_generate(req: BatchTTSRequest):
    """Process multiple TTS requests sequentially on one GPU.
    Streams NDJSON — one JSON line per completed item.
    Sends heartbeat newlines every 10 s to keep the proxy connection alive
    while the GPU is busy."""
    if not req.requests:
        return {"results": []}

    result_queue: queue.Queue[tuple[str, str]] = queue.Queue()

    def worker():
        ref_cache: dict[str, str] = {}
        try:
            for item in req.requests:
                audio_base64 = _generate_single(item, ref_cache)
                result_queue.put(("result", audio_base64))
        except Exception as exc:
            result_queue.put(("error", str(exc)))
        finally:
            result_queue.put(("done", ""))
            for url, path in ref_cache.items():
                if path != _startup_ref_path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def stream():
        while True:
            try:
                msg_type, value = result_queue.get(timeout=10)
            except queue.Empty:
                # Heartbeat — keeps proxy from closing idle connection
                yield "\n"
                continue
            if msg_type == "result":
                yield json.dumps({"audio_base64": value}) + "\n"
            elif msg_type == "error":
                yield json.dumps({"error": value}) + "\n"
                break
            elif msg_type == "done":
                break
        thread.join()

    return StreamingResponse(stream(), media_type="application/x-ndjson")

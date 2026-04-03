import runpod
import torch
import torchaudio as ta
import base64
import tempfile
import os
import requests
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")

def handler(event):
    input = event["input"]
    text = input["text"]
    exaggeration = input.get("exaggeration", 0.5)
    cfg_weight = input.get("cfg_weight", 0.5)
    seed = input.get("seed", 0)
    reference_audio_url = input.get("reference_audio_url", None)

    if seed > 0:
        torch.manual_seed(seed)

    audio_prompt_path = None
    if reference_audio_url:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(requests.get(reference_audio_url).content)
        tmp.close()
        audio_prompt_path = tmp.name

    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        ta.save(f.name, wav, model.sr)
        audio_base64 = base64.b64encode(open(f.name, "rb").read()).decode()

    if audio_prompt_path:
        os.unlink(audio_prompt_path)

    return {"audio_base64": audio_base64}

runpod.serverless.start({"handler": handler})

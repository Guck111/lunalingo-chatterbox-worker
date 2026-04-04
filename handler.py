import re
  import runpod
  import torch
  import torchaudio as ta
  import base64
  import tempfile
  import os
  import io
  import requests
  from pydub import AudioSegment
  from chatterbox.tts import ChatterboxTTS

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

      chunks = chunk_text(text)
      combined = AudioSegment.empty()

      for chunk in chunks:
          wav = model.generate(
              chunk,
              audio_prompt_path=audio_prompt_path,
              exaggeration=exaggeration,
              cfg_weight=cfg_weight,
          )
          with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
              ta.save(wav_file.name, wav, model.sr)
              segment = AudioSegment.from_wav(wav_file.name)
              os.unlink(wav_file.name)
          combined += segment

      mp3_buffer = io.BytesIO()
      combined.export(mp3_buffer, format="mp3", bitrate="128k")
      audio_base64 = base64.b64encode(mp3_buffer.getvalue()).decode()

      if audio_prompt_path:
          os.unlink(audio_prompt_path)

      return {"audio_base64": audio_base64}


  runpod.serverless.start({"handler": handler})

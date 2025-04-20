# backend/main.py

import os
import uuid
import io
import json
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

# --- Safe‑serialization patch for XTTS configs ---
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# Allowlist classes so torch.load(..., weights_only=True) works
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])

# Google Cloud Speech-to-Text
from google.cloud import speech_v1p1beta1 as speech
# Ollama Python client
from ollama import chat
# Coqui XTTS v2
from TTS.api import TTS
import numpy as np
from scipy.io.wavfile import write

# Load environment variables
load_dotenv()

app = FastAPI(title="Voice-Chat Backend with SSE")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
speech_client = speech.SpeechClient()
tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    progress_bar=False,
    gpu=False,
)
if torch.cuda.is_available():
    tts.to("cuda")
    torch.cuda.empty_cache()

DUA_PROMPT = os.path.join(os.path.dirname(__file__), "models", "dua_voice.mp3")

def transcribe_wav(path: str) -> str:
    with open(path, "rb") as f:
        audio = speech.RecognitionAudio(content=f.read())
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )
    resp = speech_client.recognize(config=config, audio=audio)
    return " ".join(r.alternatives[0].transcript for r in resp.results)

def llama_stream(messages):
    for chunk in chat(
        model="llama3.2:1b",
        messages=messages,
        stream=True,
    ):
        yield chunk["message"]["content"]

def tts_wav_stream(text: str):
    wav = tts.tts(
        text=text,
        speaker=None,
        language="en",
        speaker_wav=DUA_PROMPT,
        split_sentences=True,
    )
    arr = np.array(wav)
    if arr.dtype in (np.float32, np.float64):
        arr = (arr * 32767).astype(np.int16)
    else:
        arr = arr.astype(np.int16)
    buf = io.BytesIO()
    sr = tts.synthesizer.output_sample_rate
    write(buf, sr, arr)
    data = buf.getvalue()
    chunk_size = 4096
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]

@app.post("/transcribe/")
async def upload_and_transcribe(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, detail="Please upload a .wav file")
    tmp = f"/tmp/{uuid.uuid4()}.wav"
    with open(tmp, "wb") as f:
        f.write(await file.read())
    return {"transcript": transcribe_wav(tmp)}

@app.get("/stream_chat/")
async def stream_chat(
    request: Request,
    prompt: str,
    history: str = Query("", description="URL‑encoded JSON array of prior messages"),
):
    try:
        msgs = json.loads(history) if history else []
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid history JSON")
    msgs.append({"role": "user", "content": prompt})

    async def event_gen():
        for tok in llama_stream(msgs):
            if await request.is_disconnected():
                break
            yield {"event": "message", "data": tok}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(
        event_gen(),
        ping=15,
        headers={"Cache-Control": "no-cache"},
    )

@app.get("/stream_tts/")
async def stream_tts(text: str):
    return StreamingResponse(
        tts_wav_stream(text),
        media_type="audio/wav",
    )
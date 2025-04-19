# backend/main.py

import os
import uuid
import io

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
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

# Ollama Python client (official)
from ollama import chat

# Coqui XTTS v2
from TTS.api import TTS
import numpy as np
from scipy.io.wavfile import write

# Load environment variables (e.g. GOOGLE_APPLICATION_CREDENTIALS)
load_dotenv()

app = FastAPI(title="Voice-Chat Backend with SSE")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # your front‑end origin
    allow_credentials=True,
    allow_methods=["*"],                       # or restrict to ["GET","POST"]
    allow_headers=["*"],                       # or list specific headers
)

# 1) Google Speech-to-Text client
speech_client = speech.SpeechClient()

# 2) Coqui XTTS v2, CPU‑only at first
tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    progress_bar=False,
    gpu=False,
)
# If you do have CUDA, move it over
if torch.cuda.is_available():
    tts.to("cuda")
    torch.cuda.empty_cache()

# Path to your Dua Lipa sample WAV for voice cloning
DUA_PROMPT = os.path.join(os.path.dirname(__file__), "models", "dua_voice.mp3")


def transcribe_wav(path: str) -> str:
    """Synchronously reads a LINEAR16 WAV file and returns the transcription."""
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


def llama_stream(prompt: str):
    """
    Synchronous generator yielding each token from llama3.2:1b via Ollama streaming.
    """
    for chunk in chat(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        yield chunk["message"]["content"]


def tts_wav_stream(text: str):
    """
    Generator yielding WAV byte‐chunks from Coqui XTTS v2.
    We call the synchronous `.tts()` API and then chunk its output.
    """
    # 1) Synthesize the entire waveform (list of samples)
    wav = tts.tts(
        text=text,
        speaker=None,
        language="en",
        speaker_wav=DUA_PROMPT,
        split_sentences=True,
    )
    # 2) Convert to numpy array
    arr = np.array(wav)
    # If floats in [-1,1], scale to int16
    if arr.dtype in (np.float32, np.float64):
        arr = (arr * 32767).astype(np.int16)
    else:
        arr = arr.astype(np.int16)
    # 3) Write WAV header + data into a buffer
    buf = io.BytesIO()
    sr = tts.synthesizer.output_sample_rate  # sample rate from the synthesizer
    write(buf, sr, arr)
    data = buf.getvalue()
    # 4) Stream it in 4 kB chunks
    chunk_size = 4096
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


@app.post("/transcribe/")
async def upload_and_transcribe(file: UploadFile = File(...)):
    """
    Upload a .wav file (LINEAR16 @ 48 kHz) and return the transcription.
    """
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, detail="Please upload a .wav file")
    tmp = f"/tmp/{uuid.uuid4()}.wav"
    with open(tmp, "wb") as f:
        f.write(await file.read())
    return {"transcript": transcribe_wav(tmp)}


@app.get("/stream_chat/")
async def stream_chat(request: Request, prompt: str):
    """
    SSE endpoint: streams each LLM token as a separate SSE 'message' event.
    Automatically stops if client disconnects.
    """
    async def event_gen():
        for tok in llama_stream(prompt):
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
    """
    Stream WAV chunks for TTS synthesis.
    Returns a chunked response with media_type 'audio/wav'.
    """
    return StreamingResponse(
        tts_wav_stream(text),
        media_type="audio/wav",
    )

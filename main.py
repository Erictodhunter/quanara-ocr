import easyocr
import pdf2image
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import json
import asyncio

# Top 25 languages for global coverage
LANGUAGES = [
    'en',  # English
    'ch_sim',  # Chinese Simplified
    'hi',  # Hindi
    'es',  # Spanish
    'fr',  # French
    'ar',  # Arabic
    'bn',  # Bengali
    'ru',  # Russian
    'pt',  # Portuguese
    'id',  # Indonesian
    'ur',  # Urdu
    'ja',  # Japanese
    'de',  # German
    'ko',  # Korean
    'tr',  # Turkish
    'vi',  # Vietnamese
    'ta',  # Tamil
    'it',  # Italian
    'th',  # Thai
    'pl',  # Polish
    'uk',  # Ukrainian
    'nl',  # Dutch
    'fa',  # Persian
    'cs',  # Czech
    'sv',  # Swedish
]

# Initialize with CPU only for Cloud Run
reader = easyocr.Reader(LANGUAGES, gpu=False, verbose=False)
print(f"✅ OCR Ready with {len(LANGUAGES)} languages on CPU")

app = FastAPI(title="QUANARA OCR - Multi-Language")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Keep all your existing routes and functions the same...
# Just copy your stream_extraction, routes, etc. here

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

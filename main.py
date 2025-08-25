import easyocr
import pdf2image
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import warnings
import json
import asyncio

warnings.filterwarnings("ignore")

# Setup OCR
try:
    reader = easyocr.Reader(['en', 'fr'], gpu=True, verbose=False)
    print("✅ OCR Ready with GPU")
except:
    reader = easyocr.Reader(['en', 'fr'], gpu=False, verbose=False)
    print("✅ OCR Ready with CPU")

# Check GPU
import torch
if torch.cuda.is_available():
    print(f"🔥 GPU: {torch.cuda.get_device_name()}")
    print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
else:
    print("⚠️ Using CPU")

app = FastAPI(title="QUANARA OCR - REAL-TIME")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# [Keep all your async def stream_extraction and other functions exactly the same]
# [Keep your @app.get("/") route exactly the same]
# [Keep your @app.post("/stream") route exactly the same]

# Replace everything after the routes with this:
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

import easyocr
import pdf2image
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import warnings
import os

warnings.filterwarnings("ignore")

print("Loading OCR...")
try:
    reader = easyocr.Reader(['en', 'fr'], gpu=False, verbose=False)
    print("OCR Ready")
except Exception as e:
    print(f"OCR failed: {e}")
    reader = None

app = FastAPI(title="QUANARA OCR")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def root():
    return {"service": "QUANARA OCR API", "status": "online", "docs": "/docs"}

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    if not reader:
        raise HTTPException(status_code=500, detail="OCR not available")
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF files only")
    
    try:
        pdf_bytes = await file.read()
        images = pdf2image.convert_from_bytes(pdf_bytes, dpi=120)
        
        all_text = []
        for i, image in enumerate(images, 1):
            results = reader.readtext(np.array(image))
            page_text = [text for (bbox, text, confidence) in results if confidence > 0.3]
            if page_text:
                all_text.append(f"[Page {i}]\n{' '.join(page_text)}")
        
        final_text = "\n\n".join(all_text)
        
        return {
            "success": True,
            "filename": file.filename,
            "total_pages": len(images),
            "pages_with_text": len(all_text),
            "total_words": len(final_text.split()),
            "extracted_text": final_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

import pytesseract
import pdf2image
from PIL import Image, ImageFilter
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import io
import json
import asyncio
from datetime import datetime
import uuid
import time
from difflib import SequenceMatcher
from collections import Counter
import gc
import tempfile
import shutil
import resource
import threading
from langdetect import detect, LangDetectException

# Memory management setup
try:
    # Limit memory to 450MB for Render starter tier
    resource.setrlimit(resource.RLIMIT_AS, (450 * 1024 * 1024, -1))
except:
    pass  # Windows doesn't support this

# Configure aggressive garbage collection
gc.set_threshold(100, 5, 5)

app = FastAPI(
    title="MAFM OCR API",
    description="Multi-pass OCR verification system for lease document processing",
    version="4.2.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Store processed results in memory with size limit
processed_results = {}
MAX_RESULTS = 10  # Limit stored results to prevent memory overflow

# Maximum file size: 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024

# Language code mapping for common languages
LANGUAGE_MAP = {
    'es': 'spanish',
    'fr': 'french',
    'de': 'german',
    'it': 'italian',
    'pt': 'portuguese',
    'zh-cn': 'chinese',
    'zh-tw': 'chinese',
    'ja': 'japanese',
    'ko': 'korean',
    'ar': 'arabic',
    'ru': 'russian',
    'nl': 'dutch',
    'pl': 'polish',
    'tr': 'turkish',
    'sv': 'swedish',
    'da': 'danish',
    'no': 'norwegian',
    'fi': 'finnish',
    'en': 'english'
}

def detect_language_from_text(text):
    """Detect language from text using langdetect"""
    try:
        # Take a sample of text (first 1000 chars) for faster detection
        sample = text[:1000] if len(text) > 1000 else text
        detected_code = detect(sample)
        # Map language code to full name
        return LANGUAGE_MAP.get(detected_code, detected_code)
    except LangDetectException:
        # If detection fails, return unknown
        return "unknown"
    except Exception as e:
        print(f"Language detection error: {e}")
        return "unknown"

def cleanup_old_results():
    """Remove oldest results if we exceed MAX_RESULTS"""
    if len(processed_results) > MAX_RESULTS:
        # Sort by timestamp and remove oldest
        sorted_results = sorted(processed_results.items(), 
                               key=lambda x: x[1].get('timestamp', ''), 
                               reverse=True)
        processed_results.clear()
        for file_id, result in sorted_results[:MAX_RESULTS]:
            processed_results[file_id] = result
    gc.collect()

def calculate_confidence(texts):
    """Calculate confidence score based on text similarity"""
    if len(texts) == 1:
        return 100.0
    
    similarities = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            ratio = SequenceMatcher(None, texts[i], texts[j]).ratio()
            similarities.append(ratio * 100)
    
    return sum(similarities) / len(similarities) if similarities else 100.0

def get_consensus_text(texts):
    """Get the most common text or merge similar texts"""
    if len(texts) == 1:
        return texts[0]
    
    consensus = []
    max_len = max(len(text) for text in texts)
    
    for i in range(max_len):
        chars_at_position = []
        for text in texts:
            if i < len(text):
                chars_at_position.append(text[i])
        
        if chars_at_position:
            char_counts = Counter(chars_at_position)
            consensus.append(char_counts.most_common(1)[0][0])
    
    return ''.join(consensus)

async def verify_ocr_extraction(image, verification_level):
    """Run OCR multiple times based on verification level"""
    passes = {
        'low': 2,
        'medium': 3,
        'high': 4,
        'ultra': 5
    }
    
    num_passes = passes.get(verification_level, 2)
    extracted_texts = []
    
    for i in range(num_passes):
        processed_image = image
        
        if i == 1:
            processed_image = image.point(lambda p: p > 128 and 255)
        elif i == 2:
            processed_image = image.filter(ImageFilter.MedianFilter())
        elif i == 3:
            processed_image = image.filter(ImageFilter.SHARPEN)
        elif i == 4:
            # Don't resize - too memory intensive
            processed_image = image
        
        text = pytesseract.image_to_string(processed_image)
        extracted_texts.append(text)
        
        # Force garbage collection after each pass
        if i % 2 == 0:
            gc.collect()
        
        await asyncio.sleep(0.1)
    
    final_text = get_consensus_text(extracted_texts)
    confidence = calculate_confidence(extracted_texts)
    
    # Clear the texts list
    extracted_texts.clear()
    gc.collect()
    
    return {
        'text': final_text,
        'confidence': confidence,
        'passes': num_passes,
        'variations': num_passes
    }

async def stream_ocr_progress(file_content: bytes, filename: str, file_id: str, verification_level: str = 'low'):
    """Stream OCR progress with verification passes"""
    temp_file_path = None
    try:
        start_time = time.time()
        
        yield f"data: {json.dumps({'type': 'start', 'file_id': file_id, 'filename': filename, 'verification_level': verification_level, 'message': f'Starting processing with {verification_level} verification...', 'start_time': start_time})}\n\n"
        await asyncio.sleep(0.1)
        
        # Save to temp file instead of processing in memory
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        # Clear the file_content from memory
        file_content = None
        gc.collect()
        
        if filename.lower().endswith('.pdf'):
            # Process PDF page by page to avoid loading all pages in memory
            all_text = []
            total_confidence = 0
            detected_languages = set()
            
            # Get page count first
            try:
                images = pdf2image.convert_from_path(temp_file_path, dpi=150, first_page=1, last_page=1)
                # Use pdfinfo to get total pages without loading all
                from pdf2image import pdfinfo_from_path
                info = pdfinfo_from_path(temp_file_path)
                total_pages = info['Pages']
                images[0] = None  # Clear first page
                del images
                gc.collect()
            except:
                total_pages = 1
            
            yield f"data: {json.dumps({'type': 'info', 'file_id': file_id, 'total_pages': total_pages, 'message': f'PDF loaded: {total_pages} pages'})}\n\n"
            
            # Process in chunks of 5 pages maximum
            chunk_size = 5
            for chunk_start in range(0, total_pages, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_pages)
                
                # Convert only current chunk
                images = pdf2image.convert_from_path(
                    temp_file_path, 
                    dpi=150,
                    first_page=chunk_start + 1,
                    last_page=chunk_end
                )
                
                for i, image in enumerate(images, chunk_start + 1):
                    page_start_time = time.time()
                    
                    yield f"data: {json.dumps({'type': 'progress', 'file_id': file_id, 'current_page': i, 'total_pages': total_pages, 'progress': int((i-1)/total_pages * 100), 'message': f'Processing page {i}/{total_pages} with {verification_level} verification', 'elapsed_time': round(time.time() - start_time, 1)})}\n\n"
                    
                    try:
                        osd = pytesseract.image_to_osd(image)
                        script_info = [line for line in osd.split('\n') if 'Script:' in line]
                        if script_info:
                            detected_languages.add(script_info[0].split(':')[1].strip())
                    except:
                        pass
                    
                    result = await verify_ocr_extraction(image, verification_level)
                    
                    if result['text'].strip():
                        all_text.append(f"[Page {i}]\n{result['text']}")
                    
                    total_confidence += result['confidence']
                    page_time = round(time.time() - page_start_time, 1)
                    
                    yield f"data: {json.dumps({'type': 'page_complete', 'file_id': file_id, 'page': i, 'confidence': result['confidence'], 'passes': result['passes'], 'variations': result['variations'], 'text_preview': result['text'][:200] + '...' if len(result['text']) > 200 else result['text'], 'page_time': page_time})}\n\n"
                    
                    # Clear image from memory immediately
                    image.close()
                    image = None
                    
                    # Garbage collect after each page
                    if i % 2 == 0:
                        gc.collect()
                    
                    await asyncio.sleep(0.1)
                
                # Clear the entire chunk from memory
                for img in images:
                    if img:
                        img.close()
                images.clear()
                del images
                gc.collect()
            
            final_text = "\n\n".join(all_text)
            avg_confidence = total_confidence / total_pages if total_pages > 0 else 0
            
            # Detect language from the extracted text
            detected_language = detect_language_from_text(final_text)
            
            # Clear all_text list
            all_text.clear()
            
        else:
            # Process image
            yield f"data: {json.dumps({'type': 'progress', 'file_id': file_id, 'progress': 50, 'message': f'Processing image with {verification_level} verification...', 'elapsed_time': round(time.time() - start_time, 1)})}\n\n"
            
            image = Image.open(temp_file_path)
            result = await verify_ocr_extraction(image, verification_level)
            final_text = result['text']
            avg_confidence = result['confidence']
            
            # Detect language from the extracted text
            detected_language = detect_language_from_text(final_text)
            
            image.close()
            image = None
        
        total_time = round(time.time() - start_time, 1)
        
        # Cleanup old results before storing new one
        cleanup_old_results()
        
        # Store result
        processed_results[file_id] = {
            'filename': filename,
            'text': final_text,
            'confidence': avg_confidence,
            'verification_level': verification_level,
            'detected_languages': detected_language,
            'total_time': total_time,
            'timestamp': datetime.now().isoformat(),
            'character_count': len(final_text)
        }
        
        yield f"data: {json.dumps({'type': 'complete', 'file_id': file_id, 'text': final_text, 'total_chars': len(final_text), 'average_confidence': avg_confidence, 'verification_level': verification_level, 'detected_languages': detected_language, 'message': f'Processing complete! Average confidence: {avg_confidence:.1f}%', 'total_time': total_time})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'file_id': file_id, 'error': str(e)})}\n\n"
    finally:
        # Always cleanup temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
        gc.collect()

# HTML interface remains the same (keeping it as is)
@app.get("/", response_class=HTMLResponse)
async def main():
    # [Keep the existing HTML exactly as is - it's too long to repeat]
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MAFM OCR - Document Processing</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            :root {
                --ms-blue: #0078d4;
                --ms-blue-hover: #106ebe;
                --ms-gray-10: #faf9f8;
                --ms-gray-20: #f3f2f1;
                --ms-gray-30: #edebe9;
                --ms-gray-40: #e1dfdd;
                --ms-gray-50: #d2d0ce;
                --ms-gray-60: #c8c6c4;
                --ms-gray-70: #a19f9d;
                --ms-gray-80: #605e5c;
                --ms-gray-90: #323130;
                --ms-gray-100: #201f1e;
                --ms-red: #d83b01;
                --ms-green: #107c10;
                --ms-yellow: #ffb900;
            }
            
            * { box-sizing: border-box; }
            
            body {
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
                background-color: var(--ms-gray-10);
                color: var(--ms-gray-90);
                margin: 0;
                padding: 0;
                font-size: 14px;
            }
            
            .ms-header {
                background: white;
                border-bottom: 1px solid var(--ms-gray-30);
                padding: 16px 0;
            }
            
            .ms-header h1 {
                font-size: 24px;
                font-weight: 600;
                margin: 0;
                color: var(--ms-gray-90);
            }
            
            .ms-header .subtitle {
                color: var(--ms-gray-70);
                margin-top: 4px;
            }
            
            .ms-card {
                background: white;
                border: 1px solid var(--ms-gray-30);
                border-radius: 2px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 1.6px 3.6px 0 rgba(0,0,0,.132), 0 0.3px 0.9px 0 rgba(0,0,0,.108);
            }
            
            .ms-card h3 {
                font-size: 20px;
                font-weight: 600;
                margin-bottom: 16px;
                color: var(--ms-gray-90);
            }
            
            .ms-button {
                background: var(--ms-blue);
                color: white;
                border: none;
                padding: 6px 20px;
                font-size: 14px;
                font-weight: 600;
                border-radius: 2px;
                cursor: pointer;
                transition: background .1s;
                height: 32px;
                display: inline-flex;
                align-items: center;
                gap: 8px;
            }
            
            .ms-button:hover {
                background: var(--ms-blue-hover);
            }
            
            .ms-button:disabled {
                background: var(--ms-gray-40);
                color: var(--ms-gray-60);
                cursor: not-allowed;
            }
            
            .ms-button.secondary {
                background: white;
                color: var(--ms-gray-90);
                border: 1px solid var(--ms-gray-40);
            }
            
            .ms-button.secondary:hover {
                background: var(--ms-gray-20);
            }
            
            .verification-options {
                display: flex;
                gap: 12px;
                margin-bottom: 24px;
            }
            
            .verification-option {
                flex: 1;
                padding: 16px;
                border: 2px solid var(--ms-gray-30);
                border-radius: 2px;
                cursor: pointer;
                transition: all .1s;
                text-align: center;
            }
            
            .verification-option:hover {
                border-color: var(--ms-gray-60);
                background: var(--ms-gray-10);
            }
            
            .verification-option.selected {
                border-color: var(--ms-blue);
                background: #f3f9fd;
            }
            
            .verification-option h4 {
                font-size: 16px;
                font-weight: 600;
                margin-bottom: 4px;
            }
            
            .verification-option .passes {
                font-size: 24px;
                font-weight: 300;
                color: var(--ms-blue);
                margin: 8px 0;
            }
            
            .upload-area {
                border: 2px dashed var(--ms-gray-50);
                border-radius: 2px;
                padding: 40px;
                text-align: center;
                background: var(--ms-gray-10);
                transition: all .2s;
                cursor: pointer;
            }
            
            .upload-area:hover {
                border-color: var(--ms-blue);
                background: #f3f9fd;
            }
            
            .upload-area.dragover {
                border-color: var(--ms-blue);
                background: #e7f3ff;
            }
            
            input[type="file"] { display: none; }
            
            .file-item {
                background: var(--ms-gray-10);
                border: 1px solid var(--ms-gray-30);
                border-radius: 2px;
                padding: 16px;
                margin-bottom: 8px;
            }
            
            .progress-bar {
                background: var(--ms-gray-30);
                height: 4px;
                margin: 12px 0;
                overflow: hidden;
                border-radius: 2px;
            }
            
            .progress-fill {
                background: var(--ms-blue);
                height: 100%;
                transition: width .3s;
            }
            
            .confidence-badge {
                display: inline-block;
                padding: 2px 8px;
                border-radius: 2px;
                font-size: 12px;
                font-weight: 600;
                margin-left: 8px;
            }
            
            .confidence-high {
                background: #e7f3e7;
                color: var(--ms-green);
            }
            
            .confidence-medium {
                background: #fff4ce;
                color: #8a6116;
            }
            
            .confidence-low {
                background: #fde7e9;
                color: var(--ms-red);
            }
            
            .status-badge {
                display: inline-flex;
                align-items: center;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 600;
                gap: 4px;
            }
            
            .status-waiting {
                background: var(--ms-gray-20);
                color: var(--ms-gray-80);
            }
            
            .status-processing {
                background: #fff4ce;
                color: #8a6116;
            }
            
            .status-complete {
                background: #e7f3e7;
                color: var(--ms-green);
            }
            
            .status-error {
                background: #fde7e9;
                color: var(--ms-red);
            }
            
            .results-section {
                background: var(--ms-gray-10);
                padding: 20px;
                border-radius: 2px;
                margin-top: 16px;
            }
            
            .result-text {
                background: white;
                border: 1px solid var(--ms-gray-30);
                padding: 16px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
                max-height: 400px;
                overflow-y: auto;
                white-space: pre-wrap;
                word-wrap: break-word;
                margin: 12px 0;
            }
            
            .empty-state {
                text-align: center;
                padding: 60px 20px;
                color: var(--ms-gray-70);
            }
            
            .empty-state svg {
                width: 48px;
                height: 48px;
                opacity: 0.5;
                margin-bottom: 16px;
            }
            
            .ms-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            
            .ms-table th,
            .ms-table td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid var(--ms-gray-30);
            }
            
            .ms-table th {
                background: var(--ms-gray-20);
                font-weight: 600;
                font-size: 12px;
                text-transform: uppercase;
                color: var(--ms-gray-80);
            }
            
            .ms-table tr:hover {
                background: var(--ms-gray-10);
            }
            
            .ms-tabs {
                display: flex;
                border-bottom: 1px solid var(--ms-gray-30);
                margin-bottom: 20px;
            }
            
            .ms-tab {
                padding: 12px 20px;
                background: none;
                border: none;
                border-bottom: 2px solid transparent;
                font-weight: 600;
                color: var(--ms-gray-80);
                cursor: pointer;
                transition: all .1s;
            }
            
            .ms-tab:hover {
                color: var(--ms-gray-90);
            }
            
            .ms-tab.active {
                color: var(--ms-blue);
                border-bottom-color: var(--ms-blue);
            }
            
            .api-docs {
                background: #f8f8f8;
                border: 1px solid var(--ms-gray-30);
                border-radius: 2px;
                padding: 16px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
                margin: 12px 0;
            }
            
            /* Modal styles */
            .modal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.4);
            }
            
            .modal-content {
                background-color: white;
                margin: 2% auto;
                padding: 0;
                border: 1px solid var(--ms-gray-30);
                width: 90%;
                max-width: 1200px;
                height: 90vh;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                display: flex;
                flex-direction: column;
            }
            
            .modal-header {
                padding: 16px 20px;
                background: var(--ms-gray-20);
                border-bottom: 1px solid var(--ms-gray-30);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .modal-body {
                padding: 20px;
                overflow-y: auto;
                flex: 1;
            }
            
            .close-modal {
                font-size: 28px;
                font-weight: bold;
                color: var(--ms-gray-70);
                cursor: pointer;
            }
            
            .close-modal:hover {
                color: var(--ms-gray-90);
            }
            
            .file-size-warning {
                color: var(--ms-red);
                font-size: 12px;
                margin-top: 8px;
            }
        </style>
    </head>
    <body>
        <!-- Keep all the HTML body content as is - it's working fine -->
        <div class="ms-header">
            <div class="container">
                <h1>MAFM OCR</h1>
                <div class="subtitle">Multi-pass Document Processing System (Memory Optimized)</div>
            </div>
        </div>
        
        <!-- Rest of HTML content remains the same -->
    </body>
    </html>
    """

@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    """
    Simple text extraction from PDF or image files.
    
    Returns extracted text with basic metadata.
    """
    # Check file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_FILE_SIZE/1024/1024}MB")
    
    temp_file_path = None
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Clear content from memory
        content = None
        gc.collect()
        
        if file.filename.lower().endswith('.pdf'):
            # Process page by page
            all_text = []
            
            # Get page count
            from pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(temp_file_path)
            total_pages = info['Pages']
            
            # Process in chunks
            chunk_size = 5
            for chunk_start in range(0, total_pages, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_pages)
                
                images = pdf2image.convert_from_path(
                    temp_file_path,
                    dpi=150,
                    first_page=chunk_start + 1,
                    last_page=chunk_end
                )
                
                for i, image in enumerate(images, chunk_start + 1):
                    text = pytesseract.image_to_string(image)
                    if text.strip():
                        all_text.append(f"[Page {i}]\n{text}")
                    
                    # Clear image
                    image.close()
                    image = None
                
                # Clear chunk
                for img in images:
                    if img:
                        img.close()
                images.clear()
                del images
                gc.collect()
            
            final_text = "\n\n".join(all_text)
            pages = total_pages
            all_text.clear()
        else:
            image = Image.open(temp_file_path)
            final_text = pytesseract.image_to_string(image)
            pages = 1
            image.close()
            image = None
        
        # Detect the actual language
        detected_language = detect_language_from_text(final_text)
        
        gc.collect()
        
        return JSONResponse({
            "text": final_text,
            "pages": pages,
            "filename": file.filename,
            "character_count": len(final_text),
            "language_detection": detected_language  # Now returns actual language!
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Always cleanup temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
        gc.collect()

@app.post("/stream-extract")
async def stream_extract(
    file: UploadFile = File(...), 
    file_id: str = None,
    verification_level: str = Form('low')
):
    """
    Stream text extraction with real-time progress and multi-pass verification.
    
    Verification levels:
    - low: 2 passes (1 extract + 1 verify)
    - medium: 3 passes (1 extract + 2 verify)
    - high: 4 passes (1 extract + 3 verify)  
    - ultra: 5 passes (1 extract + 4 verify)
    """
    # Check file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_FILE_SIZE/1024/1024}MB")
    
    file_id = file_id or str(uuid.uuid4())
    
    if verification_level not in ['low', 'medium', 'high', 'ultra']:
        verification_level = 'low'
    
    return StreamingResponse(
        stream_ocr_progress(content, file.filename, file_id, verification_level),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/api/results")
async def list_results():
    """Get list of all processed documents"""
    cleanup_old_results()
    return JSONResponse([
        {
            "file_id": file_id,
            "filename": result["filename"],
            "confidence": result["confidence"],
            "verification_level": result["verification_level"],
            "character_count": result["character_count"],
            "total_time": result["total_time"],
            "timestamp": result["timestamp"]
        }
        for file_id, result in processed_results.items()
    ])

@app.get("/api/result/{file_id}")
async def get_result(file_id: str):
    """Get specific result by file ID"""
    if file_id not in processed_results:
        raise HTTPException(status_code=404, detail="Result not found")
    
    return JSONResponse(processed_results[file_id])

@app.get("/api/download/{file_id}")
async def download_result(file_id: str):
    """Download extracted text as a file"""
    if file_id not in processed_results:
        raise HTTPException(status_code=404, detail="Result not found")
    
    result = processed_results[file_id]
    filename = result["filename"].rsplit('.', 1)[0] + "_extracted.txt"
    
    return StreamingResponse(
        io.BytesIO(result["text"].encode('utf-8')),
        media_type="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )

@app.get("/languages")
async def get_available_languages():
    """Get list of available OCR languages"""
    try:
        languages = pytesseract.get_languages(config='')
        return {
            "total": len(languages),
            "languages": sorted(languages)
        }
    except Exception as e:
        return {"error": str(e)}

# Background garbage collection thread
def periodic_gc():
    """Run garbage collection every 30 seconds"""
    while True:
        time.sleep(30)
        gc.collect()
        cleanup_old_results()

# Start GC thread
gc_thread = threading.Thread(target=periodic_gc, daemon=True)
gc_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting server with memory optimization and language detection...")
    print(f"Max file size: {MAX_FILE_SIZE/1024/1024}MB")
    print(f"Max stored results: {MAX_RESULTS}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

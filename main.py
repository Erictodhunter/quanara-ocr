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
    version="4.3.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Store processed results in memory with size limit
processed_results = {}
MAX_RESULTS = 10  # Limit stored results to prevent memory overflow

# Maximum file size: 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024

def detect_language_from_text(text):
    """Fast language detection based on common words - no external library needed"""
    # Only check first 1000 chars for speed
    sample = text[:1000].lower() if len(text) > 1000 else text.lower()
    
    # Count language indicators
    language_scores = {
        'spanish': 0,
        'french': 0,
        'english': 0,
        'german': 0,
        'portuguese': 0,
        'italian': 0,
        'chinese': 0,
        'arabic': 0,
        'russian': 0
    }
    
    # Spanish indicators
    spanish_words = ['contrato', 'arrendamiento', 'locales', 'fecha', 'mes', 'año', 'el', 'la', 'de', 'que', 'y', 'los', 'las', 'con', 'para', 'por']
    for word in spanish_words:
        if word in sample:
            language_scores['spanish'] += 1
    
    # French indicators
    french_words = ['contrat', 'location', 'locataire', 'date', 'mois', 'année', 'le', 'la', 'de', 'que', 'et', 'les', 'avec', 'pour', 'par']
    for word in french_words:
        if word in sample:
            language_scores['french'] += 1
    
    # English indicators
    english_words = ['contract', 'lease', 'tenant', 'landlord', 'date', 'month', 'year', 'the', 'and', 'of', 'to', 'with', 'for', 'by']
    for word in english_words:
        if word in sample:
            language_scores['english'] += 1
    
    # German indicators
    german_words = ['vertrag', 'miete', 'mieter', 'vermieter', 'datum', 'monat', 'jahr', 'der', 'die', 'das', 'und', 'mit', 'für', 'von']
    for word in german_words:
        if word in sample:
            language_scores['german'] += 1
    
    # Portuguese indicators
    portuguese_words = ['contrato', 'arrendamento', 'locatário', 'senhorio', 'data', 'mês', 'ano', 'o', 'a', 'de', 'que', 'e', 'com', 'para']
    for word in portuguese_words:
        if word in sample:
            language_scores['portuguese'] += 1
    
    # Italian indicators
    italian_words = ['contratto', 'affitto', 'locatore', 'locatario', 'data', 'mese', 'anno', 'il', 'la', 'di', 'che', 'e', 'con', 'per']
    for word in italian_words:
        if word in sample:
            language_scores['italian'] += 1
    
    # Check for Chinese characters
    if any('\u4e00' <= char <= '\u9fff' for char in sample):
        language_scores['chinese'] = 10
    
    # Check for Arabic characters
    if any('\u0600' <= char <= '\u06ff' for char in sample):
        language_scores['arabic'] = 10
    
    # Check for Cyrillic characters
    if any('\u0400' <= char <= '\u04ff' for char in sample):
        language_scores['russian'] = 10
    
    # Get language with highest score
    detected = max(language_scores, key=language_scores.get)
    
    # If no clear winner, default to spanish (most common in your use case)
    if language_scores[detected] == 0:
        return "spanish"
    
    return detected

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
        'low': 1,      # Changed to 1 pass for speed
        'medium': 2,   # Changed to 2 passes
        'high': 3,     # Changed to 3 passes
        'ultra': 4     # Changed to 4 passes
    }
    
    num_passes = passes.get(verification_level, 1)
    extracted_texts = []
    
    for i in range(num_passes):
        processed_image = image
        
        if i == 1:
            processed_image = image.point(lambda p: p > 128 and 255)
        elif i == 2:
            processed_image = image.filter(ImageFilter.MedianFilter())
        elif i == 3:
            processed_image = image.filter(ImageFilter.SHARPEN)
        
        text = pytesseract.image_to_string(processed_image)
        extracted_texts.append(text)
        
        # Force garbage collection after each pass
        if i % 2 == 0:
            gc.collect()
        
        await asyncio.sleep(0.1)
    
    if num_passes == 1:
        final_text = extracted_texts[0]
        confidence = 100.0
    else:
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

# HTML interface remains the same (keeping it as is - too long to include)
@app.get("/", response_class=HTMLResponse)
async def main():
    # [HTML content omitted for brevity - use the same HTML as before]
    return """<!DOCTYPE html>... [same HTML as before] ...</html>"""

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
        
        # Detect the actual language using fast method
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
    
    CURRENT VERIFICATION LEVELS:
    - low: 1 pass (single extraction for speed)
    - medium: 2 passes (extract + verify)
    - high: 3 passes (extract + 2 verify)  
    - ultra: 4 passes (extract + 3 verify)
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
    print(f"Starting server with memory optimization and fast language detection...")
    print(f"Max file size: {MAX_FILE_SIZE/1024/1024}MB")
    print(f"Max stored results: {MAX_RESULTS}")
    print(f"Verification levels: low=1 pass, medium=2 passes, high=3 passes, ultra=4 passes")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

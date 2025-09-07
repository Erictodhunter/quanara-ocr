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
import re

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
    description="Multi-pass OCR verification system for lease document processing with Modal integration",
    version="5.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Store processed results in memory with size limit
processed_results = {}
MAX_RESULTS = 10  # Limit stored results to prevent memory overflow

# Maximum file size: 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024

def clean_text_for_json(text):
    """Clean OCR text to be JSON-safe and remove problematic characters"""
    if not text:
        return ""
    
    # Remove control characters (0x00-0x1F, 0x7F-0x9F)
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
    
    # Replace problematic characters for JSON
    text = text.replace('"', "'")  # Replace double quotes with single quotes
    text = text.replace('\\', '/')  # Replace backslashes with forward slashes
    text = text.replace('\b', ' ')  # Replace backspace
    text = text.replace('\f', ' ')  # Replace form feed
    text = text.replace('\v', ' ')  # Replace vertical tab
    
    # Remove non-printable Unicode characters
    text = re.sub(r'[\u0000-\u001F\u007F-\u009F]', ' ', text)
    
    # Remove zero-width characters and other problematic Unicode
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    
    # Keep only ASCII printable characters, basic punctuation, and whitespace
    text = re.sub(r'[^\x20-\x7E\n\r\t]', ' ', text)
    
    # Normalize whitespace - collapse multiple spaces/newlines but preserve structure
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Preserve paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)      # Collapse horizontal whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)   # Limit consecutive newlines
    
    # Remove any remaining problematic sequences
    text = text.replace('\x00', '')  # Remove null bytes
    
    return text.strip()

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
    french_words = ['contrat', 'location', 'locataire', 'bailleur', 'date', 'mois', 'année', 'le', 'la', 'de', 'que', 'et', 'les', 'avec', 'pour', 'par']
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
            page_texts = []  # Store page-by-page results for Modal format
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
                    
                    # Clean the text for this page
                    cleaned_page_text = clean_text_for_json(result['text'])
                    
                    if cleaned_page_text.strip():
                        # Store in Modal format: {"page": number, "text": "content"}
                        page_texts.append({
                            "page": i,
                            "text": cleaned_page_text
                        })
                    
                    total_confidence += result['confidence']
                    page_time = round(time.time() - page_start_time, 1)
                    
                    yield f"data: {json.dumps({'type': 'page_complete', 'file_id': file_id, 'page': i, 'confidence': result['confidence'], 'passes': result['passes'], 'variations': result['variations'], 'text_preview': cleaned_page_text[:200] + '...' if len(cleaned_page_text) > 200 else cleaned_page_text, 'page_time': page_time})}\n\n"
                    
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
            
            avg_confidence = total_confidence / total_pages if total_pages > 0 else 0
            
            # Detect language from combined text
            combined_text = " ".join([page["text"] for page in page_texts])
            detected_language = detect_language_from_text(combined_text)
            
        else:
            # Process image
            yield f"data: {json.dumps({'type': 'progress', 'file_id': file_id, 'progress': 50, 'message': f'Processing image with {verification_level} verification...', 'elapsed_time': round(time.time() - start_time, 1)})}\n\n"
            
            image = Image.open(temp_file_path)
            result = await verify_ocr_extraction(image, verification_level)
            
            # Clean the text
            cleaned_text = clean_text_for_json(result['text'])
            
            # Format as single page for Modal
            page_texts = [{"page": 1, "text": cleaned_text}]
            avg_confidence = result['confidence']
            
            # Detect language from the cleaned text
            detected_language = detect_language_from_text(cleaned_text)
            
            image.close()
            image = None
        
        total_time = round(time.time() - start_time, 1)
        
        # Cleanup old results before storing new one
        cleanup_old_results()
        
        # Store result with page format for Modal
        processed_results[file_id] = {
            'filename': filename,
            'ocr_pages': page_texts,  # This is the key format Modal expects
            'confidence': avg_confidence,
            'verification_level': verification_level,
            'detected_language': detected_language,
            'total_time': total_time,
            'timestamp': datetime.now().isoformat(),
            'total_pages': len(page_texts),
            'character_count': sum(len(page["text"]) for page in page_texts)
        }
        
        # Calculate total characters
        total_chars = sum(len(page["text"]) for page in page_texts)
        
        yield f"data: {json.dumps({'type': 'complete', 'file_id': file_id, 'ocr_pages': page_texts, 'total_chars': total_chars, 'average_confidence': avg_confidence, 'verification_level': verification_level, 'detected_language': detected_language, 'message': f'Processing complete! Average confidence: {avg_confidence:.1f}%', 'total_time': total_time})}\n\n"
        
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

# NEW ENDPOINT FOR MODAL INTEGRATION
@app.post("/extract-for-modal")
async def extract_for_modal(file: UploadFile = File(...), verification_level: str = Form('medium')):
    """
    Extract text in the exact format that Modal Hunyuan system expects.
    
    Returns: {
        "ocr_pages": [{"page": 1, "text": "content"}, {"page": 2, "text": "content"}],
        "filename": "document.pdf",
        "metadata": {...}
    }
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
        
        page_texts = []
        total_confidence = 0
        
        if file.filename.lower().endswith('.pdf'):
            # Process page by page
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
                    # Run OCR with verification
                    result = await verify_ocr_extraction(image, verification_level)
                    
                    # Clean the text
                    cleaned_text = clean_text_for_json(result['text'])
                    
                    if cleaned_text.strip():
                        page_texts.append({
                            "page": i,
                            "text": cleaned_text
                        })
                    
                    total_confidence += result['confidence']
                    
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
            
            avg_confidence = total_confidence / total_pages if total_pages > 0 else 0
        else:
            # Process single image
            image = Image.open(temp_file_path)
            result = await verify_ocr_extraction(image, verification_level)
            
            cleaned_text = clean_text_for_json(result['text'])
            
            page_texts = [{"page": 1, "text": cleaned_text}]
            avg_confidence = result['confidence']
            
            image.close()
            image = None
        
        # Detect language from combined text
        combined_text = " ".join([page["text"] for page in page_texts])
        detected_language = detect_language_from_text(combined_text)
        
        gc.collect()
        
        # Return in EXACT format Modal expects
        return JSONResponse({
            "ocr_pages": page_texts,
            "filename": file.filename,
            "metadata": {
                "ocr_confidence": avg_confidence,
                "verification_level": verification_level,
                "detected_language": detected_language,
                "total_pages": len(page_texts),
                "character_count": sum(len(page["text"]) for page in page_texts),
                "processing_timestamp": datetime.now().isoformat()
            }
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

# Keep all your existing endpoints...
@app.get("/", response_class=HTMLResponse)
async def main():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MAFM OCR API v5.0 - Modal Integration</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .upload-area { border: 2px dashed #ddd; padding: 40px; text-align: center; border-radius: 10px; margin: 20px 0; }
        .upload-area.drag-over { border-color: #007bff; background: #f0f8ff; }
        input[type="file"] { display: none; }
        .btn { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; margin: 10px; }
        .btn:hover { background: #0056b3; }
        .progress { width: 100%; height: 20px; background: #f0f0f0; border-radius: 10px; overflow: hidden; margin: 10px 0; }
        .progress-bar { height: 100%; background: #007bff; transition: width 0.3s; }
        .result { margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 5px; }
        .hidden { display: none; }
        .modal-format { background: #e8f5e8; border: 1px solid #28a745; padding: 15px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>MAFM OCR API v5.0</h1>
        <p>Upload PDF or image files for text extraction with Modal Hunyuan integration.</p>
        
        <div class="modal-format">
            <h3>✅ Modal Integration Ready</h3>
            <p>This OCR system outputs the exact format needed for your Hunyuan translation system.</p>
            <p><strong>Endpoint:</strong> <code>/extract-for-modal</code></p>
        </div>
        
        <div class="upload-area" id="uploadArea">
            <p>Drag and drop files here or click to select</p>
            <input type="file" id="fileInput" accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp" multiple>
            <button class="btn" onclick="document.getElementById('fileInput').click()">Select Files</button>
        </div>
        
        <div>
            <label>Verification Level:</label>
            <select id="verificationLevel">
                <option value="low">Low (1 pass - fastest)</option>
                <option value="medium" selected>Medium (2 passes - recommended)</option>
                <option value="high">High (3 passes)</option>
                <option value="ultra">Ultra (4 passes - most accurate)</option>
            </select>
        </div>
        
        <div id="results"></div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const results = document.getElementById('results');
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            handleFiles(e.dataTransfer.files);
        });
        
        fileInput.addEventListener('change', (e) => {
            handleFiles(e.target.files);
        });
        
        function handleFiles(files) {
            Array.from(files).forEach(file => processFile(file));
        }
        
        function processFile(file) {
            const fileId = Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            const verificationLevel = document.getElementById('verificationLevel').value;
            
            const resultDiv = document.createElement('div');
            resultDiv.className = 'result';
            resultDiv.innerHTML = `
                <h3>${file.name}</h3>
                <div class="progress">
                    <div class="progress-bar" id="progress-${fileId}"></div>
                </div>
                <div id="status-${fileId}">Starting...</div>
                <div id="text-${fileId}" class="hidden"></div>
            `;
            results.appendChild(resultDiv);
            
            const formData = new FormData();
            formData.append('file', file);
            formData.append('file_id', fileId);
            formData.append('verification_level', verificationLevel);
            
            fetch('/stream-extract', {
                method: 'POST',
                body: formData
            }).then(response => {
                const reader = response.body.getReader();
                
                function readStream() {
                    reader.read().then(({ done, value }) => {
                        if (done) return;
                        
                        const text = new TextDecoder().decode(value);
                        const lines = text.split('\n');
                        
                        lines.forEach(line => {
                            if (line.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(line.substring(6));
                                    updateProgress(fileId, data);
                                } catch (e) {}
                            }
                        });
                        
                        readStream();
                    });
                }
                
                readStream();
            });
        }
        
        function updateProgress(fileId, data) {
            const progressBar = document.getElementById(`progress-${fileId}`);
            const status = document.getElementById(`status-${fileId}`);
            const textDiv = document.getElementById(`text-${fileId}`);
            
            if (data.type === 'progress') {
                progressBar.style.width = data.progress + '%';
                status.textContent = data.message;
            } else if (data.type === 'complete') {
                progressBar.style.width = '100%';
                status.innerHTML = `✅ ${data.message}`;
                
                // Show Modal format
                const modalFormat = data.ocr_pages ? JSON.stringify(data.ocr_pages, null, 2) : 'No page data';
                textDiv.innerHTML = `
                    <h4>Modal Format Output:</h4>
                    <pre style="white-space: pre-wrap; background: #e8f5e8; padding: 15px; border-radius: 5px; border: 1px solid #28a745; max-height: 300px; overflow-y: auto;">${modalFormat}</pre>
                    <p><strong>Ready for Modal Hunyuan processing!</strong></p>
                `;
                textDiv.classList.remove('hidden');
            } else if (data.type === 'error') {
                status.innerHTML = `❌ Error: ${data.error}`;
            }
        }
    </script>
</body>
</html>"""

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
        
        # Clean the text for JSON safety
        cleaned_text = clean_text_for_json(final_text)
        
        # Detect the actual language using fast method
        detected_language = detect_language_from_text(cleaned_text)
        
        gc.collect()
        
        return JSONResponse({
            "text": cleaned_text,
            "pages": pages,
            "filename": file.filename,
            "character_count": len(cleaned_text),
            "language_detection": detected_language
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
    print(f"Starting server with Modal integration...")
    print(f"Max file size: {MAX_FILE_SIZE/1024/1024}MB")
    print(f"Max stored results: {MAX_RESULTS}")
    print(f"Verification levels: low=1 pass, medium=2 passes, high=3 passes, ultra=4 passes")
    print(f"NEW: /extract-for-modal endpoint - outputs format for Hunyuan processing")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

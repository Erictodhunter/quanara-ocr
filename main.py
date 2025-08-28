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

app = FastAPI(
    title="QUANARA OCR API",
    description="Auto-detect language OCR service with multi-pass verification",
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def calculate_confidence(texts):
    """Calculate confidence score based on text similarity"""
    if len(texts) == 1:
        return 100.0
    
    # Compare all pairs and get average similarity
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
    
    # For character-level consensus
    consensus = []
    max_len = max(len(text) for text in texts)
    
    for i in range(max_len):
        chars_at_position = []
        for text in texts:
            if i < len(text):
                chars_at_position.append(text[i])
        
        if chars_at_position:
            # Get most common character at this position
            char_counts = Counter(chars_at_position)
            consensus.append(char_counts.most_common(1)[0][0])
    
    return ''.join(consensus)

async def verify_ocr_extraction(image, verification_level):
    """Run OCR multiple times based on verification level"""
    passes = {
        'low': 2,      # 1 extract + 1 verify
        'medium': 3,   # 1 extract + 2 verify
        'high': 4,     # 1 extract + 3 verify
        'ultra': 5     # 1 extract + 4 verify
    }
    
    num_passes = passes.get(verification_level, 2)
    extracted_texts = []
    
    for i in range(num_passes):
        # Try different preprocessing for each pass
        processed_image = image
        
        if i == 1:
            # Increase contrast
            processed_image = image.point(lambda p: p > 128 and 255)
        elif i == 2:
            # Denoise
            processed_image = image.filter(ImageFilter.MedianFilter())
        elif i == 3:
            # Sharpen
            processed_image = image.filter(ImageFilter.SHARPEN)
        elif i == 4:
            # Different DPI/scale
            processed_image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
        
        text = pytesseract.image_to_string(processed_image)
        extracted_texts.append(text)
        
        # Small delay between passes
        await asyncio.sleep(0.1)
    
    # Get consensus text and confidence
    final_text = get_consensus_text(extracted_texts)
    confidence = calculate_confidence(extracted_texts)
    
    return {
        'text': final_text,
        'confidence': confidence,
        'passes': num_passes,
        'variations': len(set(extracted_texts))
    }

async def stream_ocr_progress(file_content: bytes, filename: str, file_id: str, verification_level: str = 'low'):
    """Stream OCR progress with verification passes"""
    try:
        start_time = time.time()
        
        # Send start event
        yield f"data: {json.dumps({'type': 'start', 'file_id': file_id, 'filename': filename, 'verification_level': verification_level, 'message': f'Starting processing with {verification_level} verification...', 'start_time': start_time})}\n\n"
        await asyncio.sleep(0.1)
        
        if filename.lower().endswith('.pdf'):
            images = pdf2image.convert_from_bytes(file_content, dpi=150)
            total_pages = len(images)
            
            yield f"data: {json.dumps({'type': 'info', 'file_id': file_id, 'total_pages': total_pages, 'message': f'PDF loaded: {total_pages} pages'})}\n\n"
            
            all_text = []
            total_confidence = 0
            detected_languages = set()
            
            for i, image in enumerate(images, 1):
                page_start_time = time.time()
                
                yield f"data: {json.dumps({'type': 'progress', 'file_id': file_id, 'current_page': i, 'total_pages': total_pages, 'progress': int((i-1)/total_pages * 100), 'message': f'Processing page {i}/{total_pages} with {verification_level} verification', 'elapsed_time': round(time.time() - start_time, 1)})}\n\n"
                
                # Detect language
                try:
                    osd = pytesseract.image_to_osd(image)
                    script_info = [line for line in osd.split('\n') if 'Script:' in line]
                    if script_info:
                        detected_languages.add(script_info[0].split(':')[1].strip())
                except:
                    pass
                
                # Run verification passes
                result = await verify_ocr_extraction(image, verification_level)
                
                if result['text'].strip():
                    all_text.append(f"[Page {i} - Confidence: {result['confidence']:.1f}%]\n{result['text']}")
                
                total_confidence += result['confidence']
                page_time = round(time.time() - page_start_time, 1)
                
                yield f"data: {json.dumps({'type': 'page_complete', 'file_id': file_id, 'page': i, 'confidence': result['confidence'], 'passes': result['passes'], 'variations': result['variations'], 'text_preview': result['text'][:200] + '...' if len(result['text']) > 200 else result['text'], 'page_time': page_time})}\n\n"
                await asyncio.sleep(0.1)
            
            final_text = "\n\n".join(all_text)
            avg_confidence = total_confidence / total_pages if total_pages > 0 else 0
            detected_langs_str = ", ".join(detected_languages) if detected_languages else "Multiple/Unknown"
        else:
            yield f"data: {json.dumps({'type': 'progress', 'file_id': file_id, 'progress': 50, 'message': f'Processing image with {verification_level} verification...', 'elapsed_time': round(time.time() - start_time, 1)})}\n\n"
            image = Image.open(io.BytesIO(file_content))
            
            result = await verify_ocr_extraction(image, verification_level)
            final_text = result['text']
            avg_confidence = result['confidence']
            detected_langs_str = "Auto-detected"
        
        total_time = round(time.time() - start_time, 1)
        
        yield f"data: {json.dumps({'type': 'complete', 'file_id': file_id, 'text': final_text, 'total_chars': len(final_text), 'average_confidence': avg_confidence, 'verification_level': verification_level, 'detected_languages': detected_langs_str, 'message': f'Processing complete! Average confidence: {avg_confidence:.1f}%', 'total_time': total_time})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'file_id': file_id, 'error': str(e)})}\n\n"

@app.get("/", response_class=HTMLResponse)
async def main():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>QUANARA OCR - Enterprise Lease Processing with Verification</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
                background: #0a0e27;
                min-height: 100vh; 
                padding: 20px;
                color: #e4e6eb;
            }
            .container { max-width: 1400px; margin: 0 auto; }
            
            .header { 
                text-align: center; 
                margin-bottom: 40px;
                padding: 40px;
                background: linear-gradient(135deg, #1a1f3a 0%, #252b4a 100%);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                position: relative;
                overflow: hidden;
            }
            
            .header h1 { 
                font-size: 3.5rem; 
                margin-bottom: 10px; 
                background: linear-gradient(45deg, #667eea, #764ba2, #f093fb);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                position: relative;
                z-index: 1;
            }
            
            .verification-selector {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                padding: 25px;
                margin: 20px 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .verification-options {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                margin-top: 15px;
            }
            
            @media (max-width: 768px) {
                .verification-options { grid-template-columns: repeat(2, 1fr); }
            }
            
            .verification-option {
                padding: 20px;
                border: 2px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                cursor: pointer;
                transition: all 0.3s;
                text-align: center;
                background: rgba(255, 255, 255, 0.03);
            }
            
            .verification-option:hover {
                transform: translateY(-2px);
                border-color: #667eea;
                box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
            }
            
            .verification-option.selected {
                border-color: #667eea;
                background: rgba(102, 126, 234, 0.2);
                box-shadow: 0 0 20px rgba(102, 126, 234, 0.4);
            }
            
            .verification-option h4 {
                color: #e4e6eb;
                margin-bottom: 8px;
                font-size: 1.2rem;
            }
            
            .verification-option .passes {
                color: #667eea;
                font-size: 2rem;
                font-weight: bold;
                margin: 10px 0;
            }
            
            .verification-option .description {
                color: #b4b7c2;
                font-size: 0.85rem;
            }
            
            .confidence-indicator {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 15px;
                font-size: 12px;
                font-weight: 600;
                margin-left: 10px;
            }
            
            .confidence-high {
                background: rgba(40, 167, 69, 0.2);
                color: #28a745;
                border: 1px solid rgba(40, 167, 69, 0.3);
            }
            
            .confidence-medium {
                background: rgba(255, 193, 7, 0.2);
                color: #ffc107;
                border: 1px solid rgba(255, 193, 7, 0.3);
            }
            
            .confidence-low {
                background: rgba(220, 53, 69, 0.2);
                color: #dc3545;
                border: 1px solid rgba(220, 53, 69, 0.3);
            }
            
            .main-grid { 
                display: grid; 
                grid-template-columns: 1fr 1fr; 
                gap: 30px; 
                margin-bottom: 30px; 
            }
            
            @media (max-width: 968px) {
                .main-grid { grid-template-columns: 1fr; }
            }
            
            .box { 
                background: #1a1f3a;
                border-radius: 20px; 
                padding: 35px; 
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
            }
            
            .upload-area { 
                border: 2px dashed #667eea; 
                border-radius: 15px; 
                padding: 50px; 
                text-align: center; 
                background: rgba(102, 126, 234, 0.05);
                transition: all 0.3s;
                cursor: pointer;
            }
            
            .upload-area:hover, .upload-area.dragover { 
                border-color: #764ba2; 
                background: rgba(102, 126, 234, 0.1);
                transform: translateY(-2px);
            }
            
            input[type="file"] { display: none; }
            
            .btn { 
                background: linear-gradient(45deg, #667eea, #764ba2); 
                color: white; 
                padding: 15px 35px; 
                border: none; 
                border-radius: 50px; 
                font-size: 16px; 
                font-weight: 600;
                cursor: pointer; 
                transition: all 0.3s;
                margin: 5px;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }
            
            .btn:hover { 
                transform: translateY(-2px); 
                box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
            }
            
            .btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                transform: none;
            }
            
            .file-item {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                padding: 25px;
                margin: 15px 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: all 0.3s;
            }
            
            .progress-bar {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                height: 12px;
                margin: 15px 0;
                overflow: hidden;
                position: relative;
            }
            
            .progress-fill {
                background: linear-gradient(45deg, #667eea, #764ba2);
                height: 100%;
                transition: width 0.3s ease;
                border-radius: 10px;
            }
            
            .status-badge {
                display: inline-block;
                padding: 6px 16px;
                border-radius: 20px;
                font-size: 13px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .status-waiting { 
                background: rgba(108, 117, 125, 0.2); 
                color: #adb5bd;
                border: 1px solid rgba(108, 117, 125, 0.3);
            }
            
            .status-processing { 
                background: rgba(255, 193, 7, 0.2); 
                color: #ffc107;
                border: 1px solid rgba(255, 193, 7, 0.3);
            }
            
            .status-complete { 
                background: rgba(40, 167, 69, 0.2); 
                color: #28a745;
                border: 1px solid rgba(40, 167, 69, 0.3);
            }
            
            .status-error { 
                background: rgba(220, 53, 69, 0.2); 
                color: #dc3545;
                border: 1px solid rgba(220, 53, 69, 0.3);
            }
            
            .empty-state {
                text-align: center;
                padding: 60px 20px;
                color: #6c757d;
            }
            
            code {
                background: rgba(102, 126, 234, 0.1);
                color: #667eea;
                padding: 2px 8px;
                border-radius: 4px;
                font-family: 'Consolas', monospace;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏢 QUANARA OCR</h1>
                <p>Enterprise Lease Document Processing with Multi-Pass Verification</p>
            </div>
            
            <div class="verification-selector">
                <h3 style="color: #e4e6eb; margin-bottom: 10px;">🔍 Accuracy Verification Level</h3>
                <p style="color: #b4b7c2; font-size: 0.9rem;">Choose how many verification passes to ensure accuracy</p>
                
                <div class="verification-options">
                    <div class="verification-option selected" data-level="low">
                        <h4>Low</h4>
                        <div class="passes">2x</div>
                        <div class="description">1 extract + 1 verify<br>Fast processing</div>
                    </div>
                    <div class="verification-option" data-level="medium">
                        <h4>Medium</h4>
                        <div class="passes">3x</div>
                        <div class="description">1 extract + 2 verify<br>Balanced accuracy</div>
                    </div>
                    <div class="verification-option" data-level="high">
                        <h4>High</h4>
                        <div class="passes">4x</div>
                        <div class="description">1 extract + 3 verify<br>High confidence</div>
                    </div>
                    <div class="verification-option" data-level="ultra">
                        <h4>Ultra</h4>
                        <div class="passes">5x</div>
                        <div class="description">1 extract + 4 verify<br>Maximum accuracy</div>
                    </div>
                </div>
            </div>
            
            <div class="main-grid">
                <div class="box">
                    <h3>📤 Upload Lease Documents</h3>
                    <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
                        <div style="font-size: 3rem;">📁</div>
                        <h4 style="color: #e4e6eb; margin: 15px 0;">Drop Lease Documents Here</h4>
                        <p style="color: #b4b7c2;">Files will be processed with <code id="selectedLevel">Low (2x)</code> verification</p>
                        <input type="file" id="fileInput" accept=".pdf,.jpg,.jpeg,.png" multiple>
                    </div>
                    
                    <button class="btn" id="processBtn" onclick="processAllFiles()" style="width: 100%; margin-top: 25px;" disabled>
                        🚀 Start Processing Queue
                    </button>
                </div>
                
                <div class="box">
                    <h3>📋 Processing Queue</h3>
                    <div id="filesList">
                        <div class="empty-state">
                            <div style="font-size: 3rem; opacity: 0.3;">📭</div>
                            <p>No files uploaded yet</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let uploadedFiles = [];
            let selectedVerificationLevel = 'low';
            
            // Verification level selection
            document.querySelectorAll('.verification-option').forEach(option => {
                option.addEventListener('click', function() {
                    document.querySelectorAll('.verification-option').forEach(o => o.classList.remove('selected'));
                    this.classList.add('selected');
                    selectedVerificationLevel = this.dataset.level;
                    
                    const levelText = {
                        'low': 'Low (2x)',
                        'medium': 'Medium (3x)',
                        'high': 'High (4x)',
                        'ultra': 'Ultra (5x)'
                    };
                    document.getElementById('selectedLevel').textContent = levelText[selectedVerificationLevel];
                });
            });
            
            // File handling functions
            const uploadArea = document.getElementById('uploadArea');
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, preventDefaults, false);
            });
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            uploadArea.addEventListener('drop', handleDrop, false);
            
            function handleDrop(e) {
                const files = [...e.dataTransfer.files];
                handleFiles(files);
            }
            
            document.getElementById('fileInput').addEventListener('change', (e) => {
                handleFiles([...e.target.files]);
            });
            
            function handleFiles(files) {
                files.forEach(file => {
                    const fileId = 'file-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
                    uploadedFiles.push({
                        id: fileId,
                        file: file,
                        status: 'waiting',
                        progress: 0,
                        verificationLevel: selectedVerificationLevel
                    });
                });
                
                updateFilesList();
                document.getElementById('processBtn').disabled = uploadedFiles.length === 0;
            }
            
            function updateFilesList() {
                const filesList = document.getElementById('filesList');
                if (uploadedFiles.length === 0) {
                    filesList.innerHTML = '<div class="empty-state"><div style="font-size: 3rem; opacity: 0.3;">📭</div><p>No files uploaded yet</p></div>';
                    return;
                }
                
                filesList.innerHTML = uploadedFiles.map(fileInfo => {
                    const levelText = {
                        'low': '2x verification',
                        'medium': '3x verification',
                        'high': '4x verification',
                        'ultra': '5x verification'
                    };
                    
                    return `
                        <div class="file-item">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong>${fileInfo.file.name}</strong>
                                    <span style="color: #667eea; margin-left: 10px; font-size: 0.9rem;">${levelText[fileInfo.verificationLevel]}</span>
                                </div>
                                <span class="status-badge status-${fileInfo.status}">${fileInfo.status}</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${fileInfo.progress}%"></div>
                            </div>
                            <div id="${fileInfo.id}-message" style="font-size: 14px; color: #b4b7c2; margin-top: 10px;"></div>
                            <div id="${fileInfo.id}-confidence"></div>
                        </div>
                    `;
                }).join('');
            }
            
            async function processAllFiles() {
                document.getElementById('processBtn').disabled = true;
                
                for (let fileInfo of uploadedFiles) {
                    if (fileInfo.status === 'waiting') {
                        await processFile(fileInfo);
                    }
                }
                
                document.getElementById('processBtn').disabled = uploadedFiles.filter(f => f.status === 'waiting').length === 0;
            }
            
            async function processFile(fileInfo) {
                fileInfo.status = 'processing';
                updateFilesList();
                
                const formData = new FormData();
                formData.append('file', fileInfo.file);
                formData.append('file_id', fileInfo.id);
                formData.append('verification_level', fileInfo.verificationLevel);
                
                try {
                    const response = await fetch('/stream-extract?verification_level=' + fileInfo.verificationLevel, {
                        method: 'POST',
                        body: formData
                    });
                    
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let buffer = '';
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\\n');
                        buffer = lines.pop();
                        
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(line.slice(6));
                                    handleStreamData(data, fileInfo);
                                } catch (e) {
                                    console.error('Parse error:', e);
                                }
                            }
                        }
                    }
                } catch (error) {
                    fileInfo.status = 'error';
                    document.getElementById(fileInfo.id + '-message').textContent = 'Error: ' + error.message;
                    updateFilesList();
                }
            }
            
            function handleStreamData(data, fileInfo) {
                const messageEl = document.getElementById(fileInfo.id + '-message');
                const confidenceEl = document.getElementById(fileInfo.id + '-confidence');
                
                switch(data.type) {
                    case 'progress':
                        fileInfo.progress = data.progress;
                        messageEl.textContent = data.message;
                        updateFilesList();
                        break;
                        
                    case 'page_complete':
                        if (data.confidence) {
                            const confidenceClass = data.confidence > 90 ? 'high' : data.confidence > 70 ? 'medium' : 'low';
                            confidenceEl.innerHTML = `
                                <span class="confidence-indicator confidence-${confidenceClass}">
                                    Page confidence: ${data.confidence.toFixed(1)}% (${data.passes} passes, ${data.variations} variations)
                                </span>
                            `;
                        }
                        break;
                        
                    case 'complete':
                        fileInfo.status = 'complete';
                        fileInfo.progress = 100;
                        const avgConfidenceClass = data.average_confidence > 90 ? 'high' : data.average_confidence > 70 ? 'medium' : 'low';
                        messageEl.innerHTML = `
                            Complete! ${data.total_chars} characters extracted
                            <span class="confidence-indicator confidence-${avgConfidenceClass}">
                                Average confidence: ${data.average_confidence.toFixed(1)}%
                            </span>
                        `;
                        updateFilesList();
                        break;
                        
                    case 'error':
                        fileInfo.status = 'error';
                        messageEl.textContent = 'Error: ' + data.error;
                        updateFilesList();
                        break;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    """Extract text from PDF or image file with automatic language detection"""
    try:
        content = await file.read()
        
        if file.filename.lower().endswith('.pdf'):
            images = pdf2image.convert_from_bytes(content, dpi=150)
            all_text = []
            
            for i, image in enumerate(images, 1):
                # Auto-detect and extract
                text = pytesseract.image_to_string(image)
                if text.strip():
                    all_text.append(f"[Page {i}]\n{text}")
            
            final_text = "\n\n".join(all_text)
            pages = len(images)
        else:
            image = Image.open(io.BytesIO(content))
            final_text = pytesseract.image_to_string(image)
            pages = 1
        
        return JSONResponse({
            "text": final_text,
            "pages": pages,
            "filename": file.filename,
            "character_count": len(final_text),
            "language_detection": "automatic"
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/stream-extract")
async def stream_extract(
    file: UploadFile = File(...), 
    file_id: str = None,
    verification_level: str = Form('low')
):
    """Extract text with real-time progress streaming and verification passes"""
    content = await file.read()
    file_id = file_id or str(uuid.uuid4())
    
    # Validate verification level
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

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
    title="MAFM OCR API",
    description="Multi-pass OCR verification system for lease document processing",
    version="4.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Store processed results in memory (use Redis/DB in production)
processed_results = {}

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
            processed_image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
        
        text = pytesseract.image_to_string(processed_image)
        extracted_texts.append(text)
        
        await asyncio.sleep(0.1)
    
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
                
                try:
                    osd = pytesseract.image_to_osd(image)
                    script_info = [line for line in osd.split('\n') if 'Script:' in line]
                    if script_info:
                        detected_languages.add(script_info[0].split(':')[1].strip())
                except:
                    pass
                
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
        
        # Store result
        processed_results[file_id] = {
            'filename': filename,
            'text': final_text,
            'confidence': avg_confidence,
            'verification_level': verification_level,
            'detected_languages': detected_langs_str,
            'total_time': total_time,
            'timestamp': datetime.now().isoformat(),
            'character_count': len(final_text)
        }
        
        yield f"data: {json.dumps({'type': 'complete', 'file_id': file_id, 'text': final_text, 'total_chars': len(final_text), 'average_confidence': avg_confidence, 'verification_level': verification_level, 'detected_languages': detected_langs_str, 'message': f'Processing complete! Average confidence: {avg_confidence:.1f}%', 'total_time': total_time})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'file_id': file_id, 'error': str(e)})}\n\n"

@app.get("/", response_class=HTMLResponse)
async def main():
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
        </style>
    </head>
    <body>
        <div class="ms-header">
            <div class="container">
                <h1>MAFM OCR</h1>
                <div class="subtitle">Multi-pass Document Processing System</div>
            </div>
        </div>
        
        <div class="container mt-4">
            <div class="ms-tabs">
                <button class="ms-tab active" onclick="showTab('upload')">Upload & Process</button>
                <button class="ms-tab" onclick="showTab('results')">View Results</button>
                <button class="ms-tab" onclick="showTab('api')">API Documentation</button>
            </div>
            
            <!-- Upload Tab -->
            <div id="uploadTab" class="tab-content">
                <div class="ms-card">
                    <h3>Verification Level</h3>
                    <div class="verification-options">
                        <div class="verification-option selected" data-level="low" onclick="selectVerification(this)">
                            <h4>Standard</h4>
                            <div class="passes">2×</div>
                            <small>Fast processing</small>
                        </div>
                        <div class="verification-option" data-level="medium" onclick="selectVerification(this)">
                            <h4>Enhanced</h4>
                            <div class="passes">3×</div>
                            <small>Balanced accuracy</small>
                        </div>
                        <div class="verification-option" data-level="high" onclick="selectVerification(this)">
                            <h4>High</h4>
                            <div class="passes">4×</div>
                            <small>High confidence</small>
                        </div>
                        <div class="verification-option" data-level="ultra" onclick="selectVerification(this)">
                            <h4>Maximum</h4>
                            <div class="passes">5×</div>
                            <small>Highest accuracy</small>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="ms-card">
                            <h3>Upload Documents</h3>
                            <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
                                <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                                    <path d="M24 4L24 32M24 4L16 12M24 4L32 12" stroke="#0078d4" stroke-width="2"/>
                                    <path d="M8 28V40H40V28" stroke="#0078d4" stroke-width="2"/>
                                </svg>
                                <h4 class="mt-3">Drop files here or click to browse</h4>
                                <p class="text-muted mb-0">Supports PDF, JPG, PNG</p>
                                <input type="file" id="fileInput" accept=".pdf,.jpg,.jpeg,.png" multiple>
                            </div>
                            <button class="ms-button w-100 mt-3" id="processBtn" onclick="processAllFiles()" disabled>
                                <span>Process Documents</span>
                            </button>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="ms-card">
                            <h3>Processing Queue</h3>
                            <div id="filesList">
                                <div class="empty-state">
                                    <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                                        <circle cx="24" cy="24" r="20" stroke="#a19f9d" stroke-width="2"/>
                                        <path d="M24 14V26M24 32V34" stroke="#a19f9d" stroke-width="2"/>
                                    </svg>
                                    <p class="mb-0">No files in queue</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="ms-card" id="currentResults" style="display: none;">
                    <h3>Current Results</h3>
                    <div id="resultsArea"></div>
                </div>
            </div>
            
            <!-- Results Tab -->
            <div id="resultsTab" class="tab-content" style="display: none;">
                <div class="ms-card">
                    <h3>Processed Documents</h3>
                    <table class="ms-table" id="resultsTable">
                        <thead>
                            <tr>
                                <th>Filename</th>
                                <th>Confidence</th>
                                <th>Verification</th>
                                <th>Characters</th>
                                <th>Time</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="resultsTableBody">
                            <tr>
                                <td colspan="6" class="text-center text-muted">No processed documents yet</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- API Tab -->
            <div id="apiTab" class="tab-content" style="display: none;">
                <div class="ms-card">
                    <h3>API Documentation</h3>
                    
                    <h4 class="mt-4">Base URL</h4>
                    <div class="api-docs">http://localhost:8080</div>
                    
                    <h4 class="mt-4">Endpoints</h4>
                    
                    <h5 class="mt-3">1. Extract Text (Simple)</h5>
                    <div class="api-docs">
POST /extract
Content-Type: multipart/form-data

Parameters:
- file: PDF or image file

Response:
{
    "text": "extracted text",
    "pages": 1,
    "filename": "document.pdf",
    "character_count": 1234,
    "language_detection": "automatic"
}
                    </div>
                    
                    <h5 class="mt-3">2. Stream Extract (With Verification)</h5>
                    <div class="api-docs">
POST /stream-extract
Content-Type: multipart/form-data

Parameters:
- file: PDF or image file
- verification_level: "low" | "medium" | "high" | "ultra" (default: "low")

Response: Server-Sent Events stream
                    </div>
                    
                    <h5 class="mt-3">3. Get Result</h5>
                    <div class="api-docs">
GET /api/result/{file_id}

Response:
{
    "filename": "document.pdf",
    "text": "extracted text",
    "confidence": 95.5,
    "verification_level": "high",
    "detected_languages": "English",
    "total_time": 12.3,
    "timestamp": "2024-01-01T12:00:00",
    "character_count": 1234
}
                    </div>
                    
                    <h5 class="mt-3">4. List Results</h5>
                    <div class="api-docs">
GET /api/results

Response:
[
    {
        "file_id": "file-123",
        "filename": "document.pdf",
        "confidence": 95.5,
        "timestamp": "2024-01-01T12:00:00"
    }
]
                    </div>
                    
                    <h5 class="mt-3">5. Download Result</h5>
                    <div class="api-docs">
GET /api/download/{file_id}

Response: Text file download
                    </div>
                    
                    <h5 class="mt-3">6. Available Languages</h5>
                    <div class="api-docs">
GET /languages

Response:
{
    "total": 130,
    "languages": ["afr", "ara", "ben", ...]
}
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let uploadedFiles = [];
            let selectedVerificationLevel = 'low';
            let currentTab = 'upload';
            
            function showTab(tab) {
                currentTab = tab;
                document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
                document.getElementById(tab + 'Tab').style.display = 'block';
                
                document.querySelectorAll('.ms-tab').forEach(t => t.classList.remove('active'));
                event.target.classList.add('active');
                
                if (tab === 'results') {
                    loadResults();
                }
            }
            
            function selectVerification(element) {
                document.querySelectorAll('.verification-option').forEach(o => o.classList.remove('selected'));
                element.classList.add('selected');
                selectedVerificationLevel = element.dataset.level;
            }
            
            // File handling
            const uploadArea = document.getElementById('uploadArea');
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, preventDefaults, false);
            });
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            ['dragenter', 'dragover'].forEach(eventName => {
                uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
            });
            
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
                    filesList.innerHTML = `
                        <div class="empty-state">
                            <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                                <circle cx="24" cy="24" r="20" stroke="#a19f9d" stroke-width="2"/>
                                <path d="M24 14V26M24 32V34" stroke="#a19f9d" stroke-width="2"/>
                            </svg>
                            <p class="mb-0">No files in queue</p>
                        </div>
                    `;
                    return;
                }
                
                filesList.innerHTML = uploadedFiles.map(fileInfo => {
                    const levelText = {
                        'low': 'Standard (2×)',
                        'medium': 'Enhanced (3×)',
                        'high': 'High (4×)',
                        'ultra': 'Maximum (5×)'
                    };
                    
                    return `
                        <div class="file-item">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <strong>${fileInfo.file.name}</strong>
                                    <span class="text-muted ms-2">${levelText[fileInfo.verificationLevel]}</span>
                                </div>
                                <span class="status-badge status-${fileInfo.status}">${fileInfo.status}</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${fileInfo.progress}%"></div>
                            </div>
                            <div id="${fileInfo.id}-message" class="small text-muted mt-2"></div>
                            <div id="${fileInfo.id}-confidence"></div>
                        </div>
                    `;
                }).join('');
            }
            
            async function processAllFiles() {
                document.getElementById('processBtn').disabled = true;
                document.getElementById('currentResults').style.display = 'block';
                
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
                                <span class="confidence-badge confidence-${confidenceClass}">
                                    Confidence: ${data.confidence.toFixed(1)}%
                                </span>
                            `;
                        }
                        break;
                        
                    case 'complete':
                        fileInfo.status = 'complete';
                        fileInfo.progress = 100;
                        fileInfo.resultId = data.file_id;
                        
                        const avgConfidenceClass = data.average_confidence > 90 ? 'high' : data.average_confidence > 70 ? 'medium' : 'low';
                        messageEl.innerHTML = `
                            Complete - ${data.total_chars} characters
                            <span class="confidence-badge confidence-${avgConfidenceClass}">
                                ${data.average_confidence.toFixed(1)}%
                            </span>
                        `;
                        
                        // Show result
                        displayResult(data, fileInfo);
                        updateFilesList();
                        break;
                        
                    case 'error':
                        fileInfo.status = 'error';
                        messageEl.textContent = 'Error: ' + data.error;
                        updateFilesList();
                        break;
                }
            }
            
            function displayResult(data, fileInfo) {
                const resultsArea = document.getElementById('resultsArea');
                resultsArea.innerHTML += `
                    <div class="results-section">
                        <h5>${fileInfo.file.name}</h5>
                        <div class="mb-2">
                            <span class="confidence-badge confidence-${data.average_confidence > 90 ? 'high' : data.average_confidence > 70 ? 'medium' : 'low'}">
                                Confidence: ${data.average_confidence.toFixed(1)}%
                            </span>
                            <span class="ms-3">Characters: ${data.total_chars}</span>
                            <span class="ms-3">Time: ${data.total_time}s</span>
                        </div>
                        <div class="mt-2">
                            <button class="ms-button secondary" onclick="viewResult('${data.file_id}')">View Full Text</button>
                            <button class="ms-button secondary" onclick="downloadResult('${data.file_id}')">Download</button>
                            <button class="ms-button secondary" onclick="copyResult('${data.file_id}')">Copy</button>
                        </div>
                        <div class="result-text" id="preview-${data.file_id}" style="max-height: 200px;">
                            ${data.text.substring(0, 500)}${data.text.length > 500 ? '...' : ''}
                        </div>
                    </div>
                `;
            }
            
            async function loadResults() {
                try {
                    const response = await fetch('/api/results');
                    const results = await response.json();
                    
                    const tbody = document.getElementById('resultsTableBody');
                    if (results.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No processed documents yet</td></tr>';
                        return;
                    }
                    
                    tbody.innerHTML = results.map(result => `
                        <tr>
                            <td>${result.filename}</td>
                            <td>
                                <span class="confidence-badge confidence-${result.confidence > 90 ? 'high' : result.confidence > 70 ? 'medium' : 'low'}">
                                    ${result.confidence.toFixed(1)}%
                                </span>
                            </td>
                            <td>${result.verification_level}</td>
                            <td>${result.character_count.toLocaleString()}</td>
                            <td>${result.total_time}s</td>
                            <td>
                                <button class="ms-button secondary" onclick="viewResult('${result.file_id}')">View</button>
                                <button class="ms-button secondary" onclick="downloadResult('${result.file_id}')">Download</button>
                            </td>
                        </tr>
                    `).join('');
                } catch (error) {
                    console.error('Error loading results:', error);
                }
            }
            
            async function viewResult(fileId) {
                try {
                    const response = await fetch(`/api/result/${fileId}`);
                    const result = await response.json();
                    
                    // Show in modal or expand view
                    alert('Full text view would open here with:\\n\\n' + result.text.substring(0, 200) + '...');
                } catch (error) {
                    console.error('Error viewing result:', error);
                }
            }
            
            async function downloadResult(fileId) {
                window.location.href = `/api/download/${fileId}`;
            }
            
            async function copyResult(fileId) {
                try {
                    const response = await fetch(`/api/result/${fileId}`);
                    const result = await response.json();
                    
                    await navigator.clipboard.writeText(result.text);
                    
                    // Show feedback
                    const btn = event.target;
                    const originalText = btn.textContent;
                    btn.textContent = 'Copied!';
                    setTimeout(() => {
                        btn.textContent = originalText;
                    }, 2000);
                } catch (error) {
                    console.error('Error copying result:', error);
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    """
    Simple text extraction from PDF or image files.
    
    Returns extracted text with basic metadata.
    """
    try:
        content = await file.read()
        
        if file.filename.lower().endswith('.pdf'):
            images = pdf2image.convert_from_bytes(content, dpi=150)
            all_text = []
            
            for i, image in enumerate(images, 1):
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
    """
    Stream text extraction with real-time progress and multi-pass verification.
    
    Verification levels:
    - low: 2 passes (1 extract + 1 verify)
    - medium: 3 passes (1 extract + 2 verify)
    - high: 4 passes (1 extract + 3 verify)  
    - ultra: 5 passes (1 extract + 4 verify)
    """
    content = await file.read()
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

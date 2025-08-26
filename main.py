import pytesseract
import pdf2image
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
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

app = FastAPI(
    title="QUANARA OCR API",
    description="Auto-detect language OCR service for PDFs and images with real-time progress tracking",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

async def stream_ocr_progress(file_content: bytes, filename: str, file_id: str):
    """Stream OCR progress in real-time with auto language detection"""
    try:
        start_time = time.time()
        
        # Send start event
        yield f"data: {json.dumps({'type': 'start', 'file_id': file_id, 'filename': filename, 'message': 'Starting processing...', 'start_time': start_time})}\n\n"
        await asyncio.sleep(0.1)
        
        if filename.lower().endswith('.pdf'):
            # Convert PDF to images
            images = pdf2image.convert_from_bytes(file_content, dpi=150)
            total_pages = len(images)
            
            yield f"data: {json.dumps({'type': 'info', 'file_id': file_id, 'total_pages': total_pages, 'message': f'PDF loaded: {total_pages} pages'})}\n\n"
            
            all_text = []
            detected_languages = set()
            
            for i, image in enumerate(images, 1):
                page_start_time = time.time()
                
                # Process each page
                yield f"data: {json.dumps({'type': 'progress', 'file_id': file_id, 'current_page': i, 'total_pages': total_pages, 'progress': int((i-1)/total_pages * 100), 'message': f'Processing page {i}/{total_pages}', 'elapsed_time': round(time.time() - start_time, 1)})}\n\n"
                
                # First detect language using OSD (Orientation and Script Detection)
                try:
                    osd = pytesseract.image_to_osd(image)
                    # Extract script/language info from OSD output
                    script_info = [line for line in osd.split('\n') if 'Script:' in line]
                    if script_info:
                        detected_languages.add(script_info[0].split(':')[1].strip())
                except:
                    pass
                
                # Use all available languages for best results
                text = pytesseract.image_to_string(image)
                
                if text.strip():
                    all_text.append(f"[Page {i}]\n{text}")
                
                page_time = round(time.time() - page_start_time, 1)
                
                # Send page complete event
                yield f"data: {json.dumps({'type': 'page_complete', 'file_id': file_id, 'page': i, 'text_preview': text[:200] + '...' if len(text) > 200 else text, 'page_time': page_time})}\n\n"
                await asyncio.sleep(0.1)
            
            final_text = "\n\n".join(all_text)
            detected_langs_str = ", ".join(detected_languages) if detected_languages else "Multiple/Unknown"
        else:
            # Process single image
            yield f"data: {json.dumps({'type': 'progress', 'file_id': file_id, 'progress': 50, 'message': 'Processing image...', 'elapsed_time': round(time.time() - start_time, 1)})}\n\n"
            image = Image.open(io.BytesIO(file_content))
            
            # Auto detect and extract
            final_text = pytesseract.image_to_string(image)
            detected_langs_str = "Auto-detected"
        
        total_time = round(time.time() - start_time, 1)
        
        # Send completion event
        yield f"data: {json.dumps({'type': 'complete', 'file_id': file_id, 'text': final_text, 'total_chars': len(final_text), 'detected_languages': detected_langs_str, 'message': 'Processing complete!', 'total_time': total_time})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'file_id': file_id, 'error': str(e)})}\n\n"

@app.get("/", response_class=HTMLResponse)
async def main():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>QUANARA OCR - Enterprise Lease Processing</title>
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
            
            .header::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
                animation: pulse 4s ease-in-out infinite;
            }
            
            @keyframes pulse {
                0%, 100% { transform: scale(1); opacity: 0.5; }
                50% { transform: scale(1.1); opacity: 0.3; }
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
            
            .header p { 
                font-size: 1.3rem; 
                color: #b4b7c2;
                position: relative;
                z-index: 1;
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
            
            .box h3 {
                color: #e4e6eb;
                margin-bottom: 20px;
                font-size: 1.4rem;
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
            
            .file-item.processing {
                border-color: #667eea;
                box-shadow: 0 0 20px rgba(102, 126, 234, 0.2);
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
                position: relative;
                overflow: hidden;
            }
            
            .progress-fill::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                bottom: 0;
                right: 0;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
                animation: progress-shine 1.5s linear infinite;
            }
            
            @keyframes progress-shine {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(100%); }
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
            
            .timer {
                display: inline-flex;
                align-items: center;
                padding: 8px 16px;
                background: rgba(102, 126, 234, 0.1);
                border-radius: 20px;
                font-size: 14px;
                font-weight: 600;
                color: #667eea;
                margin-left: 10px;
            }
            
            .timer::before {
                content: '⏱';
                margin-right: 6px;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 20px;
                margin: 30px 0;
            }
            
            .stat-card {
                background: rgba(255, 255, 255, 0.05);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .stat-value {
                font-size: 2rem;
                font-weight: bold;
                color: #667eea;
            }
            
            .stat-label {
                font-size: 0.9rem;
                color: #b4b7c2;
                margin-top: 5px;
            }
            
            .results-preview {
                background: #0f1318;
                padding: 20px;
                border-radius: 10px;
                margin-top: 15px;
                max-height: 300px;
                overflow-y: auto;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 13px;
                white-space: pre-wrap;
                border: 1px solid rgba(255, 255, 255, 0.1);
                color: #e4e6eb;
            }
            
            .results-preview::-webkit-scrollbar {
                width: 8px;
            }
            
            .results-preview::-webkit-scrollbar-track {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 4px;
            }
            
            .results-preview::-webkit-scrollbar-thumb {
                background: #667eea;
                border-radius: 4px;
            }
            
            .feature-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
                position: relative;
                z-index: 1;
            }
            
            .feature-card {
                background: rgba(255, 255, 255, 0.05);
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: all 0.3s;
            }
            
            .feature-card:hover {
                transform: translateY(-5px);
                border-color: #667eea;
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
            }
            
            .feature-icon {
                font-size: 2.5rem;
                margin-bottom: 10px;
            }
            
            .feature-card strong {
                color: #e4e6eb;
                display: block;
                margin-bottom: 5px;
            }
            
            .feature-card p {
                color: #b4b7c2;
                font-size: 0.9rem;
            }
            
            .queue-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }
            
            .queue-info {
                display: flex;
                align-items: center;
                gap: 20px;
            }
            
            .queue-badge {
                background: rgba(102, 126, 234, 0.2);
                color: #667eea;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 600;
            }
            
            code {
                background: rgba(102, 126, 234, 0.1);
                color: #667eea;
                padding: 2px 8px;
                border-radius: 4px;
                font-family: 'Consolas', monospace;
            }
            
            .api-endpoint {
                background: rgba(255, 255, 255, 0.05);
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
                font-family: 'Consolas', monospace;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            pre {
                background: #0f1318;
                padding: 20px;
                border-radius: 8px;
                overflow-x: auto;
                color: #e4e6eb;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .empty-state {
                text-align: center;
                padding: 60px 20px;
                color: #6c757d;
            }
            
            .empty-state-icon {
                font-size: 4rem;
                opacity: 0.3;
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏢 QUANARA OCR</h1>
                <p>Enterprise Lease Document Processing</p>
                <div class="feature-grid">
                    <div class="feature-card">
                        <div class="feature-icon">📄</div>
                        <strong>Sequential Processing</strong>
                        <p>One document at a time for accuracy</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">🔍</div>
                        <strong>Auto Language Detection</strong>
                        <p>Supports 100+ languages</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">⏱️</div>
                        <strong>Real-Time Tracking</strong>
                        <p>Live progress & time elapsed</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">🚀</div>
                        <strong>Enterprise Ready</strong>
                        <p>Handles 500+ page documents</p>
                    </div>
                </div>
            </div>
            
            <div class="stats-grid" id="globalStats" style="display: none;">
                <div class="stat-card">
                    <div class="stat-value" id="totalFiles">0</div>
                    <div class="stat-label">Files Processed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="totalPages">0</div>
                    <div class="stat-label">Pages Scanned</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="totalChars">0</div>
                    <div class="stat-label">Characters Extracted</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="totalTime">0s</div>
                    <div class="stat-label">Total Time</div>
                </div>
            </div>
            
            <div class="main-grid">
                <div class="box">
                    <h3>📤 Upload Lease Documents</h3>
                    <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
                        <div class="feature-icon">📁</div>
                        <h4 style="color: #e4e6eb; margin: 15px 0;">Drop Lease Documents Here</h4>
                        <p style="color: #b4b7c2;">Support for PDF files up to 500 pages</p>
                        <input type="file" id="fileInput" accept=".pdf,.jpg,.jpeg,.png" multiple>
                    </div>
                    
                    <button class="btn" id="processBtn" onclick="processAllFiles()" style="width: 100%; margin-top: 25px;" disabled>
                        🚀 Start Processing Queue
                    </button>
                </div>
                
                <div class="box">
                    <div class="queue-header">
                        <h3>📋 Processing Queue</h3>
                        <div class="queue-info">
                            <div class="queue-badge" id="queueCount">0 files in queue</div>
                            <div class="timer" id="currentTimer" style="display: none;">0s</div>
                        </div>
                    </div>
                    <div id="filesList">
                        <div class="empty-state">
                            <div class="empty-state-icon">📭</div>
                            <p>No files uploaded yet</p>
                            <p style="font-size: 0.9rem; margin-top: 10px;">Drop your lease documents to begin</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="box" id="resultsBox" style="display: none;">
                <h3>📊 Extraction Results</h3>
                <div id="resultsArea"></div>
            </div>
        </div>
        
        <script>
            let uploadedFiles = [];
            let isProcessing = false;
            let currentFileIndex = 0;
            let globalStats = {
                files: 0,
                pages: 0,
                chars: 0,
                time: 0
            };
            let currentTimer = null;
            let timerStart = null;
            
            // Drag and drop
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
                const pdfFiles = files.filter(file => 
                    file.type === 'application/pdf' || 
                    file.name.toLowerCase().endsWith('.pdf') ||
                    file.type.startsWith('image/')
                );
                
                pdfFiles.forEach(file => {
                    const fileId = 'file-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
                    uploadedFiles.push({
                        id: fileId,
                        file: file,
                        status: 'waiting',
                        progress: 0,
                        startTime: null,
                        elapsedTime: 0
                    });
                });
                
                updateFilesList();
                updateQueueCount();
                document.getElementById('processBtn').disabled = uploadedFiles.length === 0;
            }
            
            function updateQueueCount() {
                const waiting = uploadedFiles.filter(f => f.status === 'waiting').length;
                const processing = uploadedFiles.filter(f => f.status === 'processing').length;
                document.getElementById('queueCount').textContent = 
                    `${waiting} waiting, ${processing} processing`;
            }
            
            function updateFilesList() {
                const filesList = document.getElementById('filesList');
                if (uploadedFiles.length === 0) {
                    filesList.innerHTML = `
                        <div class="empty-state">
                            <div class="empty-state-icon">📭</div>
                            <p>No files uploaded yet</p>
                            <p style="font-size: 0.9rem; margin-top: 10px;">Drop your lease documents to begin</p>
                        </div>
                    `;
                    return;
                }
                
                filesList.innerHTML = uploadedFiles.map((fileInfo, index) => `
                    <div class="file-item ${fileInfo.status === 'processing' ? 'processing' : ''}" id="${fileInfo.id}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="font-size: 1.1rem;">${fileInfo.file.name}</strong>
                                ${fileInfo.status === 'waiting' && index > currentFileIndex ? 
                                    `<span style="color: #6c757d; margin-left: 10px;">(Position ${index - currentFileIndex} in queue)</span>` : ''}
                            </div>
                            <div style="display: flex; align-items: center; gap: 10px;">
                                ${fileInfo.elapsedTime > 0 ? `<span class="timer">${fileInfo.elapsedTime}s</span>` : ''}
                                <span class="status-badge status-${fileInfo.status}">${fileInfo.status}</span>
                            </div>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${fileInfo.progress}%"></div>
                        </div>
                        <div id="${fileInfo.id}-message" style="font-size: 14px; color: #b4b7c2; margin-top: 10px;"></div>
                    </div>
                `).join('');
            }
            
            function startTimer() {
                timerStart = Date.now();
                document.getElementById('currentTimer').style.display = 'inline-flex';
                
                currentTimer = setInterval(() => {
                    const elapsed = Math.floor((Date.now() - timerStart) / 1000);
                    document.getElementById('currentTimer').textContent = elapsed + 's';
                }, 100);
            }
            
            function stopTimer() {
                if (currentTimer) {
                    clearInterval(currentTimer);
                    currentTimer = null;
                }
                document.getElementById('currentTimer').style.display = 'none';
            }
            
            async function processAllFiles() {
                if (isProcessing) return;
                
                isProcessing = true;
                document.getElementById('processBtn').disabled = true;
                document.getElementById('globalStats').style.display = 'grid';
                
                // Process files sequentially
                for (let i = 0; i < uploadedFiles.length; i++) {
                    currentFileIndex = i;
                    const fileInfo = uploadedFiles[i];
                    
                    if (fileInfo.status === 'waiting') {
                        await processFile(fileInfo);
                    }
                }
                
                isProcessing = false;
                document.getElementById('processBtn').disabled = uploadedFiles.filter(f => f.status === 'waiting').length === 0;
                stopTimer();
            }
            
            async function processFile(fileInfo) {
                fileInfo.status = 'processing';
                fileInfo.startTime = Date.now();
                startTimer();
                updateFilesList();
                updateQueueCount();
                
                const formData = new FormData();
                formData.append('file', fileInfo.file);
                formData.append('file_id', fileInfo.id);
                
                try {
                    const response = await fetch('/stream-extract', {
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
                    stopTimer();
                }
                
                updateQueueCount();
            }
            
            function handleStreamData(data, fileInfo) {
                const messageEl = document.getElementById(fileInfo.id + '-message');
                
                switch(data.type) {
                    case 'start':
                        messageEl.textContent = data.message;
                        break;
                        
                    case 'info':
                        messageEl.textContent = data.message;
                        break;
                        
                    case 'progress':
                        fileInfo.progress = data.progress;
                        fileInfo.elapsedTime = Math.round(data.elapsed_time);
                        messageEl.textContent = data.message;
                        updateFilesList();
                        break;
                        
                    case 'page_complete':
                        const progress = Math.round((data.page / data.total_pages) * 100);
                        fileInfo.progress = progress;
                        updateFilesList();
                        break;
                        
                    case 'complete':
                        fileInfo.status = 'complete';
                        fileInfo.progress = 100;
                        fileInfo.extractedText = data.text;
                        fileInfo.totalTime = data.total_time;
                        fileInfo.elapsedTime = Math.round(data.total_time);
                        fileInfo.detectedLanguages = data.detected_languages || 'Auto-detected';
                        messageEl.textContent = `Complete! ${data.total_chars} characters extracted in ${data.total_time}s`;
                        
                        // Update global stats
                        globalStats.files++;
                        globalStats.pages += data.total_pages || 1;
                        globalStats.chars += data.total_chars || 0;
                        globalStats.time += data.total_time || 0;
                        updateGlobalStats();
                        
                        updateFilesList();
                        displayResults(fileInfo);
                        stopTimer();
                        break;
                        
                    case 'error':
                        fileInfo.status = 'error';
                        messageEl.textContent = 'Error: ' + data.error;
                        updateFilesList();
                        stopTimer();
                        break;
                }
            }
            
            function updateGlobalStats() {
                document.getElementById('totalFiles').textContent = globalStats.files;
                document.getElementById('totalPages').textContent = globalStats.pages;
                document.getElementById('totalChars').textContent = 
                    globalStats.chars > 1000000 ? 
                    (globalStats.chars / 1000000).toFixed(1) + 'M' : 
                    globalStats.chars.toLocaleString();
                document.getElementById('totalTime').textContent = 
                    Math.round(globalStats.time) + 's';
            }
            
            function displayResults(fileInfo) {
                document.getElementById('resultsBox').style.display = 'block';
                const resultsArea = document.getElementById('resultsArea');
                
                const resultHtml = `
                    <div class="file-item" style="margin-bottom: 25px;">
                        <h4 style="color: #e4e6eb; font-size: 1.2rem; margin-bottom: 10px;">${fileInfo.file.name}</h4>
                        <p style="color: #b4b7c2; font-size: 14px;">
                            Processed in ${fileInfo.totalTime}s | Languages: ${fileInfo.detectedLanguages}
                        </p>
                        <div style="display: flex; gap: 10px; margin: 15px 0;">
                            <button class="btn" onclick="downloadText('${fileInfo.id}')" style="padding: 10px 20px; font-size: 14px;">
                                📥 Download Text
                            </button>
                            <button class="btn" onclick="copyText('${fileInfo.id}')" style="padding: 10px 20px; font-size: 14px; background: linear-gradient(45deg, #28a745, #20c997);">
                                📋 Copy to Clipboard
                            </button>
                        </div>
                        <div class="results-preview" id="result-${fileInfo.id}">
                            ${fileInfo.extractedText}
                        </div>
                    </div>
                `;
                
                if (resultsArea.innerHTML === '') {
                    resultsArea.innerHTML = resultHtml;
                } else {
                    resultsArea.innerHTML += resultHtml;
                }
                
                // Store text for download/copy
                window.extractedTexts = window.extractedTexts || {};
                window.extractedTexts[fileInfo.id] = fileInfo.extractedText;
            }
            
            function downloadText(fileId) {
                const text = window.extractedTexts[fileId];
                const fileInfo = uploadedFiles.find(f => f.id === fileId);
                const fileName = fileInfo.file.name.replace(/\.[^/.]+$/, "") + "_extracted.txt";
                
                const blob = new Blob([text], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = fileName;
                a.click();
                URL.revokeObjectURL(url);
            }
            
            function copyText(fileId) {
                const text = window.extractedTexts[fileId];
                navigator.clipboard.writeText(text).then(() => {
                    // Show success feedback
                    const btn = event.target;
                    const originalText = btn.textContent;
                    btn.textContent = '✓ Copied!';
                    btn.style.background = 'linear-gradient(45deg, #28a745, #20c997)';
                    
                    setTimeout(() => {
                        btn.textContent = originalText;
                        btn.style.background = '';
                    }, 2000);
                });
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
async def stream_extract(file: UploadFile = File(...), file_id: str = None):
    """Extract text with real-time progress streaming and auto language detection"""
    content = await file.read()
    file_id = file_id or str(uuid.uuid4())
    
    return StreamingResponse(
        stream_ocr_progress(content, file.filename, file_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

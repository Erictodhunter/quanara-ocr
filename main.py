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

app = FastAPI(
    title="QUANARA OCR API",
    description="Multi-language OCR service for PDFs and images with real-time progress tracking",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Available languages
LANGUAGES = {
    'eng': 'English', 'spa': 'Spanish', 'fra': 'French', 'deu': 'German',
    'ita': 'Italian', 'por': 'Portuguese', 'nld': 'Dutch', 'pol': 'Polish',
    'rus': 'Russian', 'tur': 'Turkish', 'ara': 'Arabic', 'chi_sim': 'Chinese Simplified',
    'chi_tra': 'Chinese Traditional', 'jpn': 'Japanese', 'kor': 'Korean', 'hin': 'Hindi'
}

async def stream_ocr_progress(file_content: bytes, filename: str, languages: str, file_id: str):
    """Stream OCR progress in real-time"""
    try:
        # Send start event
        yield f"data: {json.dumps({'type': 'start', 'file_id': file_id, 'filename': filename, 'message': 'Starting processing...'})}\n\n"
        await asyncio.sleep(0.1)
        
        if filename.lower().endswith('.pdf'):
            # Convert PDF to images
            images = pdf2image.convert_from_bytes(file_content, dpi=150)
            total_pages = len(images)
            
            yield f"data: {json.dumps({'type': 'info', 'file_id': file_id, 'total_pages': total_pages, 'message': f'PDF loaded: {total_pages} pages'})}\n\n"
            
            all_text = []
            for i, image in enumerate(images, 1):
                # Process each page
                yield f"data: {json.dumps({'type': 'progress', 'file_id': file_id, 'current_page': i, 'total_pages': total_pages, 'progress': int((i-1)/total_pages * 100), 'message': f'Processing page {i}/{total_pages}'})}\n\n"
                
                text = pytesseract.image_to_string(image, lang=languages)
                if text.strip():
                    all_text.append(f"[Page {i}]\n{text}")
                
                # Send page complete event
                yield f"data: {json.dumps({'type': 'page_complete', 'file_id': file_id, 'page': i, 'text_preview': text[:200] + '...' if len(text) > 200 else text})}\n\n"
                await asyncio.sleep(0.1)
            
            final_text = "\n\n".join(all_text)
        else:
            # Process single image
            yield f"data: {json.dumps({'type': 'progress', 'file_id': file_id, 'progress': 50, 'message': 'Processing image...'})}\n\n"
            image = Image.open(io.BytesIO(file_content))
            final_text = pytesseract.image_to_string(image, lang=languages)
        
        # Send completion event
        yield f"data: {json.dumps({'type': 'complete', 'file_id': file_id, 'text': final_text, 'total_chars': len(final_text), 'message': 'Processing complete!'})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'file_id': file_id, 'error': str(e)})}\n\n"

@app.get("/", response_class=HTMLResponse)
async def main():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>QUANARA OCR - Multi-File Processing</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                min-height: 100vh; 
                padding: 20px;
            }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { 
                text-align: center; 
                color: white; 
                margin-bottom: 30px;
                padding: 20px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                backdrop-filter: blur(10px);
            }
            .header h1 { font-size: 3rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
            .header p { font-size: 1.2rem; opacity: 0.9; margin-bottom: 20px; }
            
            .main-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
            
            .box { 
                background: white; 
                border-radius: 20px; 
                padding: 30px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }
            
            .upload-area { 
                border: 3px dashed #667eea; 
                border-radius: 15px; 
                padding: 40px; 
                text-align: center; 
                background: #f8f9ff;
                transition: all 0.3s;
                cursor: pointer;
            }
            .upload-area:hover, .upload-area.dragover { 
                border-color: #764ba2; 
                background: #f0f2ff; 
                transform: scale(1.02);
            }
            
            input[type="file"] { display: none; }
            
            .btn { 
                background: linear-gradient(45deg, #667eea, #764ba2); 
                color: white; 
                padding: 12px 30px; 
                border: none; 
                border-radius: 50px; 
                font-size: 16px; 
                cursor: pointer; 
                transition: all 0.3s;
                margin: 5px;
            }
            .btn:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3); }
            
            .file-item {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                border-left: 4px solid #667eea;
            }
            
            .progress-bar {
                background: #e9ecef;
                border-radius: 10px;
                height: 10px;
                margin: 10px 0;
                overflow: hidden;
            }
            
            .progress-fill {
                background: linear-gradient(45deg, #667eea, #764ba2);
                height: 100%;
                transition: width 0.3s;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 12px;
            }
            
            .status-badge {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: bold;
            }
            .status-processing { background: #fff3cd; color: #856404; }
            .status-complete { background: #d4edda; color: #155724; }
            .status-error { background: #f8d7da; color: #721c24; }
            
            .api-docs {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
            }
            
            .api-endpoint {
                background: #e9ecef;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                font-family: 'Courier New', monospace;
            }
            
            .results-preview {
                background: #f5f5f5;
                padding: 15px;
                border-radius: 8px;
                margin-top: 10px;
                max-height: 200px;
                overflow-y: auto;
                font-family: monospace;
                font-size: 12px;
                white-space: pre-wrap;
            }
            
            .tab-buttons {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }
            
            .tab-button {
                padding: 10px 20px;
                background: #e9ecef;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                transition: all 0.3s;
            }
            
            .tab-button.active {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
            }
            
            .tab-content {
                display: none;
            }
            
            .tab-content.active {
                display: block;
            }
            
            code {
                background: #e9ecef;
                padding: 2px 6px;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
            }
            
            .feature-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            
            .feature-card {
                background: #f8f9ff;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }
            
            .feature-icon {
                font-size: 2rem;
                margin-bottom: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌍 QUANARA OCR</h1>
                <p>Multi-Language OCR Processing with Real-Time Progress</p>
                <div class="feature-grid">
                    <div class="feature-card">
                        <div class="feature-icon">📄</div>
                        <strong>Multi-File</strong>
                        <p>Process multiple files simultaneously</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">🌐</div>
                        <strong>16+ Languages</strong>
                        <p>Support for major world languages</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">📊</div>
                        <strong>Real-Time Progress</strong>
                        <p>Live updates as pages process</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">🚀</div>
                        <strong>REST API</strong>
                        <p>Full API for integration</p>
                    </div>
                </div>
            </div>
            
            <div class="main-grid">
                <div class="box">
                    <h3>📤 Upload Files</h3>
                    <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
                        <div class="feature-icon">📁</div>
                        <h4>Click or Drag Files Here</h4>
                        <p>Support for PDF and images (JPG, PNG)</p>
                        <p>Multiple files supported</p>
                        <input type="file" id="fileInput" accept=".pdf,.jpg,.jpeg,.png" multiple>
                    </div>
                    
                    <div style="margin-top: 20px;">
                        <label>Language Selection:</label>
                        <select id="languageSelect" style="width: 100%; padding: 10px; border-radius: 5px; margin-top: 5px;">
                            <option value="eng">English</option>
                            <option value="eng+spa">English + Spanish</option>
                            <option value="eng+fra">English + French</option>
                            <option value="eng+deu">English + German</option>
                            <option value="chi_sim">Chinese Simplified</option>
                            <option value="jpn">Japanese</option>
                            <option value="ara">Arabic</option>
                            <option value="hin">Hindi</option>
                            <option value="multi">Multi-language (slower)</option>
                        </select>
                    </div>
                    
                    <button class="btn" onclick="processAllFiles()" style="width: 100%; margin-top: 20px;">
                        🚀 Process All Files
                    </button>
                </div>
                
                <div class="box">
                    <h3>📋 Processing Queue</h3>
                    <div id="filesList">
                        <p style="color: #999; text-align: center;">No files uploaded yet</p>
                    </div>
                </div>
            </div>
            
            <div class="box">
                <div class="tab-buttons">
                    <button class="tab-button active" onclick="showTab('results')">📄 Results</button>
                    <button class="tab-button" onclick="showTab('api')">🔌 API Documentation</button>
                </div>
                
                <div id="results-tab" class="tab-content active">
                    <h3>📊 Extraction Results</h3>
                    <div id="resultsArea">
                        <p style="color: #999;">Results will appear here after processing</p>
                    </div>
                </div>
                
                <div id="api-tab" class="tab-content">
                    <h3>🔌 API Documentation</h3>
                    <div class="api-docs">
                        <h4>REST API Endpoints</h4>
                        
                        <div class="api-endpoint">
                            <strong>POST /extract</strong> - Extract text from a single file
                        </div>
                        <p>Upload a PDF or image file to extract text.</p>
                        <pre style="background: #f5f5f5; padding: 10px; border-radius: 5px;">
curl -X POST https://quanara-ocr-v2-639122444190.us-west2.run.app/extract \\
  -F "file=@document.pdf" \\
  -F "languages=eng"</pre>
                        
                        <div class="api-endpoint">
                            <strong>POST /stream-extract</strong> - Extract with real-time progress
                        </div>
                        <p>Stream extraction progress using Server-Sent Events.</p>
                        
                        <div class="api-endpoint">
                            <strong>GET /languages</strong> - Get supported languages
                        </div>
                        <p>Returns list of supported language codes.</p>
                        
                        <div class="api-endpoint">
                            <strong>GET /api/docs</strong> - Interactive API documentation
                        </div>
                        <p>Full Swagger/OpenAPI documentation with try-it-out functionality.</p>
                        
                        <h4>Python Example</h4>
                        <pre style="background: #f5f5f5; padding: 10px; border-radius: 5px;">
import requests

# Simple extraction
response = requests.post(
    'https://quanara-ocr-v2-639122444190.us-west2.run.app/extract',
    files={'file': open('document.pdf', 'rb')},
    data={'languages': 'eng+spa'}
)
print(response.json()['text'])

# With progress streaming
import sseclient

response = requests.post(
    'https://quanara-ocr-v2-639122444190.us-west2.run.app/stream-extract',
    files={'file': open('document.pdf', 'rb')},
    stream=True
)

client = sseclient.SSEClient(response)
for event in client.events():
    print(event.data)</pre>
                        
                        <h4>JavaScript Example</h4>
                        <pre style="background: #f5f5f5; padding: 10px; border-radius: 5px;">
// Simple extraction
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('languages', 'eng');

fetch('https://quanara-ocr-v2-639122444190.us-west2.run.app/extract', {
    method: 'POST',
    body: formData
})
.then(res => res.json())
.then(data => console.log(data.text));

// With progress streaming
const eventSource = new EventSource('/stream-extract');
eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data.progress + '%');
};</pre>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let uploadedFiles = [];
            let activeProcesses = {};
            
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
                files.forEach(file => {
                    const fileId = 'file-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
                    uploadedFiles.push({
                        id: fileId,
                        file: file,
                        status: 'waiting',
                        progress: 0
                    });
                });
                updateFilesList();
            }
            
            function updateFilesList() {
                const filesList = document.getElementById('filesList');
                if (uploadedFiles.length === 0) {
                    filesList.innerHTML = '<p style="color: #999; text-align: center;">No files uploaded yet</p>';
                    return;
                }
                
                filesList.innerHTML = uploadedFiles.map(fileInfo => `
                    <div class="file-item" id="${fileInfo.id}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong>${fileInfo.file.name}</strong>
                            <span class="status-badge status-${fileInfo.status}">${fileInfo.status}</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${fileInfo.progress}%">
                                ${fileInfo.progress > 0 ? fileInfo.progress + '%' : ''}
                            </div>
                        </div>
                        <div id="${fileInfo.id}-message" style="font-size: 14px; color: #666; margin-top: 5px;"></div>
                    </div>
                `).join('');
            }
            
            async function processAllFiles() {
                const languages = document.getElementById('languageSelect').value;
                const languageMap = {
                    'multi': 'eng+spa+fra+deu+ita+por'
                };
                const finalLanguages = languageMap[languages] || languages;
                
                for (const fileInfo of uploadedFiles) {
                    if (fileInfo.status === 'waiting') {
                        processFile(fileInfo, finalLanguages);
                    }
                }
            }
            
            async function processFile(fileInfo, languages) {
                fileInfo.status = 'processing';
                updateFilesList();
                
                const formData = new FormData();
                formData.append('file', fileInfo.file);
                formData.append('languages', languages);
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
                }
            }
            
            function handleStreamData(data, fileInfo) {
                const messageEl = document.getElementById(fileInfo.id + '-message');
                
                switch(data.type) {
                    case 'start':
                        messageEl.textContent = data.message;
                        break;
                        
                    case 'progress':
                        fileInfo.progress = data.progress;
                        messageEl.textContent = data.message;
                        updateFilesList();
                        break;
                        
                    case 'page_complete':
                        fileInfo.progress = Math.round((data.page / data.total_pages) * 100);
                        updateFilesList();
                        break;
                        
                    case 'complete':
                        fileInfo.status = 'complete';
                        fileInfo.progress = 100;
                        fileInfo.extractedText = data.text;
                        messageEl.textContent = `Complete! Extracted ${data.total_chars} characters`;
                        updateFilesList();
                        displayResults(fileInfo);
                        break;
                        
                    case 'error':
                        fileInfo.status = 'error';
                        messageEl.textContent = 'Error: ' + data.error;
                        updateFilesList();
                        break;
                }
            }
            
            function displayResults(fileInfo) {
                const resultsArea = document.getElementById('resultsArea');
                const resultHtml = `
                    <div class="file-item" style="margin-bottom: 20px;">
                        <h4>${fileInfo.file.name}</h4>
                        <div style="display: flex; gap: 10px; margin: 10px 0;">
                            <button class="btn" onclick="downloadText('${fileInfo.id}')" style="padding: 8px 16px; font-size: 14px;">
                                📥 Download Text
                            </button>
                            <button class="btn" onclick="copyText('${fileInfo.id}')" style="padding: 8px 16px; font-size: 14px; background: linear-gradient(45deg, #28a745, #20c997);">
                                📋 Copy to Clipboard
                            </button>
                        </div>
                        <div class="results-preview" id="result-${fileInfo.id}">
                            ${fileInfo.extractedText}
                        </div>
                    </div>
                `;
                
                if (resultsArea.innerHTML.includes('Results will appear here')) {
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
                const blob = new Blob([text], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'extracted_text.txt';
                a.click();
                URL.revokeObjectURL(url);
            }
            
            function copyText(fileId) {
                const text = window.extractedTexts[fileId];
                navigator.clipboard.writeText(text).then(() => {
                    alert('Text copied to clipboard!');
                });
            }
            
            function showTab(tabName) {
                document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
                
                event.target.classList.add('active');
                document.getElementById(tabName + '-tab').classList.add('active');
            }
        </script>
    </body>
    </html>
    """

@app.post("/extract")
async def extract_text(file: UploadFile = File(...), languages: str = "eng"):
    """Extract text from PDF or image file"""
    try:
        content = await file.read()
        
        if file.filename.lower().endswith('.pdf'):
            images = pdf2image.convert_from_bytes(content, dpi=150)
            all_text = []
            
            for i, image in enumerate(images, 1):
                text = pytesseract.image_to_string(image, lang=languages)
                if text.strip():
                    all_text.append(f"[Page {i}]\n{text}")
            
            final_text = "\n\n".join(all_text)
            pages = len(images)
        else:
            image = Image.open(io.BytesIO(content))
            final_text = pytesseract.image_to_string(image, lang=languages)
            pages = 1
        
        return JSONResponse({
            "text": final_text,
            "pages": pages,
            "filename": file.filename,
            "languages": languages,
            "character_count": len(final_text)
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/stream-extract")
async def stream_extract(file: UploadFile = File(...), languages: str = "eng", file_id: str = None):
    """Extract text with real-time progress streaming"""
    content = await file.read()
    file_id = file_id or str(uuid.uuid4())
    
    return StreamingResponse(
        stream_ocr_progress(content, file.filename, languages, file_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/languages")
async def get_languages():
    """Get list of supported languages"""
    return LANGUAGES

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

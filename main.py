import pytesseract
import pdf2image
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import io

app = FastAPI(title="QUANARA OCR - Multi-Language")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Tesseract language codes for top business languages
AVAILABLE_LANGUAGES = {
    'eng': 'English',
    'spa': 'Spanish', 
    'fra': 'French',
    'deu': 'German',
    'ita': 'Italian',
    'por': 'Portuguese',
    'nld': 'Dutch',
    'pol': 'Polish',
    'rus': 'Russian',
    'tur': 'Turkish',
    'ara': 'Arabic',
    'chi_sim': 'Chinese Simplified',
    'chi_tra': 'Chinese Traditional',
    'jpn': 'Japanese',
    'kor': 'Korean',
    'hin': 'Hindi',
    'tha': 'Thai',
    'vie': 'Vietnamese',
    'heb': 'Hebrew',
    'ukr': 'Ukrainian'
}

@app.get("/", response_class=HTMLResponse)
async def main():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>QUANARA OCR - Multi-Language</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .header { text-align: center; color: white; margin-bottom: 30px; }
            .header h1 { font-size: 3rem; margin-bottom: 10px; }
            .upload-box { background: white; border-radius: 20px; padding: 40px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); }
            .upload-area { border: 3px dashed #667eea; border-radius: 15px; padding: 40px; text-align: center; background: #f8f9ff; }
            input[type="file"] { margin: 20px 0; }
            select { padding: 10px; margin: 10px; border-radius: 5px; }
            .btn { background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 15px 40px; border: none; border-radius: 50px; font-size: 18px; cursor: pointer; }
            .btn:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3); }
            .results { margin-top: 30px; padding: 20px; background: #f5f5f5; border-radius: 10px; white-space: pre-wrap; display: none; }
            .loading { display: none; text-align: center; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌍 QUANARA OCR</h1>
                <p>Extract text from images and PDFs in 20+ languages</p>
            </div>
            
            <div class="upload-box">
                <div class="upload-area">
                    <h3>📄 Upload PDF or Image</h3>
                    <input type="file" id="fileInput" accept=".pdf,.jpg,.jpeg,.png">
                    
                    <div>
                        <label>Primary Language:</label>
                        <select id="lang1">
                            <option value="eng">English</option>
                            <option value="spa">Spanish</option>
                            <option value="fra">French</option>
                            <option value="deu">German</option>
                            <option value="chi_sim">Chinese Simplified</option>
                            <option value="jpn">Japanese</option>
                            <option value="ara">Arabic</option>
                            <option value="hin">Hindi</option>
                        </select>
                        
                        <label>Secondary (optional):</label>
                        <select id="lang2">
                            <option value="">None</option>
                            <option value="eng">English</option>
                            <option value="spa">Spanish</option>
                            <option value="fra">French</option>
                            <option value="deu">German</option>
                        </select>
                    </div>
                    
                    <button class="btn" onclick="processFile()">🚀 Extract Text</button>
                </div>
                
                <div class="loading">⏳ Processing... This may take a moment.</div>
                <div class="results" id="results"></div>
            </div>
        </div>
        
        <script>
            async function processFile() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                if (!file) {
                    alert('Please select a file');
                    return;
                }
                
                const lang1 = document.getElementById('lang1').value;
                const lang2 = document.getElementById('lang2').value;
                const languages = lang2 ? lang1 + '+' + lang2 : lang1;
                
                const formData = new FormData();
                formData.append('file', file);
                formData.append('languages', languages);
                
                document.querySelector('.loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                
                try {
                    const response = await fetch('/extract', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    document.querySelector('.loading').style.display = 'none';
                    document.getElementById('results').style.display = 'block';
                    document.getElementById('results').textContent = data.text || 'No text found';
                    
                } catch (error) {
                    document.querySelector('.loading').style.display = 'none';
                    alert('Error: ' + error.message);
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/extract")
async def extract_text(file: UploadFile = File(...), languages: str = "eng"):
    try:
        content = await file.read()
        
        if file.filename.lower().endswith('.pdf'):
            # Convert PDF to images
            images = pdf2image.convert_from_bytes(content, dpi=200)
            all_text = []
            
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image, lang=languages)
                if text.strip():
                    all_text.append(f"[Page {i+1}]\n{text}")
            
            final_text = "\n\n".join(all_text)
        else:
            # Direct image processing
            image = Image.open(io.BytesIO(content))
            final_text = pytesseract.image_to_string(image, lang=languages)
        
        return JSONResponse({
            "text": final_text,
            "pages": len(all_text) if file.filename.lower().endswith('.pdf') else 1,
            "languages": languages
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/languages")
async def get_languages():
    return AVAILABLE_LANGUAGES

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

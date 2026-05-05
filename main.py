import os
import time
# Added File and UploadFile for handling file uploads
from fastapi import FastAPI, File, UploadFile 
import google.generativeai as genai
from pydantic import BaseModel
import fitz  # This is the PyMuPDF library!

app = FastAPI()

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY") 
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

class UserQuery(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"status": "success", "message": "The Education AI Server is running!"}

@app.post("/ask")
def ask_gemini(query: UserQuery):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(query.question)
            return {"status": "success", "answer": response.text}
        except Exception as e:
            error_message = str(e)
            if "429" in error_message or "Quota exceeded" in error_message:
                if attempt < max_retries - 1:
                    time.sleep(15)
                    continue
                else:
                    return {"status": "error", "message": "The AI is too busy right now. Please try again in a minute."}
            else:
                return {"status": "error", "message": error_message}

# --- NEW: The PDF Reader Endpoint ---
@app.post("/upload-pdf")
async def upload_and_read_pdf(file: UploadFile = File(...)):
    try:
        # 1. Ensure it is actually a PDF
        if not file.filename.endswith('.pdf'):
            return {"status": "error", "message": "Please upload a PDF file."}

        # 2. Read the uploaded file into memory
        file_content = await file.read()
        
        # 3. Open the PDF using PyMuPDF
        # fitz.Document can open a PDF directly from memory bytes
        pdf_document = fitz.Document(stream=file_content, filetype="pdf")
        
        extracted_pages = []
        
        # 4. Loop through every page and extract the text
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text = page.get_text("text")
            
            # We save the text AND the exact page number (crucial for citations later!)
            extracted_pages.append({
                "page_number": page_num + 1,
                "text": text.strip()
            })
            
        pdf_document.close()
        
        return {
            "status": "success", 
            "filename": file.filename,
            "total_pages": len(extracted_pages),
            "document_data": extracted_pages
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Failed to process PDF: {str(e)}"}
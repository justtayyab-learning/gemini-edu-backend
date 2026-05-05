import os
import time
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
import google.generativeai as genai
from pydantic import BaseModel
import fitz  

app = FastAPI()

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY") 
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# --- OUR TEMPORARY DATABASE ---
# In the future, this would be Firebase or a Vector DB. 
# For now, it lives in the server's memory.
documents_db = {}

# --- DATA MODELS ---
class UserQuery(BaseModel):
    question: str

class DocumentQuery(BaseModel):
    document_id: str
    question: str

@app.get("/")
def read_root():
    return {"status": "success", "message": "The Education AI Server is running!"}

# 1. The Standard AI Chat
@app.post("/ask")
def ask_gemini(query: UserQuery):
    try:
        response = model.generate_content(query.question)
        return {"status": "success", "answer": response.text}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 2. Upload and Save the PDF
@app.post("/upload-pdf")
async def upload_and_read_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.pdf'):
            return {"status": "error", "message": "Please upload a PDF file."}

        file_content = await file.read()
        pdf_document = fitz.Document(stream=file_content, filetype="pdf")
        
        extracted_pages = []
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text = page.get_text("text").strip()
            if text: # Only save pages that actually have text
                extracted_pages.append({
                    "page_number": page_num + 1,
                    "text": text
                })
        pdf_document.close()
        
        # --- NEW: Save to our temporary database ---
        doc_id = str(uuid.uuid4()) # Generate a random unique ID
        documents_db[doc_id] = {
            "filename": file.filename,
            "pages": extracted_pages
        }
        
        return {
            "status": "success", 
            "message": "PDF processed and saved!",
            "document_id": doc_id,  # The frontend needs this ID!
            "filename": file.filename,
            "total_pages": len(extracted_pages)
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Failed to process PDF: {str(e)}"}

# 3. The NotebookLM "Strict Citation" Chat
@app.post("/chat-with-pdf")
def chat_with_pdf(query: DocumentQuery):
    try:
        # Check if the document exists in our database
        if query.document_id not in documents_db:
            raise HTTPException(status_code=404, detail="Document not found.")
            
        doc_data = documents_db[query.document_id]
        
        # Format the PDF text into a single string with page tags
        context_string = ""
        for page in doc_data["pages"]:
            context_string += f"\n--- [Page {page['page_number']}] ---\n{page['text']}\n"

        # --- THE MASTER PROMPT ---
        # This is where the magic happens. We force Gemini to play by the rules.
        prompt = f"""You are an expert academic tutor. You are helping a student study from a document named '{doc_data['filename']}'.
        
        CRITICAL RULES:
        1. Answer the user's question USING ONLY the Source Context provided below.
        2. If the answer cannot be found in the Source Context, you must say exactly: "I cannot find this information in the uploaded document." Do not guess.
        3. Every time you state a fact or pull information, you MUST cite the exact page number using this format: [Page X]. 
        
        Source Context:
        {context_string}
        
        User Question: {query.question}
        """
        
        # Send the giant prompt to Gemini
        response = model.generate_content(prompt)
        
        return {
            "status": "success", 
            "answer": response.text
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
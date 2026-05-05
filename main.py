import os
import time
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
import google.generativeai as genai
from pydantic import BaseModel
import fitz  
# --- NEW: Firebase Tools ---
import firebase_admin
from firebase_admin import credentials, firestore

app = FastAPI()

# --- 1. SETUP GEMINI ---
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY") 
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# --- 2. SETUP FIREBASE DATABASE ---
# We build the secure key using the environment variables from Render
firebase_credentials = {
    "type": "service_account",
    "project_id": os.environ.get("FIREBASE_PROJECT_ID"),
    "private_key_id": "",
    # We use .replace to fix how Render handles new lines in the secret key
    "private_key": os.environ.get("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n'),
    "client_email": os.environ.get("FIREBASE_CLIENT_EMAIL"),
    "client_id": "",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{os.environ.get('FIREBASE_CLIENT_EMAIL')}"
}

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_credentials)
    firebase_admin.initialize_app(cred)

# Connect to Firestore
db = firestore.client()

# --- DATA MODELS ---
class UserQuery(BaseModel):
    question: str

class DocumentQuery(BaseModel):
    document_id: str
    question: str

@app.get("/")
def read_root():
    return {"status": "success", "message": "The Education AI Server is connected to Firebase!"}

@app.post("/ask")
def ask_gemini(query: UserQuery):
    try:
        response = model.generate_content(query.question)
        return {"status": "success", "answer": response.text}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- UPDATED: Save to Firebase ---
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
            if text:
                extracted_pages.append({
                    "page_number": page_num + 1,
                    "text": text
                })
        pdf_document.close()
        
        doc_id = str(uuid.uuid4())
        
        # Save the data PERMANENTLY to Firestore
        doc_ref = db.collection('documents').document(doc_id)
        doc_ref.set({
            "filename": file.filename,
            "pages": extracted_pages,
            "uploaded_at": firestore.SERVER_TIMESTAMP
        })
        
        return {
            "status": "success", 
            "message": "PDF saved to Firebase permanently!",
            "document_id": doc_id,
            "filename": file.filename
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Failed to process PDF: {str(e)}"}

# --- UPDATED: Read from Firebase ---
@app.post("/chat-with-pdf")
def chat_with_pdf(query: DocumentQuery):
    try:
        # Search Firebase for the document ID
        doc_ref = db.collection('documents').document(query.document_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Document not found in database.")
            
        doc_data = doc.to_dict()
        
        context_string = ""
        for page in doc_data.get("pages", []):
            context_string += f"\n--- [Page {page['page_number']}] ---\n{page['text']}\n"

        prompt = f"""You are an expert academic tutor. You are helping a student study from a document named '{doc_data.get('filename')}'.
        
        CRITICAL RULES:
        1. Answer the user's question USING ONLY the Source Context provided below.
        2. If the answer cannot be found in the Source Context, say exactly: "I cannot find this information in the uploaded document."
        3. Every time you state a fact, you MUST cite the exact page number using this format: [Page X]. 
        
        Source Context:
        {context_string}
        
        User Question: {query.question}
        """
        
        response = model.generate_content(prompt)
        return {"status": "success", "answer": response.text}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
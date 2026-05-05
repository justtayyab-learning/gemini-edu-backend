import os
import time
import uuid
import re
from fastapi import FastAPI, File, UploadFile, HTTPException
import google.generativeai as genai
from pydantic import BaseModel
import fitz  
import firebase_admin
from firebase_admin import credentials, firestore

# --- NEW: YouTube Library ---
from youtube_transcript_api import YouTubeTranscriptApi

app = FastAPI()

# --- 1. SETUP GEMINI ---
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY") 
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# --- 2. SETUP FIREBASE DATABASE ---
firebase_credentials = {
    "type": "service_account",
    "project_id": os.environ.get("FIREBASE_PROJECT_ID"),
    "private_key_id": "",
    "private_key": os.environ.get("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n'),
    "client_email": os.environ.get("FIREBASE_CLIENT_EMAIL"),
    "client_id": "",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{os.environ.get('FIREBASE_CLIENT_EMAIL')}"
}

if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_credentials)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# --- DATA MODELS ---
class UserQuery(BaseModel):
    question: str

class DocumentQuery(BaseModel):
    document_id: str
    question: str

# NEW: Model for the YouTube link
class YouTubeQuery(BaseModel):
    url: str

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

# --- NEW: YouTube Ingestion Endpoint ---
@app.post("/ingest-youtube")
def ingest_youtube_video(query: YouTubeQuery):
    try:
        # 1. Extract the 11-character Video ID from the URL
        video_id = None
        if "v=" in query.url:
            video_id = query.url.split("v=")[1][:11]
        elif "youtu.be/" in query.url:
            video_id = query.url.split("youtu.be/")[1][:11]
            
        if not video_id:
            return {"status": "error", "message": "Could not find a valid YouTube video ID in that URL."}

        # 2. Download the transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # 3. Format the transcript into "Pages" (chunks of 20 sentences)
        extracted_pages = []
        current_text = ""
        page_num = 1
        
        for index, item in enumerate(transcript_list):
            # Clean up the text and add it to our chunk
            text_line = item['text'].replace('\n', ' ')
            current_text += f"{text_line} "
            
            # Every 20 lines (or at the very end of the video), save it as a "page"
            if (index + 1) % 20 == 0 or (index + 1) == len(transcript_list):
                extracted_pages.append({
                    "page_number": page_num,
                    "text": current_text.strip()
                })
                page_num += 1
                current_text = "" # Reset for the next page
        
        # 4. Save to Firebase exactly like a PDF!
        doc_id = str(uuid.uuid4())
        filename = f"YouTube_Video_{video_id}"
        
        doc_ref = db.collection('documents').document(doc_id)
        doc_ref.set({
            "filename": filename,
            "pages": extracted_pages,
            "uploaded_at": firestore.SERVER_TIMESTAMP
        })
        
        return {
            "status": "success", 
            "message": "YouTube Video Transcript processed and saved to Firebase!",
            "document_id": doc_id,
            "filename": filename,
            "total_pages": len(extracted_pages)
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Failed to process YouTube Video: {str(e)}"}

@app.post("/chat-with-pdf")
def chat_with_pdf(query: DocumentQuery):
    try:
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
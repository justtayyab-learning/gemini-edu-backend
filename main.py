import os
import time
import uuid
import re
import json
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
import google.generativeai as genai
from pydantic import BaseModel
import fitz  
import firebase_admin
from firebase_admin import credentials, firestore
from youtube_transcript_api import YouTubeTranscriptApi

# --- Google TTS Libraries ---
from google.oauth2 import service_account
from google.cloud import texttospeech

# --- Auto-Researcher & Scraper Libraries ---
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

app = FastAPI()

# --- 1. SETUP GEMINI ---
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY") 
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# --- 2. SETUP FIREBASE & GOOGLE CLOUD ---
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

# Init Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_credentials)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Init Google TTS
gcp_credentials = service_account.Credentials.from_service_account_info(firebase_credentials)
tts_client = texttospeech.TextToSpeechClient(credentials=gcp_credentials)

# --- DATA MODELS ---
class UserQuery(BaseModel):
    question: str
class DocumentQuery(BaseModel):
    document_id: str
    question: str
class YouTubeQuery(BaseModel):
    url: str
class TextUploadQuery(BaseModel):
    title: str
    text_content: str
class PodcastQuery(BaseModel):
    document_id: str
class ResearchQuery(BaseModel):
    topic: str

@app.get("/")
def read_root():
    return {"status": "success", "message": "The Education AI Server is ready!"}

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
                extracted_pages.append({"page_number": page_num + 1, "text": text})
        pdf_document.close()
        doc_id = str(uuid.uuid4())
        doc_ref = db.collection('documents').document(doc_id)
        doc_ref.set({"filename": file.filename, "pages": extracted_pages, "uploaded_at": firestore.SERVER_TIMESTAMP})
        return {"status": "success", "document_id": doc_id, "filename": file.filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/ingest-youtube")
def ingest_youtube_video(query: YouTubeQuery):
    try:
        video_id = None
        if "v=" in query.url: video_id = query.url.split("v=")[1][:11]
        elif "youtu.be/" in query.url: video_id = query.url.split("youtu.be/")[1][:11]
        if not video_id: return {"status": "error", "message": "Could not find valid YouTube ID."}
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id)
        extracted_pages = []
        current_text = ""
        page_num = 1
        for index, item in enumerate(transcript_list):
            current_text += f"{item['text'].replace('\n', ' ')} "
            if (index + 1) % 20 == 0 or (index + 1) == len(transcript_list):
                extracted_pages.append({"page_number": page_num, "text": current_text.strip()})
                page_num += 1
                current_text = "" 
        doc_id = str(uuid.uuid4())
        filename = f"YouTube_{video_id}"
        doc_ref = db.collection('documents').document(doc_id)
        doc_ref.set({"filename": filename, "pages": extracted_pages, "uploaded_at": firestore.SERVER_TIMESTAMP})
        return {"status": "success", "document_id": doc_id, "filename": filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/upload-text")
def upload_raw_text(query: TextUploadQuery):
    try:
        words = query.text_content.split()
        chunk_size = 500
        extracted_pages = []
        page_num = 1
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            extracted_pages.append({"page_number": page_num, "text": chunk})
            page_num += 1
        doc_id = str(uuid.uuid4())
        doc_ref = db.collection('documents').document(doc_id)
        doc_ref.set({"filename": query.title, "pages": extracted_pages, "uploaded_at": firestore.SERVER_TIMESTAMP})
        return {"status": "success", "document_id": doc_id, "filename": query.title}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- BULLETPROOF AUTO-RESEARCHER ---
@app.post("/auto-research")
def auto_research(query: ResearchQuery):
    try:
        topic = query.topic
        print(f"--- STARTING RESEARCH FOR: {topic} ---")
        
        # 1. Get Search Results
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{topic} facts history", max_results=5))
        
        # DEBUG: Print to Render logs so you can see if search worked
        print(f"Search found {len(results)} results.")

        if not results:
            return {"status": "error", "message": "Search engine returned zero results. Try a different topic."}

        combined_research = f"--- RESEARCH DATA FOR: {topic} ---\n\n"
        
        for article in results:
            title = article.get('title', 'No Title')
            snippet = article.get('body', 'No Snippet')
            url = article.get('href', 'No URL')
            
            # Save the summary immediately
            combined_research += f"FACT: {title}\nSUMMARY: {snippet}\nSOURCE: {url}\n\n"
            print(f"Captured snippet from: {title}")

        # 2. Save to Firestore
        words = combined_research.split()
        extracted_pages = []
        for i in range(0, len(words), 500):
            chunk = " ".join(words[i:i + 500])
            extracted_pages.append({"page_number": (i // 500) + 1, "text": chunk})
            
        doc_id = str(uuid.uuid4())
        db.collection('documents').document(doc_id).set({
            "filename": f"Research: {topic}", 
            "pages": extracted_pages, 
            "uploaded_at": firestore.SERVER_TIMESTAMP
        })
        
        # We send a BIGGER preview to the frontend now
        return {
            "status": "success", 
            "document_id": doc_id,
            "filename": f"Research: {topic}",
            "scraped_text_preview": combined_research[0:1000] # Increased to 1000 chars
        }
    except Exception as e:
        print(f"RESEARCH ERROR: {str(e)}")
        return {"status": "error", "message": f"Scraper Error: {str(e)}"}
@app.post("/chat-with-document")
def chat_with_document(query: DocumentQuery):
    try:
        doc_ref = db.collection('documents').document(query.document_id)
        doc = doc_ref.get()
        if not doc.exists: raise HTTPException(status_code=404, detail="Document not found.")
        doc_data = doc.to_dict()
        context_string = "".join([p['text'] for p in doc_data.get("pages", [])])

        prompt = f"""Use the following context to answer the student's question. 
        Context: {context_string}
        Question: {query.question}
        If the answer isn't in the context, say you don't know."""
        
        response = model.generate_content(prompt)
        return {"status": "success", "answer": response.text}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/generate-podcast-audio")
def generate_podcast_audio(query: PodcastQuery):
    try:
        doc_ref = db.collection('documents').document(query.document_id)
        doc = doc_ref.get()
        if not doc.exists: raise HTTPException(status_code=404, detail="Document not found.")
        doc_data = doc.to_dict()
        context_string = "".join([p['text'] for p in doc_data.get("pages", [])])
        prompt = f"Write a 4-line podcast script about this: {context_string}. Return ONLY a JSON array of objects with 'speaker' and 'text'."
        response = model.generate_content(prompt)
        clean_json_str = response.text.replace("```json", "").replace("```", "").strip()
        script_data = json.loads(clean_json_str)
        audio_segments = []
        for line in script_data:
            speaker, text = line.get("speaker"), line.get("text")
            voice_name = "en-US-Journey-D" if speaker == "Alex" else "en-US-Journey-F"
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(language_code="en-US", name=voice_name)
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
            tts_response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            audio_base64 = base64.b64encode(tts_response.audio_content).decode('utf-8')
            audio_segments.append({"speaker": speaker, "text": text, "audio_base64": audio_base64})
        return {"status": "success", "podcast": audio_segments}
    except Exception as e:
        return {"status": "error", "message": str(e)}
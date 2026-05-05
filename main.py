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
from google.oauth2 import service_account
from google.cloud import texttospeech
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

app = FastAPI()

# --- 1. SETUP GEMINI (UPDATED FOR MAY 2026) ---
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY") 
genai.configure(api_key=GOOGLE_API_KEY)

# Use 'gemini-3-flash-preview' for the current 2026 Developer Preview
# Alternatively, 'gemini-flash-latest' will always point to this version.
model = genai.GenerativeModel('gemini-3-flash-preview') 

# --- 2. SETUP FIREBASE ---
firebase_credentials = {
    "type": "service_account",
    "project_id": os.environ.get("FIREBASE_PROJECT_ID"),
    "private_key": os.environ.get("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n'),
    "client_email": os.environ.get("FIREBASE_CLIENT_EMAIL"),
    "token_uri": "https://oauth2.googleapis.com/token",
}
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_credentials)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# --- 3. SETUP TTS ---
gcp_credentials = service_account.Credentials.from_service_account_info(firebase_credentials)
tts_client = texttospeech.TextToSpeechClient(credentials=gcp_credentials)

# --- MODELS ---
class DocumentQuery(BaseModel):
    document_id: str
    question: str
class ResearchQuery(BaseModel):
    topic: str
class TextUploadQuery(BaseModel):
    title: str
    text_content: str
class PodcastQuery(BaseModel):
    document_id: str

@app.get("/")
def read_root(): return {"status": "success", "message": "Gemini 3 Flash Online"}

# --- AUTO-RESEARCH ---
@app.post("/auto-research")
def auto_research(query: ResearchQuery):
    try:
        topic = query.topic
        search_query = f"{topic} history facts discovery english"
        combined_research = f"--- RESEARCH DOSSIER: {topic} ---\n\n"
        
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, region="wt-wt", max_results=5))
        
        for article in results:
            title = article.get('title', 'Fact')
            snippet = article.get('body', '')
            combined_research += f"SOURCE: {title}\nFACT: {snippet}\n\n"

        words = combined_research.split()
        extracted_pages = [{"page_number": (i//500)+1, "text": " ".join(words[i:i+500])} for i in range(0, len(words), 500)]
            
        doc_id = str(uuid.uuid4())
        db.collection('documents').document(doc_id).set({
            "filename": f"Research: {topic}", 
            "pages": extracted_pages, 
            "uploaded_at": firestore.SERVER_TIMESTAMP
        })
        
        return {
            "status": "success", 
            "document_id": doc_id,
            "filename": f"Research: {topic}",
            "scraped_text_preview": combined_research[:800]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- CHAT TUTOR (IRONCLAD VERSION) ---
@app.post("/chat-with-document")
def chat_with_document(query: DocumentQuery):
    try:
        doc_ref = db.collection('documents').document(query.document_id)
        doc = doc_ref.get()
        if not doc.exists: return {"status": "error", "message": "Memory not found."}
            
        doc_data = doc.to_dict()
        context_string = "".join([p['text'] for p in doc_data.get("pages", [])])

        prompt = f"Context: {context_string}\n\nQuestion: {query.question}\n\nAnswer using only the context provided. If the information is missing, state that you cannot find it."
        
        response = model.generate_content(prompt)
        
        # Ensure we return the content as 'answer' to match renderer.js
        ai_text = response.text if response.text else "The model returned an empty response."
        return {"status": "success", "answer": ai_text}

    except Exception as e:
        return {"status": "error", "message": f"API Error: {str(e)}"}

# --- UPLOAD & PODCAST ---
@app.post("/upload-text")
def upload_text(query: TextUploadQuery):
    doc_id = str(uuid.uuid4())
    db.collection('documents').document(doc_id).set({"filename": query.title, "pages": [{"page_number": 1, "text": query.text_content}], "uploaded_at": firestore.SERVER_TIMESTAMP})
    return {"status": "success", "document_id": doc_id}

@app.post("/generate-podcast-audio")
def generate_podcast(query: PodcastQuery):
    try:
        doc = db.collection('documents').document(query.document_id).get().to_dict()
        txt = "".join([p['text'] for p in doc.get("pages", [])])
        prompt = f"Write a 4-line script about this: {txt}. Return ONLY a JSON array of objects with 'speaker' and 'text'."
        res = model.generate_content(prompt)
        script = json.loads(res.text.replace("```json", "").replace("```", "").strip())
        audio_segments = []
        for line in script:
            voice = "en-US-Journey-D" if line['speaker'] == "Alex" else "en-US-Journey-F"
            synth = tts_client.synthesize_speech(input=texttospeech.SynthesisInput(text=line['text']), voice=texttospeech.VoiceSelectionParams(language_code="en-US", name=voice), audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3))
            audio_base64 = base64.b64encode(synth.audio_content).decode('utf-8')
            audio_segments.append({"audio_base64": audio_base64})
        return {"status": "success", "podcast": audio_segments}
    except Exception as e: return {"status": "error", "message": str(e)}
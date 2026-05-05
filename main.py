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

# --- 1. SETUP GEMINI ---
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY") 
genai.configure(api_key=GOOGLE_API_KEY)
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
def read_root(): return {"status": "success", "message": "Reliability Engine Online"}

# --- AUTO-RESEARCH (WITH AUTO-RETRY) ---
@app.post("/auto-research")
def auto_research(query: ResearchQuery):
    topic = query.topic
    max_retries = 3
    results = []
    
    # Try different search variations if the first one fails
    queries = [
        f'"{topic}" history discovery Fleming', # High precision
        f'{topic} facts history archive',        # Medium precision
        f'{topic} wikipedia'                    # Backup
    ]

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}: Searching for {queries[attempt]}")
            with DDGS() as ddgs:
                results = list(ddgs.text(queries[attempt], region="wt-wt", max_results=8))
            
            if results:
                # Filter out the commercial junk we saw earlier
                filtered_results = [r for r in results if not any(x in r.get('href', '') for x in ["sky", "google", "ads", "zhihu", "amazon"])]
                if filtered_results:
                    results = filtered_results
                    break # Success!
            
            time.sleep(1) # Small pause between retries to avoid rate limits
        except Exception as e:
            print(f"Search attempt failed: {str(e)}")
            continue

    if not results:
        return {"status": "error", "message": "The research bots were blocked by the search engine. Please try again in a minute."}

    combined_research = f"--- SCIENTIFIC RESEARCH DOSSIER: {topic} ---\n\n"
    for article in results:
        title = article.get('title', 'Fact')
        snippet = article.get('body', '')
        url = article.get('href', '')
        combined_research += f"SOURCE: {title}\nURL: {url}\nFACT: {snippet}\n\n"

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

# --- CHAT TUTOR ---
@app.post("/chat-with-document")
def chat_with_document(query: DocumentQuery):
    try:
        doc = db.collection('documents').document(query.document_id).get().to_dict()
        if not doc: return {"status": "error", "message": "Context not found."}
        
        context = "".join([p['text'] for p in doc.get("pages", [])])
        prompt = f"Context: {context}\n\nQuestion: {query.question}\n\nAnswer precisely. If unsure, say so."
        
        response = model.generate_content(prompt)
        return {"status": "success", "answer": response.text}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- PODCAST (MULTI-VOICE FIX) ---
@app.post("/generate-podcast-audio")
def generate_podcast(query: PodcastQuery):
    try:
        doc = db.collection('documents').document(query.document_id).get().to_dict()
        txt = "".join([p['text'] for p in doc.get("pages", [])])
        
        # Stricter prompt for names
        prompt = f"Write a 4-line script about this: {txt}. Use speakers 'Alex' and 'Sam'. Return ONLY a JSON array."
        res = model.generate_content(prompt)
        
        # Clean the JSON output
        clean_json = res.text.replace("```json", "").replace("```", "").strip()
        script = json.loads(clean_json)
        
        audio_segments = []
        for line in script:
            speaker_name = line.get('speaker', '').lower()
            voice = "en-US-Journey-D" if "alex" in speaker_name else "en-US-Journey-F"
            
            synth = tts_client.synthesize_speech(
                input=texttospeech.SynthesisInput(text=line['text']), 
                voice=texttospeech.VoiceSelectionParams(language_code="en-US", name=voice), 
                audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
            )
            audio_segments.append({
                "speaker": line.get('speaker'), 
                "text": line.get('text'), 
                "audio_base64": base64.b64encode(synth.audio_content).decode('utf-8')
            })
        return {"status": "success", "podcast": audio_segments}
    except Exception as e: return {"status": "error", "message": str(e)}

# --- RAW TEXT UPLOAD ---
@app.post("/upload-text")
def upload_text(query: TextUploadQuery):
    doc_id = str(uuid.uuid4())
    db.collection('documents').document(doc_id).set({
        "filename": query.title, 
        "pages": [{"page_number": 1, "text": query.text_content}], 
        "uploaded_at": firestore.SERVER_TIMESTAMP
    })
    return {"status": "success", "document_id": doc_id}
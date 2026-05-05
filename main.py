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

# --- NEW: Google TTS Libraries ---
from google.oauth2 import service_account
from google.cloud import texttospeech

# --- NEW: Auto-Researcher Libraries ---
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

# Init Google TTS Client using the exact same keys!
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
        
        return {"status": "success", "document_id": doc_id, "filename": file.filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/ingest-youtube")
def ingest_youtube_video(query: YouTubeQuery):
    try:
        video_id = None
        if "v=" in query.url:
            video_id = query.url.split("v=")[1][:11]
        elif "youtu.be/" in query.url:
            video_id = query.url.split("youtu.be/")[1][:11]
            
        if not video_id:
            return {"status": "error", "message": "Could not find valid YouTube ID."}

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
        doc_ref.set({
            "filename": filename,
            "pages": extracted_pages,
            "uploaded_at": firestore.SERVER_TIMESTAMP
        })
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
        doc_ref.set({
            "filename": query.title,
            "pages": extracted_pages,
            "uploaded_at": firestore.SERVER_TIMESTAMP
        })
        return {"status": "success", "document_id": doc_id, "filename": query.title}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- NEW: AUTO-RESEARCHER PIPELINE ---
@app.post("/auto-research")
def auto_research(query: ResearchQuery):
    try:
        topic = query.topic
        print(f"Researching topic: {topic}")
        combined_research = f"--- AUTOMATED RESEARCH ON: {topic} ---\n\n"
        
        # 1. Search the web
        results = DDGS().text(topic, max_results=3)
        
        # 2. Scrape the text
        for article in results:
            url = article.get('href')
            combined_research += f"Source: {url}\n"
            try:
                # We add a "User-Agent" header to look like a real browser (stops some blocks)
                headers = {'User-Agent': 'Mozilla/5.0'}
                page = requests.get(url, timeout=5, headers=headers)
                soup = BeautifulSoup(page.content, 'html.parser')
                
                paragraphs = soup.find_all('p')
                article_text = " ".join([p.get_text() for p in paragraphs])
                
                if len(article_text) > 100:
                    combined_research += article_text[:3000] + "\n\n"
                else:
                    combined_research += "[Website had no readable content]\n\n"
            except Exception:
                combined_research += "[Could not read this website]\n\n"

        # 3. Save to Firestore
        words = combined_research.split()
        chunk_size = 500
        extracted_pages = []
        page_num = 1
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            extracted_pages.append({"page_number": page_num, "text": chunk})
            page_num += 1
            
        doc_id = str(uuid.uuid4())
        filename = f"Research: {topic}"
        
        doc_ref = db.collection('documents').document(doc_id)
        doc_ref.set({
            "filename": filename,
            "pages": extracted_pages,
            "uploaded_at": firestore.SERVER_TIMESTAMP
        })
        
        # THE FIX: We now explicitly send the preview back to the frontend!
        return {
            "status": "success", 
            "message": "Research complete and saved to memory!",
            "document_id": doc_id,
            "filename": filename,
            "scraped_text_preview": combined_research[0:500].replace('\n', ' ') # Send the first 500 chars
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

        # 3. Save it to your Firestore database (using text-chunking logic)
        words = combined_research.split()
        chunk_size = 500
        extracted_pages = []
        page_num = 1
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            extracted_pages.append({"page_number": page_num, "text": chunk})
            page_num += 1
            
        doc_id = str(uuid.uuid4())
        filename = f"Research: {topic}"
        
        doc_ref = db.collection('documents').document(doc_id)
        doc_ref.set({
            "filename": filename,
            "pages": extracted_pages,
            "uploaded_at": firestore.SERVER_TIMESTAMP
        })
        
        return {
            "status": "success", 
            "message": "Research complete and saved to memory!",
            "document_id": doc_id,
            "filename": filename
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/chat-with-document")
def chat_with_document(query: DocumentQuery):
    try:
        doc_ref = db.collection('documents').document(query.document_id)
        doc = doc_ref.get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Document not found.")
            
        doc_data = doc.to_dict()
        context_string = "".join([f"\n--- [Page {p['page_number']}] ---\n{p['text']}\n" for p in doc_data.get("pages", [])])

        prompt = f"""You are an academic tutor helping a student study '{doc_data.get('filename')}'.
        1. Answer USING ONLY the Source Context.
        2. If not found, say "I cannot find this information."
        3. ALWAYS cite the exact page number like this: [Page X]. 
        Source Context: {context_string}
        User Question: {query.question}"""
        
        response = model.generate_content(prompt)
        return {"status": "success", "answer": response.text}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- THE COMPLETE AUDIO PIPELINE ---
@app.post("/generate-podcast-audio")
def generate_podcast_audio(query: PodcastQuery):
    try:
        # 1. Get document
        doc_ref = db.collection('documents').document(query.document_id)
        doc = doc_ref.get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Document not found.")
            
        doc_data = doc.to_dict()
        context_string = "".join([f"{p['text']}\n" for p in doc_data.get("pages", [])])

        # 2. Force Gemini to output pure JSON
        prompt = f"""Write a short, engaging 4-line podcast script about this text: '{doc_data.get('filename')}'.
        Host 1 is Alex (expert). Host 2 is Sam (curious).
        Return ONLY a JSON array of objects with 'speaker' and 'text' keys. Do not use markdown blocks.
        Example: [{{"speaker": "Alex", "text": "Hello!"}}, {{"speaker": "Sam", "text": "Hi!"}}]
        
        Source Text: {context_string}"""
        
        response = model.generate_content(prompt)
        
        # Strip markdown if Gemini accidentally adds it
        clean_json_str = response.text.replace("```json", "").replace("```", "").strip()
        script_data = json.loads(clean_json_str)
        
        # 3. Generate Audio for each line
        audio_segments = []
        
        for line in script_data:
            speaker = line.get("speaker")
            text = line.get("text")
            
            # Select voice based on speaker
            voice_name = "en-US-Journey-D" if speaker == "Alex" else "en-US-Journey-F"
            
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name=voice_name
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            
            # Call Google Cloud
            tts_response = tts_client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            
            # Convert raw audio bytes to base64 so we can send it safely over the internet
            audio_base64 = base64.b64encode(tts_response.audio_content).decode('utf-8')
            
            audio_segments.append({
                "speaker": speaker,
                "text": text,
                "audio_base64": audio_base64
            })

        return {
            "status": "success", 
            "filename": doc_data.get('filename'),
            "podcast": audio_segments
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
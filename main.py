import os
from fastapi import FastAPI
import google.generativeai as genai
from pydantic import BaseModel

# 1. Setup the app and the AI
app = FastAPI()

# This automatically grabs the secret key we put in Render
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY") 
genai.configure(api_key=GOOGLE_API_KEY)

# We will use the Gemini 1.5 Flash model because it is incredibly fast and cheap
model = genai.GenerativeModel('gemini-1.5-flash')

# 2. Define the data we expect to receive from your future Windows app
class UserQuery(BaseModel):
    question: str

# 3. The original health check
@app.get("/")
def read_root():
    return {"status": "success", "message": "The Education AI Server is running!"}

# 4. The New AI Brain Endpoint
@app.post("/ask")
def ask_gemini(query: UserQuery):
    try:
        # Send the user's question to Gemini
        response = model.generate_content(query.question)
        
        # Return the AI's answer
        return {"status": "success", "answer": response.text}
    except Exception as e:
        return {"status": "error", "message": str(e)}
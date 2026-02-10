from fastapi import FastAPI, UploadFile, File, Form
from dotenv import load_dotenv
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from chat_core import generate_reply, USER_DATA, generate_performance_report
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/chat")
async def chat(user_id: str = Form(...), prompt: str = Form(...), files: List[UploadFile] = File(None)):
    uploaded = []
    if files:
        for f in files:
            uploaded.append(f)
    reply = generate_reply(user_id, prompt, uploaded)
    return {"reply": reply}

@app.get("/profile/{user_id}")
def profile(user_id: str):
    user_data = USER_DATA.get(user_id.lower())
    if not user_data:
        return {"error": "User not found"}
    return {"report": generate_performance_report(user_data)}

@app.get("/")
def health():
    return {"message": "MedMentor AI API is live"}
import time

@app.on_event("startup")
def startup_event():
    time.sleep(3)  # Helps during Render cold starts

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=10000, reload=True)


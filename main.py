from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import logging
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from chat_core import generate_reply, USER_DATA, generate_performance_report
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("medmentor-backend")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/chat")
async def chat(request: Request):
    """Accepts FormData (user_id, prompt, files...) or JSON body.
    Always returns JSON and never raises on OpenAI failures.
    """
    user_id = None
    prompt = None
    uploaded = []
    try:
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
            form = await request.form()
            user_id = form.get("user_id") or form.get("userid") or form.get("userId")
            prompt = form.get("prompt") or form.get("message") or form.get("input")
            # Collect uploaded files present in form values
            try:
                files = [v for v in form.values() if hasattr(v, 'filename')]
                uploaded = files
            except Exception:
                uploaded = []
        else:
            body = await request.json()
            user_id = body.get("user_id") or body.get("userid") or body.get("userId")
            prompt = body.get("prompt") or body.get("message") or body.get("input")
    except Exception as e:
        logger.warning("Failed to parse request body: %s", str(e))

    user_id = (user_id or "anonymous").lower()
    prompt = (prompt or "").strip()

    logger.info("Incoming chat request. user_id=%s prompt_len=%d", user_id, len(prompt))

    if not prompt:
        return JSONResponse(status_code=400, content={"success": False, "user_id": user_id, "prompt": prompt, "response": "", "error": "prompt is required"})

    reply = generate_reply(user_id, prompt, uploaded)

    return JSONResponse(status_code=200, content={"success": True, "user_id": user_id, "prompt": prompt, "response": reply})

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
    # Small delay helps during cold starts on some hosts
    time.sleep(1)
    # Startup checks
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.error("OPENAI_API_KEY missing: OpenAI calls will return fallback responses.")
    else:
        logger.info("OPENAI_API_KEY found; backend ready to make OpenAI requests.")

if __name__ == "__main__":
    # Run directly with: python main.py (development)
    # Preferred production/dev invocation: `uvicorn main:app --reload`
    uvicorn.run(app, host="0.0.0.0", port=10000, reload=True)


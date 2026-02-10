import os
import re
from openai import OpenAI
from dotenv import load_dotenv
import logging
import tempfile
from PyPDF2 import PdfReader
from docx import Document
import csv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MedMentor AI")

# Load environment
load_dotenv()

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Enhanced user data storage
USER_DATA = {
    "Siva": {
        "name": "Siva",
        "college": "AIIMS Delhi",
        "specialization": "Cardiothoracic Surgery",
        "simulations_completed": 7,
        "total_simulations": 10,
        "performance": {
            "Heart Bypass": 4.8,
            "Valve Replacement": 4.2,
            "Aortic Repair": 3.9
        },
        "weak_areas": ["artery clamping", "anastomosis techniques"],
        "recent_simulations": ["CABG", "Valve Repair", "Aortic Reconstruction"]
    },
    "Likhith": {
        "name": "Likhith",
        "college": "CMC Vellore",
        "specialization": "Neurosurgery",
        "simulations_completed": 5,
        "total_simulations": 8,
        "performance": {
            "Craniotomy": 4.6,
            "Spinal Fusion": 4.3,
            "DBS Implantation": 4.1
        },
        "weak_areas": ["suture stitching", "hemostasis control"],
        "recent_simulations": ["Craniotomy", "Spinal Fusion", "Tumor Resection"]
    }
}

# Process uploaded files
def process_uploaded_files(uploaded_files):
    medical_knowledge = ""
    
    for file in uploaded_files:
        try:
            file_name = getattr(file, "filename", "upload")
            # FastAPI UploadFile always has .file attribute
            file.file.seek(0)
            raw_bytes = file.file.read()

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_name) as tmp:
                tmp.write(raw_bytes)
                tmp_path = tmp.name
            
            if file_name.endswith('.pdf'):
                reader = PdfReader(tmp_path)
                text = " ".join([page.extract_text() for page in reader.pages])
            elif file_name.endswith('.docx'):
                doc = Document(tmp_path)
                text = "\n".join([para.text for para in doc.paragraphs])
            elif file_name.endswith('.csv'):
                with open(tmp_path, 'r') as f:
                    reader = csv.reader(f)
                    text = "\n".join([",".join(row) for row in reader])
            else:
                with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            
            medical_knowledge += f"\n\n[From {file_name}]:\n{text[:5000]}"
            os.unlink(tmp_path)
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
    
    return medical_knowledge

# Enhanced personalization
def get_personalized_context(user_data, prompt):
    # Check for specific procedure mentions
    for procedure in user_data.get("performance", {}):
        if procedure.lower() in prompt.lower():
            return f" (Note: Your last {procedure} score was {user_data['performance'][procedure]}⭐)"
    
    # Check for weak area mentions
    for area in user_data.get("weak_areas", []):
        if area.lower() in prompt.lower():
            return f" (This addresses your weak area in {area})"
    
    # Check for recent simulations
    for sim in user_data.get("recent_simulations", []):
        if sim.lower() in prompt.lower():
            return f" (You recently practiced this in {sim} simulation)"
    
    return ""

# Enhanced performance report
def generate_performance_report(user_data):
    # Create performance summary
    perf_summary = "\n".join(
        [f"- {proc}: {score}⭐" for proc, score in user_data.get("performance", {}).items()]
    )
    
    return (
        f"📊 **{user_data['name']}'s Surgical Training Progress**\n"
        f"🏥 {user_data['college']} | {user_data['specialization']}\n\n"
        f"**Performance Summary**\n{perf_summary}\n\n"
        f"**Recent Simulations**: {', '.join(user_data.get('recent_simulations', []))}\n"
        f"**Areas Needing Focus**: {', '.join(user_data.get('weak_areas', []))}\n\n"
        f"Need personalized training suggestions?"
    )

# Enhanced medical response with RAG
def generate_medical_response(message, user_data, medical_context):
    # Get personalized context
    personal_context = get_personalized_context(user_data, message)
    
    system_prompt = f"""You are a Senior Professor of {user_data['specialization']} at {user_data['college']}.
You are mentoring {user_data['name']} who has completed {user_data.get('simulations_completed', 0)} simulations.

Weak Areas: {', '.join(user_data.get('weak_areas', []))}
Recent Simulations: {', '.join(user_data.get('recent_simulations', []))}

{f"Additional Medical Context: {medical_context}" if medical_context else ""}

Guidelines:
1. Use precise medical terminology
2. Reference surgical principles
3. Suggest practical exercises
4. Highlight pitfalls
5. Relate to real cases
6. Address knowledge gaps
7. Be concise (3-5 key points)
"""
    
    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API Error: {str(e)}")
        return f"⚠️ Error: {str(e)}. Check API key and model availability."

def generate_reply(user_id, prompt, uploaded_files=[]):
    try:
        user_id = user_id.lower()
        user_data = USER_DATA.get(user_id, {
            "name": user_id.title(),
            "college": "Medical College",
            "specialization": "General Surgery",
            "simulations_completed": 0,
            "performance": {},
            "weak_areas": [],
            "recent_simulations": []
        })
        
        # Process uploaded files for RAG
        medical_context = process_uploaded_files(uploaded_files) if uploaded_files else ""
        
        # Handle greetings
        if any(greeting in prompt.lower() for greeting in ["hi", "hello", "hey"]):
            return (f"👋 Dr. {user_data['name']}! Ready for {user_data['specialization']} training? "
                    f"Your recent focus: {user_data.get('recent_simulations', ['No recent simulations'])[-1]}")
        
        # Handle performance queries
        if any(word in prompt.lower() for word in ["progress", "stats", "performance", "report"]):
            return generate_performance_report(user_data)
        
        # Handle simulation requests
        if any(word in prompt.lower() for word in ["simulation", "practice", "train", "exercise"]):
            sims = user_data.get('recent_simulations', [])
            return f"🚀 Launching {user_data['specialization']} simulation...\n\n" + \
                   (("\n".join([f"- {sim}" for sim in sims]) + "\n\nWhich procedure shall we practice?") if sims else "Ready to start your training!")
        
        # Default to medical knowledge with RAG
        return generate_medical_response(prompt, user_data, medical_context)
    
    except Exception as e:
        logger.error(f"Error generating reply: {str(e)}")
        return f"⚠️ Error: {str(e)}"

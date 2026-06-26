import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "Arko007/variantscope-models")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_SPACE_TOKEN = os.getenv("HF_SPACE_TOKEN", "")
MODEL_CACHE_DIR = os.getenv("MODEL_DIR", "/model")

FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "plant-cloud-cd461")
FIREBASE_PRIVATE_KEY = os.getenv("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n")
FIREBASE_PRIVATE_KEY_ID = os.getenv("FIREBASE_PRIVATE_KEY_ID", "")
FIREBASE_CLIENT_EMAIL = os.getenv("FIREBASE_CLIENT_EMAIL", "")
FIREBASE_CLIENT_ID = os.getenv("FIREBASE_CLIENT_ID", "")

FIREBASE_API_KEY = "AIzaSyDU4EEHT3HEvKNPOrpglLdF3y5Tfs6qy4E"
FIREBASE_AUTH_DOMAIN = "plant-cloud-cd461.firebaseapp.com"
FIREBASE_PROJECT_ID_WEB = "plant-cloud-cd461"

_cors_raw = os.getenv("CORS_ORIGINS", "http://localhost:3000,https://*.vercel.app")
CORS_ORIGINS = [o for o in _cors_raw.split(",") if "*" not in o]
CORS_ORIGIN_REGEX = "|".join(
    o.strip().replace(".", "\\.").replace("*", ".*") + "$"
    for o in _cors_raw.split(",")
    if "*" in o
) or None

ESM_MODEL_NAME = os.getenv("ESM_MODEL_NAME", "facebook/esm2_t30_150M_UR50D")
DNABERT_MODEL_NAME = os.getenv("DNABERT_MODEL_NAME", "zhihan1996/DNABERT-2-117M")

MAX_SEQUENCE_LENGTH = 1024

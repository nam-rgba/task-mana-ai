
import os

# === Gemini API ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # None if not set

# Model mặc định (Gemini 1.5 Pro hoặc 1.5 Flash)
GEMINI_MODEL_SUMMARY = os.getenv("GEMINI_MODEL_SUMMARY", "")
GEMINI_MODEL_CLASSIFY = os.getenv("GEMINI_MODEL_CLASSIFY", "")
GEMINI_MODEL_GENERIC  = os.getenv("GEMINI_MODEL_GENERIC", "")

# Limits
MAX_SUMMARY_LEN = int(os.getenv("MAX_SUMMARY_LEN", "1024"))
TIMEOUT_SECS = int(os.getenv("AI_TIMEOUT_SECS", "20"))
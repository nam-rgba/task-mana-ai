from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import joblib
from pathlib import Path
from dotenv import load_dotenv
import os


#Load environment variables
load_dotenv()


class ModelsLoader:
    _llm = None
    _emb = None
    _xgb = None
    _scaler = None

    @staticmethod
    def llm():
        if ModelsLoader._llm is not None:
            return ModelsLoader._llm

        # --- Thử kết nối với Groq ---
        try:
            model_name = os.getenv("GROQ_MODEL_NAME", "deepseek-r1-distill-llama-70b")
            ModelsLoader._llm = ChatGroq(
                model_name=model_name,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0,
                max_tokens=None,
                reasoning_format="parsed",
                timeout=None,
                max_retries=2,
            )
            print("[ModelsLoader] Đã load model LLM Groq: ", model_name)
            return ModelsLoader._llm
        except Exception as e:
            print(f"[ModelsLoader] Groq LLM failed: {e}")

        # --- Fallback: Gemini ---
        try:
            gemini_api_key = os.getenv("GEMINI_API_KEY", "")
            gemini_model_name = (
                os.getenv("GEMINI_MODEL_GENERIC")
                or os.getenv("GEMINI_MODEL_SUMMARY")
                or "gemini-2.5-flash"
            )
            if not gemini_api_key:
                print("[ModelsLoader] Gemini disabled — missing GEMINI_API_KEY")
                ModelsLoader._llm = None
                return None
            ModelsLoader._llm = ChatGoogleGenerativeAI(
                model=gemini_model_name,
                google_api_key=gemini_api_key,
                temperature=0,
                max_output_tokens=2048,
            )
            print(f"[ModelsLoader] Đã load model LLM Gemini: {gemini_model_name}")
            return ModelsLoader._llm
        except Exception as e:
            print(f"[ModelsLoader] Gemini LLM failed: {e}")
            ModelsLoader._llm = None
            return None

    @staticmethod
    def embeddings():
        if ModelsLoader._emb is None:
            print("[ModelsLoader] Đang load model Embeddings...")
            ModelsLoader._emb = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={"device": "cpu"},
            )
        
        return ModelsLoader._emb


    @staticmethod
    def xgb_model():
        if ModelsLoader._xgb is None:
            #Local import tránh import vòng lặp
            from app.services.xgb_service import XGBService
            # Khởi tạo XGBService chỉ để lấy version model mới nhất
            db_conn = os.getenv("DB_CONNECT_STRING")
            xgb_service = XGBService(connection_string=db_conn)
            model_path = xgb_service.get_latest_xgb_model_path()
            if not model_path:
                model_dir = Path(__file__).resolve().parent.parent / "models"
                model_path = str(model_dir / "xgb_storypoint.pkl")
            ModelsLoader._xgb = joblib.load(model_path)
            print("[ModelsLoader] Đã load model XGB: ", model_path)
        return ModelsLoader._xgb

    @staticmethod
    def scaler():
        if ModelsLoader._scaler is None:
            model_dir = Path(__file__).resolve().parent.parent / "models"
            ModelsLoader._scaler = joblib.load(model_dir / "scaler.pkl")
        return ModelsLoader._scaler

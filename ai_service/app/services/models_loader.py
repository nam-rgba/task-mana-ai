from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
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

        model_name = os.getenv("GROQ_MODEL_NAME", "deepseek-r1-distill-llama-70b") # Currently use openai/gpt-oss-120b
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
            model_dir = Path(__file__).resolve().parent.parent / "models"
            ModelsLoader._xgb = joblib.load(model_dir / "xgb_storypoint.pkl")
        print("[ModelsLoader] Đã load model XGB")
        return ModelsLoader._xgb

    @staticmethod
    def scaler():
        if ModelsLoader._scaler is None:
            model_dir = Path(__file__).resolve().parent.parent / "models"
            ModelsLoader._scaler = joblib.load(model_dir / "scaler.pkl")
        return ModelsLoader._scaler

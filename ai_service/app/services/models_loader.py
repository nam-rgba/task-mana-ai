import os
import joblib
import psycopg2
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()


class ModelsLoader:
    _llm = None
    _emb = None
    _emb_sentence = None  # Dùng riêng cho XGB (dùng encode: normalization = True)
    _xgb = None
    _xgb_path = None  # Lưu thông tin model hiện tại
    _scaler = None
    _le_type = None
    _le_priority = None

    # Base path
    MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

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
    def xgb_sentence_embeddings():
        if ModelsLoader._emb_sentence is None:
            print("[ModelsLoader] Đang load model Sentence Embeddings cho XGB...")
            ModelsLoader._emb_sentence = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                device="cpu",
            )
        return ModelsLoader._emb_sentence

    @staticmethod
    def _get_latest_model_path_from_db():
        """Truy vấn DB lấy path model mới nhất mà không phụ thuộc XGBService."""
        db_conn = os.getenv("DB_CONNECT_STRING")
        if not db_conn:
            return None
        try:
            with psycopg2.connect(db_conn) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT model_path FROM xgb_model_versions ORDER BY updated_at DESC, id DESC LIMIT 1"
                    )
                    row = cur.fetchone()
                    return row[0] if row else None
        except Exception as e:
            print(f"[ModelsLoader] DB Query Error: {e}")
            return None

    @staticmethod
    def xgb_model(force_reload=False):
        latest_path = ModelsLoader._get_latest_model_path_from_db() or str(
            ModelsLoader.MODEL_DIR / "xgb_sp.pkl"
        )

        if (
            force_reload
            or ModelsLoader._xgb is None
            or ModelsLoader._xgb_path != latest_path
        ):
            if Path(latest_path).exists():
                ModelsLoader._xgb = joblib.load(latest_path)
                ModelsLoader._xgb_path = latest_path
                print(f"[ModelsLoader] Đã load XGB: {latest_path}")
            else:
                print(
                    f"[ModelsLoader] Warning: Model path {latest_path} không tìm thấy."
                )
        return ModelsLoader._xgb

    @staticmethod
    def get_xgb_model_info():
        """Lấy thông tin model XGB hiện tại."""
        model = ModelsLoader.xgb_model()
        if model is None:
            return {"status": "No model loaded"}
        info = {
            "model_path": ModelsLoader._xgb_path,
            "n_estimators": model.get_params().get("n_estimators", None),
            "max_depth": model.get_params().get("max_depth", None),
            "learning_rate": model.get_params().get("learning_rate", None),
        }
        return info

    @staticmethod
    def reload_xgb_model(model_path: str):
        """Reload model từ một đường dẫn cụ thể vào cache."""
        if Path(model_path).exists():
            ModelsLoader._xgb = joblib.load(model_path)
            ModelsLoader._xgb_path = model_path
            print(f"[ModelsLoader] Cache reloaded with: {model_path}")

    @staticmethod
    def scaler():
        if ModelsLoader._scaler is None:
            model_dir = Path(__file__).resolve().parent.parent / "models"
            scaler_path = model_dir / "scaler.pkl"
            if scaler_path.exists():
                ModelsLoader._scaler = joblib.load(scaler_path)
            else:
                # Nếu không có file, tạo mới như xgb_service
                ModelsLoader._scaler = MinMaxScaler()
        return ModelsLoader._scaler

    @staticmethod
    def le_type():
        if ModelsLoader._le_type is None:
            model_dir = Path(__file__).resolve().parent.parent / "models"
            le_type_path = model_dir / "le_type.pkl"
            if le_type_path.exists():
                ModelsLoader._le_type = joblib.load(le_type_path)
            else:
                le_type = LabelEncoder()
                le_type.fit(
                    [
                        "FEATURE",
                        "BUG",
                        "IMPROVEMENT",
                        "RESEARCH",
                        "DOCUMENTATION",
                        "TESTING",
                        "DEPLOYMENT",
                        "ENHANCEMENT",
                        "MAINTENANCE",
                        "OTHER",
                    ]
                )
                ModelsLoader._le_type = le_type
        return ModelsLoader._le_type

    @staticmethod
    def le_priority():
        if ModelsLoader._le_priority is None:
            model_dir = Path(__file__).resolve().parent.parent / "models"
            le_priority_path = model_dir / "le_priority.pkl"
            if le_priority_path.exists():
                ModelsLoader._le_priority = joblib.load(le_priority_path)
            else:
                le_priority = LabelEncoder()
                le_priority.fit(["HIGH", "MEDIUM", "LOW", "URGENT"])
                ModelsLoader._le_priority = le_priority
        return ModelsLoader._le_priority

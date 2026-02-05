import os
import joblib
import psycopg2
import pandas as pd
import numpy as np
from psycopg2.extras import RealDictCursor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from typing import Optional
from app.services.models_loader import ModelsLoader
import datetime
import json

# Lớp này hỗ trợ quản lý model XGBoost với versioning và retrain model
class XGBService:
    def __init__(self, connection_string: Optional[str], embed_model=None, model_dir=None):
        self.connection_string = connection_string or os.getenv("DB_CONNECT_STRING")
        self.model_dir = model_dir or "app/models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.embed_model = embed_model
        self._ensure_version_table()
        
    def _ensure_version_table(self):
        """Tự động tạo bảng version nếu chưa có."""
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS xgb_model_versions (
                        id SERIAL PRIMARY KEY,
                        model_path VARCHAR(255) NOT NULL,
                        metrics_json JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()

    def get_latest_xgb_model_path(self) -> str:
        """Lấy đường dẫn model XGBoost mới nhất."""
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT model_path FROM xgb_model_versions
                    ORDER BY updated_at DESC, id DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
                return row["model_path"] if row else None

    # HÀM DỰ ĐOÁN STORY POINT
    def predict_story_point(self, title: str, desc: str, type_val: str, priority_val: str) -> float:
        # Lấy tất cả dependency 1 lần duy nhất từ ModelsLoader
        xgb_model = ModelsLoader.xgb_model()
        scaler = ModelsLoader.scaler()
        le_type = ModelsLoader.le_type()
        le_priority = ModelsLoader.le_priority()
        
        if not all([xgb_model, self.embed_model, scaler, le_type, le_priority]):
            raise RuntimeError("Resources chưa được load đầy đủ!")

        # Xử lý input text
        combined_text = f"{str(title or '')} {str(desc or '')}"
        combined_emb = self.embed_model.encode(combined_text, normalize_embeddings=True)
        
        # Xử lý features bổ trợ
        lengths = pd.DataFrame([[len(str(title or '')), len(str(desc or ''))]], 
                               columns=["title_length", "desc_length"])
        transformed_lengths = scaler.transform(lengths)
        
        try:
            type_encoded = le_type.transform([type_val])[0]
            priority_encoded = le_priority.transform([priority_val])[0]
        except ValueError as e:
            print(f"[XGBService] Label mới chưa được học: {e}. Dùng default 0.")
            type_encoded, priority_encoded = 0, 0

        extra_features = [type_encoded, priority_encoded, transformed_lengths[0][0], transformed_lengths[0][1]]
        X_input = np.concatenate([combined_emb, extra_features]).reshape(1, -1)

        pred = xgb_model.predict(X_input)[0]
        return round(float(pred), 2)

    # HÀM GỢI Ý STORY POINT THEO PLANNING POKER
    def suggest_story_point(self, value: float) -> str:
        """
        Chuyển giá trị thô sang Story Point gần nhất theo chuẩn Planning Poker."""
        STORY_POINTS = [0.5, 1, 2, 3, 5, 8, 13]
        
        diffs = [(abs(value - sp), sp) for sp in STORY_POINTS]
        diffs.sort(key=lambda x: x[0])

        best = diffs[0][1]
        second = diffs[1][1]

        # Nếu giá trị nằm giữa 2 story point gần nhau
        if abs(value - best) < 0.4 and abs(value - second) < 0.4:
            return f"{best} - {second}"
        return str(best)


    def add_xgb_model_version(self, model_path: str, metrics: dict = None):
        """Thêm version mới cho model XGBoost, lưu metrics dạng JSON."""
        metrics_json = json.dumps(metrics) if metrics is not None else None
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO xgb_model_versions (model_path, metrics_json, updated_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    """,
                    (model_path, metrics_json)
                )
                conn.commit()

    def load_data(self, csv_path="app/data/train/tasks_done_train.csv"):
        print(f"[XGBService] Đọc dữ liệu từ: {csv_path}")
        df = pd.read_csv(csv_path)
        # Kiểm tra các cột cần thiết
        required_columns = {"Title", "Description", "Story_Point", "Type", "Priority"}
        missing = required_columns - set(df.columns)
        if missing:
            raise Exception(f"File thiếu các cột bắt buộc: {', '.join(missing)}")
        before = len(df)
        df = df.drop_duplicates().dropna(subset=["Title", "Description", "Story_Point"])
        after = len(df)
        print(f"[XGBService] Số dòng trước khi clean: {before}, sau khi clean: {after}")
        return df

    def preprocess(self, df):
        import time
        print("[XGBService] Đang load scaler và label encoders từ ModelsLoader...")
        # Lấy scaler/encoder mới nhất từ ModelsLoader mỗi lần gọi
        scaler = ModelsLoader.scaler()
        le_type = ModelsLoader.le_type()
        le_priority = ModelsLoader.le_priority()

        if scaler is None or le_type is None or le_priority is None:
            raise Exception("Thiếu scaler, le_type hoặc le_priority. Hãy đảm bảo đã fit và lưu các file này trước khi train.")

        print("[XGBService] Bắt đầu encode categorical features (Type, Priority)...")
        df["type_encoded"] = le_type.transform(df["Type"].astype(str))
        df["priority_encoded"] = le_priority.transform(df["Priority"].astype(str))

        print("[XGBService] Bắt đầu tính toán text features...")
        df["title_length"] = df["Title"].apply(lambda x: len(str(x)))
        df["desc_length"] = df["Description"].apply(lambda x: len(str(x)))
        df[["title_len_norm", "desc_len_norm"]] = scaler.transform(df[["title_length", "desc_length"]])

        print("[XGBService] Bắt đầu embedding (Title + Description)...")
        texts = (df["Title"] + " " + df["Description"]).tolist()
        print(f"[XGBService] Số lượng sample cần embedding: {len(texts)}")
        t0 = time.time()
        embeddings = self.embed_model.encode(texts, normalize_embeddings=True)
        embeddings = np.array(embeddings)
        t1 = time.time()
        print(f"[XGBService] Đã embedding xong. Shape: {embeddings.shape}. Thời gian: {t1-t0:.2f}s")
        
        #Gộp data lại
        X = np.hstack([
            embeddings,
            df[["type_encoded", "priority_encoded", "title_len_norm", "desc_len_norm"]].values
        ])
        y = df["Story_Point"].values.astype(float)

        print(f"[XGBService] X shape: {X.shape}, y shape: {y.shape}")
        return X, y, scaler, le_type, le_priority

    def retrain_and_save(self, csv_path="app/data/train/tasks_done_train.csv"):
        """
        Huấn luyện lại mô hình XGB từ đầu với dữ liệu mới.
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import time
        print(f"[XGBService] === BẮT ĐẦU TRAIN XGBRegressor ===")

        #Lấy dữ liệu
        df = self.load_data(csv_path)

        # Xử lý dữ liệu
        X, y, scaler, le_type, le_priority = self.preprocess(df)

        print(f"[XGBService] Train/test split...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"[XGBService] X_train: {X_train.shape}, X_test: {X_test.shape}")

        # Khởi tạo và train model mới
        model = XGBRegressor(
            n_estimators=5000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        t0 = time.time()
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=500)
        t1 = time.time()
        print(f"[XGBService] Đã train xong XGBRegressor. Thời gian train: {t1-t0:.2f}s")

        # Sinh tên model tự động với timestamp đầy đủ
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model_name = f"xgb_storypoint[new]_{timestamp}.pkl"

        # Đánh giá
        y_pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))
        print(f"[XGBService] MAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
        metrics = {
            "Model": "XGBoostRegressor",
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }

        # Save model, scaler, encoders
        joblib.dump(model, os.path.join(self.model_dir, model_name))
        model_path = os.path.join(self.model_dir, model_name)
        
        # Lưu version mới vào DB (metrics dạng json)
        self.add_xgb_model_version(model_path, metrics=metrics)
        
        # Reload model vào cache sau khi train
        ModelsLoader.reload_xgb_model(model_path=model_path)
        print(f"[XGBService] === ĐÃ TRAIN XONG ===")
        return {
            "model_path": model_path,
            "metrics": metrics
        }

    def incremental_train(
        self,
        csv_path,
        num_boost_round=100,
        eval_metric="rmse"
    ):
        """
        Train thêm dữ liệu vào mô hình XGBoost đã có (load từ ModelsLoader).
        In ra metric trước và sau khi train thêm.
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import time
        #Local import tránh import vòng lặp
        from app.services.models_loader import ModelsLoader

        print(f"[XGBService] === INCREMENTAL TRAIN XGBRegressor ===")
        # Luôn lấy model mới nhất từ ModelsLoader
        booster = ModelsLoader.xgb_model()
        # Sinh tên model tự động với timestamp đầy đủ
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model_name = f"xgb_storypoint[incremental]_{timestamp}.pkl"
        model_path = os.path.join(self.model_dir, model_name)

        # Load và preprocess dữ liệu mới
        df = self.load_data(csv_path)
        X, y, scaler, le_type, le_priority = self.preprocess(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Đánh giá trước khi train thêm
        y_pred_before = booster.predict(X_test)
        mae_before = float(mean_absolute_error(y_test, y_pred_before))
        mse_before = float(mean_squared_error(y_test, y_pred_before))
        rmse_before = float(np.sqrt(mse_before))
        r2_before = float(r2_score(y_test, y_pred_before))
        print(f"[XGBService] Metric trước khi train thêm: MAE={mae_before:.4f} | RMSE={rmse_before:.4f} | R2={r2_before:.4f}")

        # Train thêm
        t0 = time.time()
        booster.fit(
            X_train, y_train,
            xgb_model=booster,
            eval_set=[(X_test, y_test)],
            verbose=500
        )
        t1 = time.time()
        print(f"[XGBService] Đã train thêm. Thời gian: {t1-t0:.2f}s")

        # Đánh giá sau khi train thêm
        y_pred_after = booster.predict(X_test)
        mae_after = float(mean_absolute_error(y_test, y_pred_after))
        mse_after = float(mean_squared_error(y_test, y_pred_after))
        rmse_after = float(np.sqrt(mse_after))
        r2_after = float(r2_score(y_test, y_pred_after))
        print(f"[XGBService] Metric SAU khi train thêm: MAE={mae_after:.4f} | RMSE={rmse_after:.4f} | R2={r2_after:.4f}")

        # Lưu lại model đã train thêm
        joblib.dump(booster, model_path)
        # Lưu version mới vào DB (metrics dạng json, chỉ lưu metrics sau khi train)
        metrics_after = {
            "Model": "XGBoostRegressor",
            "MAE": mae_after,
            "MSE": mse_after,
            "RMSE": rmse_after,
            "R2": r2_after
        }
        self.add_xgb_model_version(model_path, metrics=metrics_after)
        ModelsLoader.reload_xgb_model(model_path=model_path)
        print(f"[XGBService] Đã lưu lại model sau khi train thêm: {model_path}")

        return {
            "model_path": model_path,
            "metrics_before": {
                "MAE": mae_before,
                "RMSE": rmse_before,
                "R2": r2_before
            },
            "metrics_after": metrics_after
        }

    def get_model_info(self):
        """Lấy thông tin model XGB hiện tại từ ModelsLoader."""
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

# Tạo singleton XGBService để tái sử dụng cho các lần gọi sau
_xgb_service_instance = None

def get_xgb_service(embed_model=None, model_dir=None):
    global _xgb_service_instance
    if _xgb_service_instance is None:
        # lấy từ ModelsLoader
        if embed_model is None:
            from app.services.models_loader import ModelsLoader
            embed_model = ModelsLoader.xgb_sentence_embeddings() 
        
        db_conn = os.getenv("DB_CONNECT_STRING")
        _xgb_service_instance = XGBService(connection_string=db_conn, embed_model=embed_model, model_dir=model_dir)
    return _xgb_service_instance
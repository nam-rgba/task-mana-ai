import os
import logging
import tempfile
from app.services.fetch_data import FetchData
from app.services.xgb_service import get_xgb_service
from sentence_transformers import SentenceTransformer
from fastapi import APIRouter, UploadFile, File, HTTPException

# Hàm kiểm tra các cột cần thiết trong file CSV
def validate_required_columns(csv_path):
    import pandas as pd
    required_columns = {"Title", "Description", "Story_Point", "Type", "Priority"}
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể đọc file CSV: {str(e)}")
    missing = required_columns - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"File thiếu các cột bắt buộc: {', '.join(missing)}"
        )
    # Có thể trả về df nếu cần dùng lại
    return df

log = logging.getLogger(__name__)

xgb_router = APIRouter(
    prefix="/xgb",
    tags=["AI / XGB Model Test"]
)


@xgb_router.post("/retrain")
async def retrain_xgb_model():
    """
    Test retrain XGB model. Nếu chưa có file train thì tự động fetch dữ liệu DONE về.
    """
    train_path = "app/data/train/tasks_done_train.csv"
    if not os.path.exists(train_path):
        await FetchData.fetch_all_done_tasks_and_export(export_format="csv")

    xgb_service = get_xgb_service()
    result = xgb_service.retrain_and_save(csv_path=train_path)
    return {"status": "success", "output": result}

@xgb_router.post("/incremental_train")
async def incremental_train_xgb_model():
    """
    Test incremental train XGB model with new data.
    Nếu chưa có file train thì tự động fetch dữ liệu DONE về.
    """
    import os
    from app.services.xgb_service import get_xgb_service
    from app.services.fetch_data import FetchData
    from sentence_transformers import SentenceTransformer

    train_path = "app/data/train/tasks_done_train.csv"
    if not os.path.exists(train_path):
        await FetchData.fetch_all_done_tasks_and_export(export_format="csv")

    xgb_service = get_xgb_service()
    result = xgb_service.incremental_train(csv_path=train_path)
    return {"status": "success", "output": result}


@xgb_router.get("/current_model_info")
async def get_current_xgb_model_info():
    """
    Lấy thông tin model XGBoost hiện tại đang sử dụng.
    """
    from app.services.xgb_service import get_xgb_service
    xgb_service = get_xgb_service()
    info = xgb_service.get_model_info()
    return {"current_xgb_model": info}

@xgb_router.post("/retrain_with_file")
async def train_with_file(file: UploadFile = File(...)):
    """
    Train lại XGB model với file train do người dùng upload (multipart/form-data).
    Lưu file vào temp_uploads bằng tempfile để tránh trùng tên, tự động xóa sau khi train.
    """
    # Thư mục tạm để lưu file upload
    temp_dir = "temp_uploads"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Lấy đuôi file
    file_extension = os.path.splitext(file.filename)[1].lower()
    # Tạo file tạm với tên duy nhất
    temp_file = tempfile.NamedTemporaryFile(
        delete=False, 
        suffix=file_extension, 
        dir=temp_dir
        )
    
    # Đường dẫn file tạm
    temp_path = temp_file.name
    try:
        # Lưu nội dung file upload vào file tạm
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # Kiểm tra các cột cần thiết
        validate_required_columns(temp_path)
        xgb_service = get_xgb_service()
        result = xgb_service.retrain_and_save(csv_path=temp_path)
        return {"status": "success", "output": result}
    finally:
        temp_file.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Train tăng cường với file upload (lưu file vào temp_uploads như extractFileHelper)
@xgb_router.post("/incremental_train_with_file")
async def incremental_train_with_file(file: UploadFile = File(...)):
    """
    Train tăng cường XGB model với file train do người dùng upload (multipart/form-data).
    Lưu file vào temp_uploads bằng tempfile để tránh trùng tên, tự động xóa sau khi train.
    """
    # Thư mục tạm để lưu file upload
    temp_dir = "temp_uploads"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Lấy đuôi file
    file_extension = os.path.splitext(file.filename)[1].lower()
    # Tạo file tạm với tên duy nhất
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension, dir=temp_dir)
    logging.info(f"Created temporary file at {temp_file} for incremental training.")
    temp_path = temp_file.name
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # Kiểm tra các cột cần thiết
        validate_required_columns(temp_path)

        xgb_service = get_xgb_service()
        result = xgb_service.incremental_train(csv_path=temp_path)
        return {"status": "success", "output": result}
    finally:
        temp_file.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)


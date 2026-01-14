import os
import json
import tempfile
from typing import List
from fastapi import UploadFile
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    DirectoryLoader
)

# Hàm trích xuất nội dung văn bản từ file upload
async def extract_text_from_file(file: UploadFile) -> str:
    # 1. Đảm bảo thư mục 'temp' tồn tại
    temp_dir = "temp_uploads"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    file_extension = os.path.splitext(file.filename)[1].lower()
    
    # 2. Tạo đường dẫn file trong thư mục temp
    # Sử dụng tempfile để tạo tên file duy nhất tránh trùng lặp khi nhiều người upload cùng lúc
    temp_file = tempfile.NamedTemporaryFile(
        delete=False, 
        suffix=file_extension, 
        dir=temp_dir  # Chỉ định lưu vào thư mục temp
    )
    temp_file_path = temp_file.name
    
    try:
        # Lưu nội dung file upload
        content = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(content)
        
        # Chọn loader phù hợp dựa trên định dạng file
        if file_extension == '.pdf':
            loader = PyPDFLoader(temp_file_path)
        elif file_extension in ['.doc', '.docx']:
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == '.txt':
            loader = TextLoader(temp_file_path, encoding='utf-8')
        elif file_extension == '.csv':
            loader = CSVLoader(temp_file_path, encoding='utf-8')
        elif file_extension == '.json':
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                return json.dumps(json_data, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Định dạng file không được hỗ trợ: {file_extension}")
        
        # Đọc và trích xuất nội dung văn bản
        documents = loader.load()
        return "\n\n".join(doc.page_content for doc in documents) 
        
    finally:
        # Đóng file nếu còn mở và xóa file tạm
        temp_file.close()
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# Trích xuất nội dung từ nhiều files upload
async def extract_text_from_multiple_files(files: List[UploadFile]) -> str:
    """
    Trích xuất nội dung từ nhiều files
        files: List các UploadFile objects
        str: Nội dung văn bản từ tất cả files, phân cách bởi delimiter
    """
    all_texts = []
    
    for file in files:
        try:
            text = await extract_text_from_file(file)
            all_texts.append(f"=== {file.filename} ===\n{text}")
        except Exception as e:
            all_texts.append(f"=== {file.filename} ===\nLỗi: {str(e)}")
    
    return "\n\n\n\n".join(all_texts)


# Trích xuất nội dung từ tất cả files trong một thư mục
def extract_text_from_directory(directory_path: str) -> str:
    """
    Trích xuất nội dung từ tất cả files trong thư mục
    Args:
        directory_path: Đường dẫn đến thư mục chứa files
    Returns:
        str: Nội dung văn bản từ tất cả files
    """
     
    loader = DirectoryLoader(
        directory_path,
        glob="**/*",
        show_progress=True,
        use_multithreading=True
    )
    
    docs = loader.load()
    context_text = "\n\n".join(doc.page_content for doc in docs)
    
    return context_text
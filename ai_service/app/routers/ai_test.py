import logging
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Request, Query
from app.services.vector_store import VectorStoreService
from app.utils.extractFileHelper import extract_text_from_file, extract_text_from_multiple_files
from pydantic import Field

log = logging.getLogger(__name__)

test_router = APIRouter(
    prefix="/test",
    tags=["AI / Debug & Test"]
)

def get_vector_store_service(request: Request):
    # from app.routers import vector_store_service
    # return vector_store_service
    return request.app.state.vector_store_service

# DEBUG 
# ------------------- TEST ROUTES -------------------

@test_router.get(
    "/retrieve_tasks",
    summary="Test truy vấn task theo query",
    description="""
Truy vấn các task từ vector store dựa trên query và project_id (nếu có).

Args:
- query: Từ khóa tìm kiếm.
- project_id: ID dự án (tùy chọn).

Return:
- Danh sách các task phù hợp.
""",
)
async def test_retrieve_tasks(
    query: str = Query(..., description="Từ khóa tìm kiếm."),
    project_id: int = Query(None, description="ID dự án (tùy chọn)."),
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service),
):
    """Test retrieve tasks from vector store by query + optional project_id"""
    tasks = vector_store_svc.retrieve_tasks_by_query(query=query, project_id=project_id)
    return {"query": query, "project_id": project_id, "results": tasks}

@test_router.get(
    "/retrieve_users",
    summary="Test truy vấn user theo query",
    description="""
Truy vấn các user từ vector store dựa trên query và project_id (nếu có).

Args:
- query: Từ khóa tìm kiếm.
- project_id: ID dự án (tùy chọn).

Return:
- Danh sách các user phù hợp.
""",
)
async def test_retrieve_users(
    query: str = Query(..., description="Từ khóa tìm kiếm."),
    project_id: int = Query(..., description="ID dự án (tùy chọn)."),
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Test retrieve users from vector store by query + optional project_id"""
    users = vector_store_svc.retrieve_users_by_query(task_text=query, project_id=project_id)
    return {"query": query, "project_id": project_id, "results": users}

@test_router.get(
    "/retrieve_with_scores",
    summary="Test truy vấn task kèm điểm số",
    description="""""
Truy vấn các task từ vector store theo query và project_id, trả về kèm điểm số tương đồng.

Args:
- query: Từ khóa tìm kiếm.
- project_id: ID dự án (tùy chọn).

Return:
- Danh sách các task và điểm số tương đồng.
""",
)
async def test_retrieve_with_scores(
    query: str = Query(..., description="Từ khóa tìm kiếm."),
    project_id: int = Query(None, description="ID dự án (tùy chọn)."),
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service),
):
    """Test retrieve tasks with scores (similarity search)"""
    results = vector_store_svc.retrieve_tasks_with_scores(query=query, project_id=project_id)
    return {"query": query, "project_id": project_id, "results": results}

@test_router.get(
    "/retrieve_tasks_by_project",
    summary="Test truy vấn task theo project_id",
    description="""""
Truy vấn các task từ vector store chỉ dựa trên project_id.

Args:
- project_id: ID dự án.

Return:
- Danh sách các task thuộc dự án.
""",
)
async def test_retrieve_task_by_project(
    project_id: int = Query(..., description="ID dự án."),
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Test retrieve tasks from vector store by project_id only"""
    tasks = vector_store_svc.retrieve_tasks_by_project(project_id=project_id)
    return {"project_id": project_id, "results": tasks}
   

@test_router.get(
    "/retrieve_users_by_project",
    summary="Test truy vấn user theo project_id",
    description="""""
Truy vấn các user từ vector store chỉ dựa trên project_id.

Args:
- project_id: ID dự án.

Return:
- Danh sách các user thuộc dự án.
""",
)
async def test_retrieve_users_by_project(
    project_id: int = Query(..., description="ID dự án."),
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Test retrieve users from vector store by project_id only"""
    users = vector_store_svc.retrieve_users_by_project(project_id=project_id)
    return {"project_id": project_id, "results": users}


@test_router.get(
    "/retrieve_tasks_by_user",
    summary="Test truy vấn task theo user_id và project_id",
    description="""""
Truy vấn các task được giao cho user trong một project.

Args:
- user_id: ID người dùng.
- project_id: ID dự án.

Return:
- Danh sách các task của user trong dự án.
""",
)
async def test_retrieve_task_by_user(
    user_id: int = Query(..., description="ID người dùng."),
    project_id: int = Query(..., description="ID dự án."),
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Test retrieve tasks assigned to a specific user_id"""
    tasks = vector_store_svc.retrieve_tasks_by_user(user_id=user_id, project_id=project_id)
    return {"user_id": user_id, "project_id": project_id, "results": tasks}



@test_router.get(
    "/retrieve_guides",
    summary="Test truy vấn guides theo query",
    description="""""
Truy vấn các hướng dẫn (guides) từ vector store dựa trên query.

Args:
- query: Từ khóa tìm kiếm.
- k: Số lượng guides trả về (mặc định 3).

Return:
- Danh sách guides phù hợp và metadata.
""",
)
async def test_retrieve_guides(
    query: str = Query(..., description="Từ khóa tìm kiếm."),
    k: int = Query(3, description="Số lượng guides trả về."),
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Test retrieve guides from vector store by query"""
    try:
        retriever = vector_store_svc.guides_retriever(k=k)
        
        guides = retriever.invoke(query)
        
        return {
            "query": query,
            "k": k,
            "results": [
                {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in guides[:5]  # Limit output size
            ],
            "total": len(guides)
        }
    except Exception as e:
        log.error(f"Retriever error for query '{query}': {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

@test_router.post(
    "/read_file",
    summary="Test đọc nội dung file",
    description="""""
Đọc nội dung từ file upload.

Args:
- file: File upload.

Return:
- Nội dung file.
""",
)
async def test_read_file(
    file: UploadFile = File(..., description="File upload."),
): 
    try:
        text = await extract_text_from_file(file)
        return {"filename": file.filename, "content": text}
    except Exception as e:
        log.error(f"Failed to read file: {e}")
        raise HTTPException(status_code=500, detail=f"File read failed: {str(e)}")

@test_router.post(
    "/read_multiple_files",
    summary="Test đọc nội dung nhiều file",
    description="""""
Đọc nội dung từ nhiều file upload.

Args:
- files: Danh sách file upload.

Return:
- Nội dung tổng hợp từ các file.
""",
)
async def test_read_file(
    files: list[UploadFile] = File(..., description="Danh sách file upload."),
): 
    try:
        text = await extract_text_from_multiple_files(files)
        return {"filenames": [file.filename for file in files], "content": text}
    except Exception as e:
        log.error(f"Failed to read file: {e}")
        raise HTTPException(status_code=500, detail=f"File read failed: {str(e)}")

@test_router.post(
    "/export_done_tasks",
    summary="Test export các task DONE",
    description="""""
Export các task đã hoàn thành (DONE) ra file CSV hoặc JSON.

Args:
- export_format: Định dạng export (csv hoặc json).

Return:
- Đường dẫn file export.
""",
)
async def test_export_done_tasks(
    export_format: str = Query("csv", description="Định dạng export (csv hoặc json).")
):
    """Test exporting DONE tasks to CSV or JSON"""
    from app.services.fetch_data import FetchData

    out_path = await FetchData.fetch_all_done_tasks_and_export(export_format=export_format)
    return {"export_format": export_format, "output_path": out_path}


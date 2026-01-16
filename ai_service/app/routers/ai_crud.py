import logging
from fastapi import APIRouter, Depends, HTTPException, status
from app.services.vector_store import VectorStoreService
from fastapi.encoders import jsonable_encoder
from app.schema.input import *


log = logging.getLogger(__name__)

crud_router = APIRouter(
    prefix="/vector",
    tags=["AI / CRUD (Tasks & Users)"]
)

def get_vector_store_service():
    from app.routers import vector_store_service #import instance
    return vector_store_service


# ------------------- CRUD DOCUMENT ROUTES -------------------
@crud_router.get("/tasks/{task_id}")
async def get_task_by_id(
    task_id: str,
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Lấy thông tin task từ vector store theo task ID."""
    task = vector_store_svc.get_task_by_id(task_id=task_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found.")
    return task

@crud_router.get("/users/{user_id}")
async def get_user_by_id(
    user_id: str,
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Lấy thông tin user từ vector store theo user ID."""
    user = vector_store_svc.get_user_by_id(user_id=user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return user

@crud_router.get("/projects/{project_id}")
async def get_project_by_id(
     project_id: str,
     vector_store_svc: VectorStoreService = Depends(get_vector_store_service)   
):
    """Lấy thông tin project từ vector store theo project ID."""
    project = vector_store_svc.get_project_by_id(project_id=project_id)
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")
    return project

@crud_router.post("/tasks/upsert")
async def upsert_task(
    task_req: UpdateTaskRequest,
    force: bool = True, # Bắt buộc xóa bản cũ để tránh trùng lặp task
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """
    Thêm mới hoặc cập nhật task. BẮT BUỘC phải có task['id']. force=True sẽ xóa bản cũ trước khi insert.
    """
    task_data = jsonable_encoder(task_req)
    success = vector_store_svc.upsert_task(task=task_data, force=force)
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to upsert task.")
    updated_task = vector_store_svc.get_task_by_id(task_id=task_req.id)
    return {
            "detail": "Upsert task thành công.", 
            "task": updated_task
            }
   

@crud_router.post("/users/upsert")
async def upsert_user(
    user_req: UpdateUserRequest,
    force: bool = True, # Bắt buộc xóa bản cũ để tránh trùng lặp user
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """
    Thêm mới hoặc cập nhật user. BẮT BUỘC phải có user['id']. force=True sẽ xóa bản cũ trước khi insert.
    """
    user_data = jsonable_encoder(user_req)
    success = vector_store_svc.upsert_user(user=user_data, force=force)
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to upsert user.")
    updated_user = vector_store_svc.get_user_by_id(user_id=user_req.id)
    return {
            "detail": "Upsert user thành công.", 
            "user": updated_user
            }

@crud_router.post("/projects/upsert")
async def upsert_project(  
    project_req: UpdateProjectRequest,
    force: bool = True, # Bắt buộc xóa bản cũ để tránh trùng lặp project
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """
    Thêm mới hoặc cập nhật project. 

    BẮT BUỘC: phải có project['id']. force=True sẽ xóa bản cũ trước khi insert.
    """
    project_data = jsonable_encoder(project_req)
    success = vector_store_svc.upsert_project(project=project_data, force=force)
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to upsert project.")
    updated_project = vector_store_svc.get_project_by_id(project_id=project_req.id)
    return {
            "detail": "Upsert project thành công.", 
            "project": updated_project
            }

@crud_router.delete("/tasks/{task_id}")
async def delete_task_by_id(
    task_id: str,
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Xoá task khỏi vector store theo task ID."""
    success = vector_store_svc.delete_task_by_id(task_id=task_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found or could not be deleted.")
    task_deleted = vector_store_svc.get_task_by_id(task_id=task_id)
    return {"detail": "Xóa tasks thành công.", "task_deleted": task_deleted}

@crud_router.delete("/users/{user_id}")
async def delete_user_by_id(
    user_id: str,
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Xoá user khỏi vector store theo user ID."""
    success = vector_store_svc.delete_user_by_id(user_id=user_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found or could not be deleted.")
    user_deleted = vector_store_svc.get_user_by_id(user_id=user_id)
    return {"detail": "Xóa user thành công.", "user_deleted": user_deleted}

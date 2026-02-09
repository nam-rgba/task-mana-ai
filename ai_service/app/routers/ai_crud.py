import logging
from fastapi import APIRouter, Depends, HTTPException, status, Request, Path, Query
from app.services.vector_store import VectorStoreService
from fastapi.encoders import jsonable_encoder
from app.schema.input import *
from pydantic import Field

log = logging.getLogger(__name__)

crud_router = APIRouter(
    prefix="/vector",
    tags=["AI / CRUD (Tasks & Users)"]
)

def get_vector_store_service(request: Request):
    # from app.routers import vector_store_service #import instance
    # return vector_store_service
    return request.app.state.vector_store_service


# ------------------- CRUD DOCUMENT ROUTES -------------------

@crud_router.get(
    "/tasks/{task_id}",
    summary="Lấy thông tin task theo ID",
    description="""
Lấy thông tin chi tiết của một task từ vector store dựa trên task ID.

Args:
- task_id: ID của task.

Return:
- Thông tin task (nếu tồn tại).
""",
)
async def get_task_by_id(
    task_id: int = Path(..., description="ID của task cần lấy thông tin."),
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Lấy thông tin task từ vector store theo task ID."""
    task = vector_store_svc.get_task_by_id(task_id=task_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found.")
    return task

@crud_router.get(
    "/users/{user_id}",
    summary="Lấy thông tin user theo ID",
    description="""""
Lấy thông tin chi tiết của một user từ vector store dựa trên user ID.

Args:
- user_id: ID của user.

Return:
- Thông tin user (nếu tồn tại).
""""",
)
async def get_user_by_id(
    user_id: int = Path(..., description="ID của user cần lấy thông tin."),
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Lấy thông tin user từ vector store theo user ID."""
    user = vector_store_svc.get_user_by_id(user_id=user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return user

@crud_router.get(
    "/projects/{project_id}",
    summary="Lấy thông tin project theo ID",
    description="""""
Lấy thông tin chi tiết của một project từ vector store dựa trên project ID.

Args:
- project_id: ID của project.

Return:
- Thông tin project (nếu tồn tại).
""""",
)
async def get_project_by_id(
     project_id: int = Path(..., description="ID của project cần lấy thông tin."),
     vector_store_svc: VectorStoreService = Depends(get_vector_store_service)   
):
    """Lấy thông tin project từ vector store theo project ID."""
    project = vector_store_svc.get_project_by_id(project_id=project_id)
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")
    return project

@crud_router.get(
    "/teams/{team_id}",
    summary="Lấy mapping teamId -> list các userId và role của họ",
)
async def get_team_by_id(
    team_id: int = Path(..., description="ID của team cần lấy thành viên."),
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)):
    """Lấy mapping teamId -> list các userId và role của họ từ vector store."""
    team_map = vector_store_svc.get_team_by_id(team_id=team_id)
    return {"team_members_map": team_map}




@crud_router.post(
    "/tasks/upsert",
    summary="Thêm mới hoặc cập nhật task",
    description="""""
Thêm mới hoặc cập nhật thông tin một task vào vector store.

Args:
- task_req: Thông tin task (bắt buộc có id).
- force: Nếu True, xóa bản cũ trước khi insert.

Return:
- Thông báo thành công và thông tin task đã cập nhật.
""""",
)
async def upsert_task(
    task_req: UpdateTaskRequest,
    force: bool = Query(True, description="Nếu True, xóa bản cũ trước khi insert."),
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

@crud_router.post(
    "/users/upsert",
    summary="Thêm mới hoặc cập nhật user",
    description="""""
Thêm mới hoặc cập nhật thông tin một user vào vector store.

Args:
- user_req: Thông tin user (bắt buộc có id).
- force: Nếu True, xóa bản cũ trước khi insert.

Return:
- Thông báo thành công và thông tin user đã cập nhật.
""""",
)
async def upsert_user(
    user_req: UpdateUserRequest,
    force: bool = Query(True, description="Nếu True, xóa bản cũ trước khi insert."),
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

@crud_router.post(
    "/projects/upsert",
    summary="Thêm mới hoặc cập nhật project",
    description="""""
Thêm mới hoặc cập nhật thông tin một project vào vector store.

Args:
- project_req: Thông tin project (bắt buộc có id).
- force: Nếu True, xóa bản cũ trước khi insert.

Return:
- Thông báo thành công và thông tin project đã cập nhật.
""""",
)
async def upsert_project(  
    project_req: UpdateProjectRequest,
    force: bool = Query(True, description="Nếu True, xóa bản cũ trước khi insert."),
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

@crud_router.post(
        "/teams/upsert",
        summary="Thêm mới hoặc cập nhật team members map",
        description="""""
Thêm mới hoặc cập nhật mapping teamId -> list các userId và role của họ vào vector store.
Args:
- team_id: ID của team (bắt buộc).
- members: List các thành viên trong team với userId và role (bắt buộc).
- force: Nếu True, xóa bản cũ trước khi insert.
Return:
- Thông báo thành công và thông tin team members map đã cập nhật.
""""",
)
async def upsert_team(
    team_req: UpdateTeamRequest,
    force: bool = Query(True, description="Nếu True, xóa bản cũ trước khi insert."),
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """
    Thêm mới hoặc cập nhật mapping teamId -> list các userId và role của họ.
    BẮT BUỘC: phải có team['id']. force=True sẽ xóa bản cũ trước khi insert.
    """
    team_data = jsonable_encoder(team_req)
    success = vector_store_svc.upsert_team(team=team_data, force=force)
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to upsert team.")
    updated_team = vector_store_svc.get_team_by_id(team_id=team_req.id)
    return {
            "detail": "Upsert team thành công.", 
            "team": updated_team
            }


@crud_router.delete(
    "/tasks/{task_id}",
    summary="Xóa task theo ID",
    description="""""
Xóa một task khỏi vector store dựa trên task ID.

Args:
- task_id: ID của task.

Return:
- Thông báo thành công và thông tin task đã bị xóa (nếu có).
""""",
)
async def delete_task_by_id(
    task_id: int = Path(..., description="ID của task cần xóa."),
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Xoá task khỏi vector store theo task ID."""
    success = vector_store_svc.delete_task_by_id(task_id=task_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found or could not be deleted.")
    task_deleted = vector_store_svc.get_task_by_id(task_id=task_id)
    return {"detail": "Xóa tasks thành công.", "task_deleted": task_deleted}

@crud_router.delete(
    "/users/{user_id}",
    summary="Xóa user theo ID",
    description="""""
Xóa một user khỏi vector store dựa trên user ID.

Args:
- user_id: ID của user.

Return:
- Thông báo thành công và thông tin user đã bị xóa (nếu có).
""""",
)
async def delete_user_by_id(
    user_id: int = Path(..., description="ID của user cần xóa."),
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Xoá user khỏi vector store theo user ID."""
    success = vector_store_svc.delete_user_by_id(user_id=user_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found or could not be deleted.")
    user_deleted = vector_store_svc.get_user_by_id(user_id=user_id)
    return {"detail": "Xóa user thành công.", "user_deleted": user_deleted}

@crud_router.delete(
    "/projects/{project_id}",
    summary="Xóa project theo ID",
    description="""""
Xóa một project khỏi vector store dựa trên project ID.

Args:
- project_id: ID của project.

Return:
- Thông báo thành công và thông tin project đã bị xóa (nếu có).
""""",
)
async def delete_project_by_id(
    project_id: int = Path(..., description="ID của project cần xóa."),
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Xoá project khỏi vector store theo project ID."""
    success = vector_store_svc.delete_project_by_id(project_id=project_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found or could not be deleted.")
    project_deleted = vector_store_svc.get_project_by_id(project_id=project_id)
    return {"detail": "Xóa project thành công.", "project_deleted": project_deleted}

@crud_router.delete(
    "/teams/{team_id}",
    summary="Xóa team members map theo ID",
    description="""""
Xóa mapping teamId -> list các userId và role của họ khỏi vector store dựa trên team ID.
Args:
- team_id: ID của team. 
Return:
- Thông báo thành công và thông tin team members map đã bị xóa (nếu có).
    """"",)
async def delete_team_by_id(
    team_id: int = Path(..., description="ID của team cần xóa."),
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Xoá mapping teamId -> list các userId và role của họ khỏi vector store theo team ID."""
    success = vector_store_svc.delete_team_by_id(team_id=team_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team not found or could not be deleted.")
    team_deleted = vector_store_svc.get_team_by_id(team_id=team_id)
    return {"detail": "Xóa team thành công.", "team_deleted": team_deleted}
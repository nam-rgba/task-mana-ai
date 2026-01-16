import logging
from fastapi import APIRouter, Depends, UploadFile, File
from app.services.vector_store import VectorStoreService
from fastapi import APIRouter, Depends, HTTPException
from app.utils.extractFileHelper import extract_text_from_file, extract_text_from_multiple_files

log = logging.getLogger(__name__)

test_router = APIRouter(
    prefix="/test",
    tags=["AI / Debug & Test"]
)

def get_vector_store_service():
    from app.routers import vector_store_service
    return vector_store_service
# DEBUG 
# ------------------- TEST ROUTES -------------------

# Test truy vần task liên quan
@test_router.get("/retrieve_tasks")
async def test_retrieve_tasks(
    query: str,
    project_id: str = None,
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service),
):
    """Test retrieve tasks from vector store by query + optional project_id"""
    tasks = vector_store_svc.retrieve_tasks_by_query(query=query, project_id=project_id)
    return {"query": query, "project_id": project_id, "results": tasks}

# Test truy vần user liên quan
@test_router.get("/retrieve_users")
async def test_retrieve_users(
    query: str,
    project_id: str = None,
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Test retrieve users from vector store by query + optional project_id"""
    users = vector_store_svc.retrieve_users_by_query(task_text=query, project_id=project_id)
    return {"query": query, "project_id": project_id, "results": users}

# Test truy vần task kèm score
@test_router.get("/retrieve_with_scores")
async def test_retrieve_with_scores(
    query: str,
    project_id: str = None,
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service),
):
    """Test retrieve tasks with scores (similarity search)"""
    results = vector_store_svc.retrieve_tasks_with_scores(query=query, project_id=project_id)
    return {"query": query, "project_id": project_id, "results": results}

# Test truy vần task chỉ bằng project_id
@test_router.get("/retrieve_tasks_by_project")
async def test_retrieve_task_by_project(
    project_id: str,
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Test retrieve tasks from vector store by project_id only"""
    tasks = vector_store_svc.retrieve_tasks_by_project(project_id=project_id)
    return {"project_id": project_id, "results": tasks}
   

# Test truy vấn user chỉ bằng project_id
@test_router.get("/retrieve_users_by_project")
async def test_retrieve_users_by_project(
    project_id: str,
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Test retrieve users from vector store by project_id only"""
    users = vector_store_svc.retrieve_users_by_project(project_id=project_id)
    return {"project_id": project_id, "results": users}


@test_router.get("/retrieve_tasks_by_user")
async def test_retrieve_task_by_user(
    user_id: str,
    project_id: str,
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Test retrieve tasks assigned to a specific user_id"""
    tasks = vector_store_svc.retrieve_tasks_by_user(user_id=user_id, project_id=project_id)
    return {"user_id": user_id, "project_id": project_id, "results": tasks}


@test_router.get("/retrieve_guides")
async def test_retrieve_guides(
    query: str,
    k: int = 3,
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

@test_router.post("/read_file")
async def test_read_file(
    file: UploadFile = File(...)
): 
    try:
        text = await extract_text_from_file(file)
        return {"filename": file.filename, "content": text}
    except Exception as e:
        log.error(f"Failed to read file: {e}")
        raise HTTPException(status_code=500, detail=f"File read failed: {str(e)}")

@test_router.post("/read_multiple_files")
async def test_read_file(
    files: list[UploadFile] = File(...)
): 
    try:
        text = await extract_text_from_multiple_files(files)
        return {"filenames": [file.filename for file in files], "content": text}
    except Exception as e:
        log.error(f"Failed to read file: {e}")
        raise HTTPException(status_code=500, detail=f"File read failed: {str(e)}")

@test_router.get("/get_data_from_api")
async def test_get_data_from_api():
    """Test fetching data from external API"""
    from app.services.fetch_data import FetchData

    projects = await FetchData.get_projects()
    tasks = await FetchData.get_tasks()
    members = await FetchData.get_members()

    return {
        "projects_count": len(projects["metadata"]) if projects and "metadata" in projects else 0,
        "tasks_count": len(tasks["metadata"]["tasks"]) if tasks and "metadata" in tasks and "tasks" in tasks["metadata"] else 0,
        "members_count": len(members["metadata"]) if members and "metadata" in members else 0,
        "projects_json": projects,
        "tasks_json": tasks,
        "members_json": members
    }
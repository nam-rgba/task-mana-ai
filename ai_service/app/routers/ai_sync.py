import logging
import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from app.services.vector_store import VectorStoreService
from typing import Optional
import os

log = logging.getLogger(__name__)

sync_router = APIRouter(
    prefix="/sync",
    tags=["AI / Data Sync"]
)

def get_vector_store_service(request: Request):
    # from app.routers import vector_store_service
    # return vector_store_service
    return request.app.state.vector_store_service

# Backend Node.js URL - có thể config trong .env
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:3000")

@sync_router.post("/tasks-from-backend")
async def sync_tasks_from_backend(
    page_limit: int = Query(default=100, description="Số lượng tasks mỗi page"),
    max_pages: Optional[int] = Query(default=None, description="Giới hạn số pages (None = sync all)"),
    force: bool = Query(default=True, description="Force upsert (xóa bản cũ trước khi insert)"),
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """
    Sync tasks từ backend Node.js vào AI service vector store.
    
    Flow:
    1. Gọi API /aidata/tasks với pagination từ backend Node.js
    2. Với mỗi task, upsert vào vector store
    3. Tiếp tục cho đến khi hết data hoặc đạt max_pages
    
    Args:
        page_limit: Số tasks mỗi lần fetch (default: 100)
        max_pages: Giới hạn số pages (None = sync tất cả)
        force: Xóa task cũ trước khi insert (default: True)
    
    Returns:
        Thống kê sync: total, success, failed, errors
    """
    
    results = {
        "total_fetched": 0,
        "total_synced": 0,
        "total_failed": 0,
        "pages_processed": 0,
        "errors": []
    }
    
    current_page = 1
    has_more = True
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        while has_more:
            # Kiểm tra giới hạn pages
            if max_pages and current_page > max_pages:
                log.info(f"⏹️  Reached max_pages limit: {max_pages}")
                break
            
            try:
                # Gọi API backend để lấy tasks
                response = await client.get(
                    f"{BACKEND_URL}/api/aidata/tasks",
                    params={
                        "page": current_page,
                        "limit": page_limit
                    }
                )
                
                # Parse response JSON
                response_data = response.json()
                
                # Kiểm tra status code
                if response_data.get("code") != 200:
                    error_msg = f"Backend API returned code {response_data.get('code')}"
                    results["errors"].append({
                        "page": current_page,
                        "error": error_msg,
                        "response": str(response_data)[:200]
                    })
                    break
                
                # Lấy metadata
                metadata = response_data.get("metadata", {})
                tasks = metadata.get("tasks", [])
                page_info = metadata.get("page", {})
                total_available = page_info.get("total", 0)
                
                # Fallback: nếu không có metadata, thử format cũ
                if not tasks and isinstance(response_data, list):
                    tasks = response_data
                    total_available = len(tasks)
                
                if not tasks or len(tasks) == 0:
                    log.info(f"✅ No more tasks on page {current_page}. Sync complete!")
                    has_more = False
                    break
                
                log.info(f"📦 Received {len(tasks)} tasks from page {current_page}")
                results["total_fetched"] += len(tasks)
                results["pages_processed"] = current_page
                
                # Sync từng task vào vector store
                for idx, task in enumerate(tasks, 1):
                    try:
                        # Chuẩn bị task data
                        task_data = {
                            "id": str(task.get("id")),
                            "title": task.get("title", ""),
                            "description": task.get("description", ""),
                            "status": task.get("status", ""),
                            "dueDate": task.get("dueDate"),
                            "estimateEffort": task.get("estimateEffort"),
                            "actualEffort": task.get("actualEffort"),
                            "implementorId": str(task.get("assigneeId")) if task.get("assigneeId") else None,
                            "reviewerId": str(task.get("reviewerId")) if task.get("reviewerId") else None,
                            "projectId": str(task.get("projectId")) if task.get("projectId") else None,
                            "priority": task.get("priority", "MEDIUM"),
                            "completedAt": task.get("completedAt"),
                            "type": task.get("type"),
                            "score": task.get("score")  # Quality score
                        }
                        
                        # Upsert vào vector store
                        success = vector_store_svc.upsert_task(task=task_data, force=force)
                        
                        if success:
                            results["total_synced"] += 1
                            if idx % 10 == 0:
                                log.info(f"   ✓ Synced {idx}/{len(tasks)} tasks from page {current_page}")
                        else:
                            results["total_failed"] += 1
                            results["errors"].append({
                                "task_id": task_data["id"],
                                "title": task_data["title"],
                                "error": "upsert_task returned False"
                            })
                            
                    except Exception as e:
                        results["total_failed"] += 1
                        results["errors"].append({
                            "task_id": task.get("id", "unknown"),
                            "title": task.get("title", "unknown"),
                            "error": str(e)
                        })
                        log.error(f"   ❌ Failed to sync task {task.get('id')}: {str(e)}")
                
                log.info(f"✅ Page {current_page} completed: {results['total_synced']}/{results['total_fetched']} synced")
                
                # Kiểm tra xem còn data không
                if len(tasks) < page_limit:
                    log.info(f"✅ Received less than {page_limit} tasks. No more data!")
                    has_more = False
                else:
                    current_page += 1
                    
            except httpx.TimeoutException:
                error_msg = f"Timeout when fetching page {current_page}"
                log.error(f"❌ {error_msg}")
                results["errors"].append({
                    "page": current_page,
                    "error": error_msg
                })
                break
                
            except Exception as e:
                error_msg = f"Error on page {current_page}: {str(e)}"
                log.error(f"❌ {error_msg}")
                results["errors"].append({
                    "page": current_page,
                    "error": error_msg
                })
                break
    
    # Summary
    log.info("=" * 70)
    log.info("📊 SYNC SUMMARY")
    log.info("=" * 70)
    log.info(f"Total fetched: {results['total_fetched']}")
    log.info(f"Total synced: {results['total_synced']}")
    log.info(f"Total failed: {results['total_failed']}")
    log.info(f"Pages processed: {results['pages_processed']}")
    log.info(f"Errors: {len(results['errors'])}")
    log.info("=" * 70)
    
    return {
        "success": True,
        "message": f"Synced {results['total_synced']}/{results['total_fetched']} tasks from {results['pages_processed']} pages",
        "stats": results
    }


@sync_router.get("/status")
async def get_sync_status(
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """
    Kiểm tra số lượng tasks hiện có trong vector store.
    """
    # Note: Cần implement method count_tasks() trong VectorStoreService
    # Tạm thời return placeholder
    return {
        "backend_url": BACKEND_URL,
        "vector_store": "ChromaDB",
        "status": "ready",
        "message": "Use POST /sync/tasks-from-backend to sync data"
    }


# Hàm gọi sync dữ liệu vào vector database
@sync_router.post("/all-new-data")
async def sync_all_new_data(
        vector_store_svc: VectorStoreService = Depends(get_vector_store_service),
        force: bool = False
):
    """
    Hàm này có thể được gọi định kỳ để tự động sync dữ liệu mới từ backend vào vector store.
    """
    # Gọi hàm sync_tasks_from_backend với các tham số mặc định
    await vector_store_svc.sync_tasks_from_api(force=force)
    await vector_store_svc.sync_projects_from_api(force=force)
    await vector_store_svc.sync_users_from_api(force=force)
    await vector_store_svc.sync_team_member_from_api(force=force)

    return {
        "success": True,
        "message": "Đã sync tất cả dữ liệu mới từ backend vào vector store"
    }

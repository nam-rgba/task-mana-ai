from fastapi import FastAPI
from app.routers import api_router
from contextlib import asynccontextmanager
import psutil
import logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Import các service và sync dữ liệu khi app start
    from app.services.vector_store import VectorStoreService
    from app.services.llm_service import LLMService

    # Khởi tạo vector store và sync dữ liệu
    vector_store_service = VectorStoreService()
    # vector_store_service.sync_data(force=False)
    await vector_store_service.sync_tasks_from_api(force=False)
    await vector_store_service.sync_projects_from_api(force=False)
    await vector_store_service.sync_users_from_api(force=False)
    await vector_store_service.sync_team_member_from_api(force=False)

    # Khởi tạo các service khác, truyền vector_store vào
    llm_service = LLMService(vector_store=vector_store_service)

    # Lưu vào app.state để các router/service khác có thể truy cập
    app.state.vector_store_service = vector_store_service
    app.state.llm_service = llm_service

    # Sau yield sẽ chạy khi app shutdown
    yield

app = FastAPI(title="Tasks AI Service", version="1.0.0", lifespan=lifespan)
app.include_router(api_router)

@app.get("/health")
def health():
    # Lấy trạng thái các service
    vector_store_status = getattr(app.state, "vector_store_service", None) is not None
    llm_status = getattr(app.state, "llm_service", None) is not None
    chat_status = getattr(app.state, "chat_service", None) is not None

    # Lấy thông tin hệ thống
    cpu_percent = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    mem_percent = mem.percent

    # Log chi tiết
    logging.info(f"Health check: vector_store={vector_store_status}, llm={llm_status}, chat={chat_status}, cpu={cpu_percent}%, mem={mem_percent}%")

    return {
        "status": "ok",
        "services": {
            "vector_store": vector_store_status,
            "llm": llm_status,
            "chat": chat_status
        },
        "system": {
            "cpu_percent": cpu_percent,
            "mem_percent": mem_percent
        }
    }

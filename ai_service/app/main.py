from fastapi import FastAPI
from app.routers import api_router
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Import các service và sync dữ liệu khi app start
    from app.services.vector_store import VectorStoreService
    from app.services.llm_service import LLMService
    from app.services.chat_service import ChatService

    # Khởi tạo vector store và sync dữ liệu
    vector_store_service = VectorStoreService()
    # vector_store_service.sync_data(force=False)
    await vector_store_service.sync_tasks_from_api(force=False)
    await vector_store_service.sync_projects_from_api(force=False)

    # Khởi tạo các service khác, truyền vector_store vào
    llm_service = LLMService(vector_store=vector_store_service)
    chat_service = ChatService(vector_store=vector_store_service)

    # Lưu vào app.state để các router/service khác có thể truy cập
    app.state.vector_store_service = vector_store_service
    app.state.llm_service = llm_service
    app.state.chat_service = chat_service

    # Sau yield sẽ chạy khi app shutdown
    yield

app = FastAPI(title="Tasks AI Service", version="1.0.0", lifespan=lifespan)
app.include_router(api_router)

@app.get("/health")
def health():
    return {"status": "ok"}

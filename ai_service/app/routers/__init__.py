# app/routers/__init__.py
from fastapi import APIRouter
from .ai_compose import compose_router
from .ai_crud import crud_router
from .ai_chat import chat_router
from .ai_test import test_router

# Khởi tạo các service global để dùng chung
from app.services.vector_store import VectorStoreService
from app.services.llm_service import LLMService
from app.services.chat_service import ChatService

vector_store_service = VectorStoreService()
llm_service = LLMService(vector_store=vector_store_service)
chat_service = ChatService(vector_store=vector_store_service)

# Tạo main API router
api_router = APIRouter(prefix="/ai")
api_router.include_router(compose_router)
api_router.include_router(crud_router)
api_router.include_router(chat_router)
api_router.include_router(test_router)

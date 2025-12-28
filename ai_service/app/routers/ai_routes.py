import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder

from app.services.llm_service import LLMService
from app.services.vector_store import VectorStoreService 
from app.services.chat_service import ChatService
from app.schema.input import *

log = logging.getLogger(__name__)



# --- KHỞI TẠO SERVICE ---
# Khởi tạo Vector Store instance lúc app khởi động (call instance method)
vector_store_service = VectorStoreService()
try:
    vector_store_service.sync_data(force=False) # Đồng bộ dữ liệu từ DB vào vector store
except Exception as e:
    log.warning("Vector store embedding initialization failed at startup: %s", e)
    
# Initialize LLM service after vector store init and inject vector store instance
llm_service = LLMService(vector_store=vector_store_service)
chat_service = ChatService(vector_store=vector_store_service)

router = APIRouter(prefix="/ai")




# --- Register grouped routers into main router ---
router.include_router(compose_router)
router.include_router(crud_router)
router.include_router(test_router)
router.include_router(chat_router)
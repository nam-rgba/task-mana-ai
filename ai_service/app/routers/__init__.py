# app/routers/__init__.py
from fastapi import APIRouter
from .ai_compose import compose_router
from .ai_crud import crud_router
from .ai_chat import chat_router
from .ai_test import test_router
from .ai_sync import sync_router
from .xgb_routes import xgb_router


# Tạo main API router
api_router = APIRouter(prefix="/ai")
api_router.include_router(compose_router)
api_router.include_router(crud_router)
api_router.include_router(chat_router)
api_router.include_router(test_router)
api_router.include_router(sync_router)
api_router.include_router(xgb_router)

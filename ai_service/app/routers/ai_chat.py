import logging
from fastapi import APIRouter, Depends, Request
from app.services.chat_service import ChatService
from app.schema.input import *

log = logging.getLogger(__name__)
chat_router = APIRouter(prefix="/chat", tags=["AI / Chat"])

def get_chat_service(request: Request):
    # from app.routers import chat_service
    # return chat_service
    return request.app.state.chat_service


# ------------------- CHAT ROUTES -------------------
#Chat
@chat_router.post("/ask")
async def chat_ask(
    body: ChatRequest,
    chat_svc: ChatService = Depends(get_chat_service)
):
    """Hỏi đáp với AI kèm lưu trữ lịch sử theo session_id."""
    answer = chat_svc.ask(question=body.question, session_id=body.session_id)
    return {"answer": answer, "session_id": body.session_id}

#Xóa session chat
@chat_router.post("/clear")
async def chat_clear_memory(
    session_id: str = "default",
    chat_svc: ChatService = Depends(get_chat_service)
):
    """Xóa lịch sử chat ngay lập tức theo session_id."""
    chat_svc.clear_memory(session_id=session_id)
    return {"detail": f"Memory for session {session_id} cleared."}

@chat_router.get("/history")
async def chat_get_history(
    session_id: str,
    chat_svc: ChatService = Depends(get_chat_service)
):
    """Lấy toàn bộ lịch sử chat theo session_id."""
    history = chat_svc.get_conversation_history(session_id=session_id)
    return {"session_id": session_id, "history": history}


import logging
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Request
from app.services.llm_service import LLMService
from app.schema.input import *
from app.utils.extractFileHelper import extract_text_from_multiple_files

log = logging.getLogger(__name__)

# Tạo group router để dễ phân biệt
compose_router = APIRouter(
    prefix="/llm", 
    tags=["AI / Compose & Generate"]
)

# Lấy LLMService từ app.state
def get_llm_service(request: Request):
    return request.app.state.llm_service


# COMPOSE TASK ROUTE
@compose_router.post("/compose")
async def compose_task(
    body: ComposeRequest,
    llm_svc: LLMService = Depends(get_llm_service)
):
    """Tạo một task mới (title, description, subtasks, tags, priority) dựa trên mô tả từ người dùng."""
    try:
        # Gọi hàm tạo task từ LLMService (accept raw dict)
        user_input = body.user_input or ""
        project_id = body.project_id  #nhận projectId từ client
        raw_result = llm_svc.compose_with_llm(user_input=user_input, project_id=project_id)
        
        if 'error' in raw_result:
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=raw_result['error'])
        if 'raw' in raw_result:
             raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="LLM response was not valid JSON.")
        
        # Return raw dict (no schema enforcement)
        return raw_result

    except (TypeError, ValueError) as e:
        log.error(f"Input/processing error in /compose: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except Exception as e:
        log.exception(f"Unexpected error in /compose: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during task composition.")

@compose_router.post("/compose_with_files")
async def compose_task_with_files(
    user_input: str = Form(...),
    project_id: int = Form(None),
    files: list[UploadFile] = File(None),
    llm_svc: LLMService = Depends(get_llm_service) 
):
    try:
        # Xử lý trích xuất nội dung từ các file đính kèm
        attach_context = ""
        if files:
            attach_context = await extract_text_from_multiple_files(files)
        # Gọi hàm tạo task từ LLMService với ngữ cảnh bổ sung từ file đính kèm
        raw_result = llm_svc.compose_with_llm(
            user_input=user_input,
            project_id=project_id,
            attach_context=attach_context
        )
        if 'error' in raw_result:
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=raw_result['error'])
        if 'raw' in raw_result:
             raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="LLM response was not valid JSON.")
        
        # Return raw dict (no schema enforcement)
        return raw_result
    except Exception as e:
        log.exception(f"Unexpected error in /compose_with_files: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during file processing.")

# ASSIGN TASK ROUTE
@compose_router.post("/assign")
async def assign_task(
    body: AssignRequest,
    llm_svc: LLMService = Depends(get_llm_service)
):
    """Phân công task cho thành viên phù hợp nhất."""
    try:
        # Gọi hàm phân công từ LLMService
        task_payload = body.task or {}
        requirement_text = body.requirement_text or ""
        project_id = body.project_id
        raw_result = llm_svc.assign_candidate(
            task=task_payload,
            project_id=project_id,
            requirement_text=requirement_text
        )
 
        if 'error' in raw_result:
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=raw_result['error'])
        if 'raw' in raw_result:
             raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="LLM response was not valid JSON for assignment.")
            
        return raw_result
        
    except (TypeError, ValueError) as e:
        log.error(f"Input/processing error in /assign: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except Exception as e:
        log.exception(f"Unexpected error in /assign: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during task assignment.")


# DUPLICATE FINDER TASK ROUTE
@compose_router.post("/duplicate")
async def find_duplicates(
    body: DuplicateRequest,
    llm_svc: LLMService = Depends(get_llm_service)
):
    """Tìm các task có nội dung tương tự (trùng lặp) dựa trên semantic search."""
    try:
        # Gọi hàm tìm kiếm trùng lặp từ LLMService
        task_payload = body.task or {}
        project_id = body.project_id  # <-- nhận projectId để giới hạn search
        raw_result = llm_svc.find_duplicate_tasks(
            task=task_payload,
            threshold=0.25, # Euclidean distance threshold
            k=3,
            project_id=project_id
        )
        
        if 'error' in raw_result:
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=raw_result['error'])
             
        return raw_result

    except (TypeError, ValueError) as e:
        log.error(f"Input/processing error in /duplicate: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except Exception as e:
        log.exception(f"Unexpected error in /duplicate: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during duplicate check.")
    

# ESTIMATE STORY POINT ROUTE
@compose_router.post("/estimate_sp")
async def estimate_story_point(
    body: EstimateSPRequest,
    llm_svc: LLMService = Depends(get_llm_service)
):
    """Ước tính Story Point cho task dựa trên Title và Description."""
    
    try:
        title = body.title or ""
        description = body.description or ""
        type_val = body.type or "FEATURE"
        priority_val = body.priority or "MEDIUM"
        # Dự đoán giá trị Story Point thô
        raw_pred = llm_svc.predict_story_point(title, description, type_val, priority_val)
        
        # Gợi ý Story Point theo Planning Poker
        suggested_sp = llm_svc.suggest_story_point(raw_pred)
        
        return {
            "model_estimate": round(raw_pred, 2),
            "suggested_story_point": suggested_sp
        }
        
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        log.exception(f"Error during Story Point estimation: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during story point estimation.")

# SUMMARIZE ROUTE (UPDATED)
@compose_router.post("/summarize")
async def summarize_text(
    body: SummarizeRequest,
    llm_svc: LLMService = Depends(get_llm_service)
):
    """
    Body:
      {
        "tasks": [ ... ],            # list of task objects (see example)
        "use_llm": true|false        # optional, default true
      }
    Returns sprint report including local metrics and optional LLM summary.
    """
    try:
        tasks = body.tasks
        if not isinstance(tasks, list):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Missing or invalid 'tasks' list in request body."
            )

        use_llm = bool(body.use_llm)

        # Generate sprint report: local metrics + optional LLM summary
        report = llm_svc.generate_sprint_report(tasks=tasks, use_llm_summary=use_llm)

        return report

    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"Unexpected error in /summarize: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during summarization."
        )

# GENERATE TASK (PIPELINE)
@compose_router.post("/generate_task")
async def generate_task(
    body: GenerateTaskRequest,
    llm_svc: LLMService = Depends(get_llm_service)
):
    """Tạo task mới với pipeline đầy đủ: compose, assign, duplicate check, estimate SP."""
    try:
        user_input = body.user_input or ""
        project_id = body.project_id  # <-- nhận projectId từ client
        requirement_text = body.requirement_text or ""


        raw_result = llm_svc.generate_task(
            user_input=user_input,
            project_id=project_id,
            requirement_text=requirement_text
        )
        
        if 'error' in raw_result:
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=raw_result['error'])
        if 'raw' in raw_result:
             raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="LLM response was not valid JSON during full task generation.")
        
        return raw_result

    except (TypeError, ValueError) as e:
        log.error(f"Input/processing error in /generate_task: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except Exception as e:
        log.exception(f"Unexpected error in /generate_task: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Internal server error during full task generation.")

@compose_router.post("/generate_task_with_files")
async def generate_task_with_files(
    user_input: str = Form(...),
    project_id: int = Form(None),
    requirement_text: str = Form(""),
    files: list[UploadFile] = File(None),
    llm_svc: LLMService = Depends(get_llm_service)
):
    try:
        # Xử lý trích xuất nội dung từ các file đính kèm
        attach_context = ""
        if files:
            attach_context = await extract_text_from_multiple_files(files)
        log.info(f"Extracted context from files: {attach_context[:500]}...")  # Log first 500 chars
        raw_result = llm_svc.generate_task(
            user_input=user_input,
            project_id=project_id,
            requirement_text=requirement_text,
            attach_context=attach_context,
        )
        
        if 'error' in raw_result:
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=raw_result['error'])
        if 'raw' in raw_result:
             raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="LLM response was not valid JSON during full task generation.")
        
        return raw_result
    except (TypeError, ValueError) as e:
        log.error(f"Input/processing error in /generate_task: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except Exception as e:
        log.exception(f"Unexpected error in /generate_task: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Internal server error during full task generation.")


# SUGGEST TASKS FOR TODAY ROUTE
@compose_router.post("/suggest_tasks_for_today")
async def suggest_tasks_for_today(
    body: SuggestTasksRequest,
    llm_svc: LLMService = Depends(get_llm_service)
):
    """Gợi ý các task nên làm trong ngày hôm nay dựa trên ưu tiên và deadline."""
    try:
        user_id = body.user_id
        project_id = body.project_id
        k = body.k or 3
        result = llm_svc.suggest_tasks_for_today(
            user_id=user_id,
            project_id=project_id,
            k=k
        )
        return result
    except Exception as e:
        log.exception(f"Unexpected error in /suggest_tasks_for_today: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during suggesting tasks for today.")

@compose_router.post("/suggest_tasks_for_today_llm")
async def suggest_tasks_for_today_llm(
    body: SuggestTasksRequest,
    llm_svc: LLMService = Depends(get_llm_service)
):
    """Gợi ý các task nên làm trong ngày hôm nay dựa trên ưu tiên và deadline, sử dụng LLM để xếp hạng."""
    try:
        user_id = body.user_id
        project_id = body.project_id
        k = body.k or 3
        result = llm_svc.suggest_tasks_for_today_llm(
            user_id=user_id,
            project_id=project_id,
            k=k
        )
        return result
    except Exception as e:
        log.exception(f"Unexpected error in /suggest_tasks_for_today_llm: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during suggesting tasks for today with LLM.")
    
@compose_router.post("/suggest_new_task_today")
async def suggest_new_task_today(
    body: SuggestTasksRequest,
    llm_svc: LLMService = Depends(get_llm_service)
):
    """Gợi ý một task mới nên tạo hôm nay dựa trên lịch sử task của project và user (nếu có)."""
    try:
        user_id = body.user_id
        project_id = body.project_id
        k = body.k or 1
        result = llm_svc.suggest_new_task_today(
            user_id=user_id,
            project_id=project_id,
            k=k
        )
        return result
    except Exception as e:
        log.exception(f"Unexpected error in /suggest_new_task_today: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during suggesting new task for today.")
import logging
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
    Request,
)
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStoreService
from app.schema.input import *
from app.utils.extractFileHelper import extract_text_from_multiple_files

log = logging.getLogger(__name__)

# Tạo group router để dễ phân biệt
compose_router = APIRouter(prefix="/llm", tags=["AI / Compose & Generate"])

MAX_SPEC_LENGTH = 10000

# Lấy LLMService từ app.state
def get_llm_service(request: Request):
    return request.app.state.llm_service

# Lấy vector store từ app.statr
def get_vector_store_service(request: Request):
    # from app.routers import vector_store_service
    # return vector_store_service
    return request.app.state.vector_store_service


# COMPOSE TASK ROUTE
@compose_router.post(
    "/compose",
    summary="Tạo task mới tự động từ mô tả ngắn",
    description="""
Flow:
1. Nhận mô tả tự nhiên về task từ người dùng và mã dự án.
2. Lấy ngữ cảnh dự án và các task liên quan.
3. Sinh ra task mới bằng AI.

Args:
- user_input: Mô tả nhiệm vụ.
- project_id: ID dự án.

Return:
- Thông tin task đã được AI sinh ra (title, description, priority, type, due_date, todos).
""",
)
async def compose_task(
    body: ComposeRequest, llm_svc: LLMService = Depends(get_llm_service)
):
    """Tạo một task mới dựa trên mô tả từ người dùng."""
    try:
        # Gọi hàm tạo task từ LLMService (accept raw dict)
        user_input = body.user_input or ""
        project_id = body.project_id  # nhận projectId từ client
        raw_result = llm_svc.compose_with_llm(
            user_input=user_input, project_id=project_id
        )

        if "error" in raw_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=raw_result["error"],
            )
        if "raw" in raw_result:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="LLM response was not valid JSON.",
            )

        # Return raw dict (no schema enforcement)
        return raw_result

    except (TypeError, ValueError) as e:
        log.error(f"Input/processing error in /compose: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)
        )
    except Exception as e:
        log.exception(f"Unexpected error in /compose: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during task composition.",
        )


@compose_router.post(
    "/compose_with_files",
    summary="Tạo task mới từ mô tả và file đính kèm",
    description="""
Flow:
1. Nhận mô tả và file đính kèm từ người dùng.
2. Trích xuất nội dung file.
3. Sinh task mới dựa trên mô tả và nội dung file.

Args:
- user_input: Mô tả nhiệm vụ.
- project_id: ID dự án.
- files: Danh sách file đính kèm.

Return:
- Thông tin task đã được AI sinh ra (title, description, priority, type, due_date, todos).
""",
)
async def compose_task_with_files(
    user_input: str = Form(...),
    project_id: int = Form(...),
    files: list[UploadFile] = File(None),
    llm_svc: LLMService = Depends(get_llm_service),
):
    try:
        # Xử lý trích xuất nội dung từ các file đính kèm
        attach_context = ""
        if files:
            attach_context = await extract_text_from_multiple_files(files)
        # Gọi hàm tạo task từ LLMService với ngữ cảnh bổ sung từ file đính kèm
        raw_result = llm_svc.compose_with_llm(
            user_input=user_input, project_id=project_id, attach_context=attach_context
        )
        if "error" in raw_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=raw_result["error"],
            )
        if "raw" in raw_result:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="LLM response was not valid JSON.",
            )

        # Return raw dict (no schema enforcement)
        return raw_result
    except Exception as e:
        log.exception(f"Unexpected error in /compose_with_files: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail= str(e),
        )


# ASSIGN TASK ROUTE
@compose_router.post(
    "/assign",
    summary="Phân công task cho thành viên phù hợp",
    description="""
Flow:
1. Nhận thông tin task và yêu cầu phân công (nếu có).
2. Lấy danh sách ứng viên theo yêu cầu (không có thì lấy toàn bộ thành viên trong nhóm) và lấy tải công việc.
3. AI phân tích và chọn người thực hiện phù hợp nhất.

Args:
- task: Thông tin nhiệm vụ.
- requirement_text: Yêu cầu phân công.
- project_id: ID dự án.

Return:
- assignment:
    - assignee: Thông tin người được phân công.
    - reason: Giải thích lý do chọn.
- related_users: Danh sách các user liên quan đến yêu cầu phân công.
""",
)
async def assign_task(
    body: AssignRequest, llm_svc: LLMService = Depends(get_llm_service)
):
    """Phân công task cho thành viên phù hợp nhất."""
    try:
        # Gọi hàm phân công từ LLMService
        task_payload = body.task or {}
        requirement_text = body.requirement_text or ""
        project_id = body.project_id
        raw_result = llm_svc.assign_candidate(
            task=task_payload, project_id=project_id, requirement_text=requirement_text
        )

        if "error" in raw_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=raw_result["error"],
            )
        if "raw" in raw_result:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="LLM response was not valid JSON for assignment.",
            )

        return raw_result

    except (TypeError, ValueError) as e:
        log.error(f"Input/processing error in /assign: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)
        )
    except Exception as e:
        log.exception(f"Unexpected error in /assign: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail= str(e),
        )


# DUPLICATE FINDER TASK ROUTE
@compose_router.post(
    "/duplicate",
    summary="Kiểm tra trùng lặp task",
    description="""
Flow:
1. Nhận thông tin task.
2. Tìm các task có nội dung tương tự trong dự án.

Args:
- task: Thông tin nhiệm vụ.
- project_id: ID dự án.

Return:
- duplicates: Danh sách các task gần giống nhất (content, score, metadata).
- nearest_tasks: Danh sách các task gần nhất với embedding.
""",
)
async def find_duplicates(
    body: DuplicateRequest, llm_svc: LLMService = Depends(get_llm_service)
):
    """Tìm các task có nội dung tương tự (trùng lặp) dựa trên semantic search."""
    try:
        # Gọi hàm tìm kiếm trùng lặp từ LLMService
        task_payload = body.task or {}
        project_id = body.project_id  # <-- nhận projectId để giới hạn search
        raw_result = llm_svc.find_duplicate_tasks(
            task=task_payload,
            threshold=0.25,  # Euclidean distance threshold
            k=3,
            project_id=project_id,
        )

        if "error" in raw_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=raw_result["error"],
            )

        return raw_result

    except (TypeError, ValueError) as e:
        log.error(f"Input/processing error in /duplicate: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)
        )
    except Exception as e:
        log.exception(f"Unexpected error in /duplicate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# ESTIMATE STORY POINT ROUTE
@compose_router.post(
    "/estimate_sp",
    summary="Ước tính Story Point cho task",
    description="""
Flow:
1. Nhận thông tin task (title, description, type, priority).
2. AI dự đoán giá trị Story Point.
3. Gợi ý Planning Poker.

Args:
- title: Tiêu đề task.
- description: Mô tả task.
- type: Loại task.
- priority: Mức độ ưu tiên.

Return:
- model_estimate: Giá trị Story Point thô.
- suggested_story_point: Gợi ý Planning Poker.
""",
)
async def estimate_story_point(
    body: EstimateSPRequest, llm_svc: LLMService = Depends(get_llm_service)
):
    """Ước tính Story Point cho task dựa trên Title và Description."""

    try:
        title = body.title or ""
        description = body.description or ""
        type_val = body.type or "FEATURE"
        priority_val = body.priority or "MEDIUM"
        # Dự đoán giá trị Story Point thô
        raw_pred = llm_svc.predict_story_point(
            title, description, type_val, priority_val
        )

        # Gợi ý Story Point theo Planning Poker
        suggested_sp = llm_svc.suggest_story_point(raw_pred)

        return {
            "model_estimate": round(raw_pred, 2),
            "suggested_story_point": suggested_sp,
        }

    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        log.exception(f"Error during Story Point estimation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# SUMMARIZE ROUTE (UPDATED)
@compose_router.post(
    "/summarize",
    summary="Tổng hợp sprint report",
    description="""
Flow:
1. Nhận danh sách tasks.
2. Tính toán metrics sprint (velocity, burndown, lead/cycle time).
3. AI tóm tắt sprint và đề xuất cải tiến.

Args:
- tasks: Danh sách nhiệm vụ.
- use_llm: Có dùng AI tóm tắt không.

Return:
- summary: Tóm tắt sprint.
- metrics: Thông số sprint.
- recommendations: Đề xuất cải tiến.
""",
)
async def summarize_text(
    body: SummarizeRequest, llm_svc: LLMService = Depends(get_llm_service)
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
                detail="Missing or invalid 'tasks' list in request body.",
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
            detail=str(e),
        )


# GENERATE TASK (PIPELINE)
@compose_router.post(
    "/generate_task",
    summary="Tạo task mới với pipeline AI đầy đủ",
    description="""
Flow:
1. Nhận mô tả, yêu cầu, project.
2. AI sinh task mới, phân công, kiểm tra trùng lặp, ước tính Story Point.

Args:
- user_input: Mô tả nhiệm vụ.
- project_id: ID dự án.
- requirement_text: Yêu cầu bổ sung.

Return:
- composed_task: Task đã sinh.
- estimated_story_point: Giá trị Story Point.
- story_point_suggestion: Gợi ý Planning Poker.
- duplicates: Danh sách task trùng lặp.
- assignment: Thông tin người được phân công.
""",
)
async def generate_task(
    body: GenerateTaskRequest, llm_svc: LLMService = Depends(get_llm_service)
):
    """Tạo task mới với pipeline đầy đủ: compose, assign, duplicate check, estimate SP."""
    try:
        user_input = body.user_input or ""
        project_id = body.project_id  # <-- nhận projectId từ client
        requirement_text = body.requirement_text or ""

        raw_result = llm_svc.generate_task(
            user_input=user_input,
            project_id=project_id,
            requirement_text=requirement_text,
        )

        if "error" in raw_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=raw_result["error"],
            )
        if "raw" in raw_result:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="LLM response was not valid JSON during full task generation.",
            )

        return raw_result

    except (TypeError, ValueError) as e:
        log.error(f"Input/processing error in /generate_task: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)
        )
    except Exception as e:
        log.exception(f"Unexpected error in /generate_task: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        )


@compose_router.post(
    "/generate_task_with_files",
    summary="Tạo task mới với pipeline AI và file đính kèm",
    description="""""
Flow:
1. Nhận mô tả, yêu cầu, project và file đính kèm.
2. AI tổng hợp thông tin từ file để sinh task, phân công, kiểm tra trùng lặp, ước tính Story Point.

Args:
- user_input: Mô tả nhiệm vụ.
- project_id: ID dự án.
- requirement_text: Yêu cầu bổ sung.
- files: Danh sách file đính kèm.

Return:
- composed_task: Task đã sinh.
- estimated_story_point: Giá trị Story Point.
- story_point_suggestion: Gợi ý Planning Poker.
- duplicates: Danh sách task trùng lặp.
- assignment: Thông tin người được phân công.
""",
)
async def generate_task_with_files(
    user_input: str = Form(...),
    project_id: int = Form(None),
    requirement_text: str = Form(""),
    files: list[UploadFile] = File(None),
    llm_svc: LLMService = Depends(get_llm_service),
):
    try:
        # Xử lý trích xuất nội dung từ các file đính kèm
        attach_context = ""
        if files:
            attach_context = await extract_text_from_multiple_files(files)
        log.info(
            f"Extracted context from files: {attach_context[:500]}..."
        )  # Log first 500 chars
        raw_result = llm_svc.generate_task(
            user_input=user_input,
            project_id=project_id,
            requirement_text=requirement_text,
            attach_context=attach_context,
        )

        if "error" in raw_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=raw_result["error"],
            )
        if "raw" in raw_result:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="LLM response was not valid JSON during full task generation.",
            )

        return raw_result
    except (TypeError, ValueError) as e:
        log.error(f"Input/processing error in /generate_task: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)
        )
    except Exception as e:
        log.exception(f"Unexpected error in /generate_task: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        )


# SUGGEST TASKS FOR TODAY ROUTE
@compose_router.post(
    "/suggest_tasks_for_today",
    summary="Gợi ý các task nên làm hôm nay",
    description="""""
Flow:
1. Nhận user_id, project_id.
2. AI lọc các task ưu tiên, deadline gần, status chưa hoàn thành.

Args:
- project_id: ID dự án.
- user_id: ID người dùng (tùy chọn).
- k: Số lượng task gợi ý.

Return:
- suggested_tasks: Danh sách task nên ưu tiên hôm nay.
""",
)
async def suggest_tasks_for_today(
    body: SuggestTasksRequest, llm_svc: LLMService = Depends(get_llm_service)
):
    """Gợi ý các task nên làm trong ngày hôm nay dựa trên ưu tiên và deadline."""
    try:
        user_id = body.user_id
        project_id = body.project_id
        k = body.k or 3
        result = llm_svc.suggest_tasks_for_today(
            user_id=user_id, project_id=project_id, k=k
        )
        return result
    except Exception as e:
        log.exception(f"Unexpected error in /suggest_tasks_for_today: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during suggesting tasks for today.",
        )


@compose_router.post(
    "/suggest_tasks_for_today_llm",
    summary="Gợi ý task hôm nay bằng LLM",
    description="""""
Flow:
1. Nhận user_id, project_id.
2. AI phân tích lịch sử task và đề xuất các task nên làm hôm nay.

Args:
- project_id: ID dự án.
- user_id: ID người dùng (tùy chọn).
- k: Số lượng task gợi ý.

Return:
- suggested_tasks: Danh sách task được LLM đánh giá nên làm.
""",
)
async def suggest_tasks_for_today_llm(
    body: SuggestTasksRequest, llm_svc: LLMService = Depends(get_llm_service)
):
    """Gợi ý các task nên làm trong ngày hôm nay, sử dụng LLM để xếp hạng."""
    try:
        user_id = body.user_id
        project_id = body.project_id
        k = body.k or 3
        result = llm_svc.suggest_tasks_for_today_llm(
            user_id=user_id, project_id=project_id, k=k
        )
        return result
    except Exception as e:
        log.exception(f"Unexpected error in /suggest_tasks_for_today_llm: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during suggesting tasks for today with LLM.",
        )


@compose_router.post(
    "/suggest_new_task_today",
    summary="Gợi ý task mới nên tạo hôm nay",
    description="""""
Flow:
1. Nhận user_id, project_id.
2. AI phân tích lịch sử task và đề xuất task mới nên tạo hôm nay.

Args:
- project_id: ID dự án.
- user_id: ID người dùng (tùy chọn).
- k: Số lượng task mới gợi ý.

Return:
- suggested_new_tasks: Danh sách task mới nên tạo hôm nay.
""",
)
async def suggest_new_task_today(
    body: SuggestTasksRequest, llm_svc: LLMService = Depends(get_llm_service)
):
    """Gợi ý một task mới nên tạo hôm nay dựa trên lịch sử task của project và user (nếu có)."""
    try:
        user_id = body.user_id
        project_id = body.project_id
        k = body.k or 1
        result = llm_svc.suggest_new_task_today(
            user_id=user_id, project_id=project_id, k=k
        )
        return result
    except Exception as e:
        log.exception(f"Unexpected error in /suggest_new_task_today: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@compose_router.post(
    "/generate_phases",
    summary="Gợi ý phases mới nên tạo hôm nay bằng LLM",
        description="""
Flow:
1. Nhận đặc tả dự án.
2. Sinh các phase phát triển chính.

Args:
- files: File đặc tả dự án.
- project_id: ID dự án.

Return:
- Thông tin task đã được AI sinh ra (title, description, priority, type, due_date, todos).
""",)
async def generate_phases(
    project_id: int = Form(None),
    files: list[UploadFile] = File(None),
    llm_svc: LLMService = Depends(get_llm_service),
    vector_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Tạo nhiều phase mới dưa trên đặc tả dự án từ file đính kèm."""
    try:
        project_id = project_id  # <-- nhận projectId từ client
        project_spec = ""
        if files:
            project_spec = await extract_text_from_multiple_files(files)
            # Gọi hàm để đồng bộ dữ liệu project spec
            vector_svc.sync_project_spec(project_id, project_spec)
            if len(project_spec) > MAX_SPEC_LENGTH:
                project_spec = project_spec[:MAX_SPEC_LENGTH]
        
        raw_result = llm_svc.generate_phases(
            project_id=project_id,
            project_spec=project_spec,
        )

        if "error" in raw_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=raw_result["error"],
            )
        if "raw" in raw_result:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="LLM response was not valid JSON during full phase generation.",
            )

        return raw_result

    except (TypeError, ValueError) as e:
        log.error(f"Input/processing error in /generate_phases: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)
        )
    except Exception as e:
        log.exception(f"Unexpected error in /generate_phases: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

@compose_router.post(
    "/generate_tasks",
    summary="Gợi ý task mới nên tạo hôm nay bằng LLM",
        description="""
Flow:
1. Nhận đặc tả dự án và danh sách người dùng.
2. Sinh các phase phát triển chính và các task tương ứng cho mỗi phase.

Args:
- files: File đặc tả dự án.
- project_id: ID dự án.

Return:
- Thông tin task đã được AI sinh ra (title, description, priority, type, due_date, todos).
""",)
async def generate_tasks(
    body: GenerateTasksRequest,
    llm_svc: LLMService = Depends(get_llm_service)
):
    """Tạo nhiều task mới dưa trên đặc tả dự án từ file đính kèm."""
    try:
        raw_result = llm_svc.generate_tasks(
            project_id=body.project_id,
            phase_content=body.phase_content.model_dump(),
            users=[u.model_dump() for u in body.users],    
        )

        if "error" in raw_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=raw_result["error"],
            )
        if "raw" in raw_result:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="LLM response was not valid JSON during full task generation.",
            )

        return raw_result

    except (TypeError, ValueError) as e:
        log.error(f"Input/processing error in /generate_tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)
        )
    except Exception as e:
        log.exception(f"Unexpected error in /generate_tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail= str(e),
        )
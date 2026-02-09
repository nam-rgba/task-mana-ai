from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# alias thể hiện tên trường trong JSON khi nhận request: requirement(JSON) -> requirement_text(thuộc tính trong class)
# Request body
class ComposeRequest(BaseModel):
    """
    Yêu cầu tạo nội dung mới dựa trên input của người dùng và project_id.
    """
    user_input: Optional[str] = Field(..., alias="user_input", description="Nội dung nhập vào từ người dùng")
    project_id: Optional[int] = Field(..., alias="project_id", description="ID của dự án")
    class Config:
        populate_by_name = True
        # extra = "allow"


class AssignRequest(BaseModel):
    """
    Yêu cầu phân công task với thông tin task, requirement và project_id.
    """
    task: Dict = Field(..., description="Thông tin chi tiết của task")
    requirement_text: Optional[str] = Field("", alias="requirement", description="Yêu cầu công việc (mặc định là rỗng)")
    project_id: Optional[int] = Field(..., alias="project_id", description="ID của dự án")
    class Config:
        populate_by_name = True
        # extra = "allow"


class DuplicateRequest(BaseModel):
    """
    Kiểm tra nhiệm vụ trùng lặp
    """
    task: Dict = Field(..., description="Thông tin chi tiết nhiệm vụ cần kiểm tra")
    project_id: Optional[int] = Field(..., alias="project_id", description="ID của dự án")
    class Config:
        populate_by_name = True
        # extra = "allow"


class EstimateSPRequest(BaseModel):
    """
    Yêu cầu ước lượng điểm SP cho task với các thông tin cơ bản.
    """
    title: Optional[str] = Field(..., description="Tiêu đề của task")
    description: Optional[str] = Field(..., description="Mô tả task")
    type: Optional[str] = Field("FEATURE", description="Loại task (mặc định là FEATURE)")
    priority: Optional[str] = Field("MEDIUM", description="Độ ưu tiên của task (mặc định là MEDIUM)")
    class Config:
        populate_by_name = True
        # extra = "allow"


class SummarizeRequest(BaseModel):
    """
    Yêu cầu tóm tắt danh sách các task, có thể sử dụng LLM.
    """
    tasks: List[Dict] = Field(..., description="Danh sách các task cần tóm tắt")
    use_llm: Optional[bool] = Field(True, alias="use_llm", description="Có sử dụng LLM để tóm tắt hay không")
    class Config:
        populate_by_name = True
        # extra = "allow"


class GenerateTaskRequest(BaseModel):
    """
    Yêu cầu sinh task mới dựa trên input người dùng, project_id và requirement.
    """
    user_input: Optional[str] = Field(..., alias="user_input", description="Mô tả nhiệm vụ")
    project_id: Optional[int] = Field(..., alias="project_id", description="ID của dự án")
    requirement_text: Optional[str] = Field("", alias="requirement", description="Yêu cầu công việc (mặc định là rỗng)")
    class Config:
        populate_by_name = True
        # extra = "allow"


class ChatRequest(BaseModel):
    """
    Yêu cầu chat với hệ thống, gồm câu hỏi và session_id.
    """
    question: str = Field(..., description="Câu hỏi của người dùng")
    session_id: str = Field("default", description="ID phiên chat (mặc định là 'default')")


class UpdateTaskRequest(BaseModel):
    """
    Yêu cầu cập nhật thông tin task (đồng bộ với metadata vector_store).
    """
    id: int = Field(..., description="Task ID phải có")
    title: Optional[str] = Field(None, alias="title", description="Tiêu đề task")
    description: Optional[str] = Field(None, alias="description", description="Mô tả task")
    status: Optional[str] = Field(None, alias="status", description="Trạng thái task")
    priority: Optional[str] = Field(None, alias="priority", description="Độ ưu tiên")
    project_id: Optional[int] = Field(None, alias="projectId", description="ID dự án")
    assignee_id: Optional[int] = Field(None, alias="assigneeId", description="ID người thực hiện")
    reviewer_id: Optional[int] = Field(None, alias="reviewerId", description="ID người review")
    due_date: Optional[str] = Field(None, alias="dueDate", description="Ngày hết hạn")
    estimate_effort: Optional[int] = Field(None, alias="estimateEffort", description="Ước lượng effort")
    actual_effort: Optional[int] = Field(None, alias="actualEffort", description="Effort thực tế")
    created_at: Optional[str] = Field(None, alias="createdAt", description="Thời gian tạo")
    updated_at: Optional[str] = Field(None, alias="updatedAt", description="Thời gian cập nhật")
    completed_at: Optional[str] = Field(None, alias="completedAt", description="Thời gian hoàn thành")
    qc_review_status: Optional[str] = Field(None, alias="qcReviewStatus", description="Trạng thái QC")
    type: Optional[str] = Field(None, alias="type", description="Loại task")  # task_type bên vector_store là type ở đây

    class Config:
        populate_by_name = True
        extra = "allow"


class UpdateUserRequest(BaseModel):
    """
    Yêu cầu cập nhật thông tin người dùng.
    """
    id: int = Field(..., description="User ID phải có")
    email: Optional[str] = Field(None, alias="email", description="Email người dùng")
    name: Optional[str] = Field(None, alias="name", description="Tên người dùng")
    position: Optional[str] = Field(None, alias="position", description="Chức vụ")
    year_of_experience: Optional[int] = Field(None, alias="yearOfExperience", description="Số năm kinh nghiệm")
    created_at: Optional[str] = Field( None, alias="createdAt", description="Thời gian tạo")
    updated_at: Optional[str] = Field(None, alias="updatedAt", description="Thời gian cập nhật")
    # is_deleted: Optional[bool] = Field(False, alias="isDeleted", description="Người dùng đã bị xóa hay chưa")

    class Config:
        populate_by_name = True
        extra = "allow"


class UpdateProjectRequest(BaseModel):
    """
    Yêu cầu cập nhật thông tin dự án.
    """
    id: int = Field(..., description="Project ID phải có")
    name: Optional[str] = Field(None, alias="name", description="Tên dự án")
    description: Optional[str] = Field(None, alias="description", description="Mô tả dự án")
    team_id: Optional[int] = Field(None, alias="teamId", description="ID team")
    lead_id: Optional[int] = Field(None, alias="leadId", description="ID trưởng dự án")
    type: Optional[str] = Field(None, alias="type", description="Loại dự án") # project_type bên vector_store là type ở đây
    status: Optional[str] = Field(None, alias="status", description="Trạng thái dự án")
    visibility: Optional[str] = Field(None, alias="visibility", description="Chế độ hiển thị dự án")
    start_date: Optional[str] = Field(None, alias="startDate", description="Ngày bắt đầu")
    end_date: Optional[str] = Field(None, alias="endDate", description="Ngày kết thúc")
    created_at: Optional[str] = Field( None, alias="createdAt", description="Thời gian tạo")
    updated_at: Optional[str] = Field(None, alias="updatedAt", description="Thời gian cập nhật")

    class Config:
        populate_by_name = True
        extra = "allow"

class UpdateTeamRequest(BaseModel):
    """
    Yêu cầu cập nhật thông tin team (khớp với metadata vector_store).
    """
    id: int = Field(..., description="Team ID phải có")
    name: Optional[str] = Field(None, alias="name", description="Tên team")
    members: Optional[List[Dict[str, Any]]] = Field(
        None, alias="members", description="Danh sách thành viên [{'user_id': int, 'role': str}, ...]"
    )
    # Các trường thời gian nếu cần
    created_at: Optional[str] = Field(None, alias="createdAt", description="Thời gian tạo")
    updated_at: Optional[str] = Field(None, alias="updatedAt", description="Thời gian cập nhật")

    class Config:
        populate_by_name = True
        extra = "allow"
        
class SuggestTasksRequest(BaseModel):
    """
    Yêu cầu gợi ý các task cho user trong project.
    """
    project_id: int = Field(..., description="ID của project là bắt buộc")
    user_id: Optional[int] = Field(None, alias="user_id", description="ID của người dùng")
    k: Optional[int] = Field(3, alias="k", description="Số lượng task gợi ý")
    class Config:
        populate_by_name = True
        # extra = "allow"

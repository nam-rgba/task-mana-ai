from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

#alias thể hiện tên trường trong JSON khi nhận request: requirement(JSON) -> requirement_text(thuộc tính trong class)
#Request body
class ComposeRequest(BaseModel):
    user_input: Optional[str] = Field("", alias="user_input")
    project_id: Optional[str] = Field(None, alias="project_id")
    class Config:
        allow_population_by_field_name = True
        extra = "allow"

class AssignRequest(BaseModel):
    task: Dict = Field(default_factory=dict)
    requirement_text: Optional[str] = Field("", alias="requirement")
    project_id: Optional[str] = Field(None, alias="project_id")
    class Config:
        allow_population_by_field_name = True
        extra = "allow"

class DuplicateRequest(BaseModel):
    task: Dict = Field(default_factory=dict)
    project_id: Optional[str] = Field(None, alias="project_id")
    class Config:
        allow_population_by_field_name = True
        extra = "allow"

class EstimateSPRequest(BaseModel):
    title: Optional[str] = ""
    description: Optional[str] = ""
    class Config:
        extra = "allow"

class SummarizeRequest(BaseModel):
    tasks: List[Dict]
    use_llm: Optional[bool] = Field(True, alias="use_llm")
    class Config:
        allow_population_by_field_name = True
        extra = "allow"

class GenerateTaskRequest(BaseModel):
    user_input: Optional[str] = Field("", alias="user_input")
    project_id: Optional[str] = Field(None, alias="project_id")
    requirement_text: Optional[str] = Field("", alias="requirement")
    class Config:
        allow_population_by_field_name = True
        extra = "allow"

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"

    
class UpdateTaskRequest(BaseModel):
    id: str = Field(..., description="User ID phải có")
    title: Optional[str] = Field(None, alias="title")
    description: Optional[str] = Field(None, alias="description")
    status: Optional[str] = Field(None, alias="status")
    due_date: Optional[str] = Field(None, alias="dueDate")
    estimate_effort: Optional[int] = Field(None, alias="estimateEffort")
    actual_effort: Optional[int] = Field(None, alias="actualEffort")
    implementor_id: Optional[int] = Field(None, alias="implementorId")
    reviewer_id: Optional[int] = Field(None, alias="reviewerId")
    project_id: Optional[str] = Field(None, alias="projectId")
    parent_task_id: Optional[str] = Field(None, alias="parentTaskId")
    priority: Optional[str] = Field(None, alias="priority")
    completed_percent: Optional[int] = Field(None, alias="completedPercent")
    completed_at: Optional[str] = Field(None, alias="completedAt")
    file_urls: Optional[List[str]] = Field(default_factory=list, alias="fileUrls")
    class Config:
        allow_population_by_field_name = True
        extra = "allow"

class UpdateUserRequest(BaseModel):
    id: str = Field(..., description="User ID phải có")
    email: Optional[str] = Field(None, alias="email")
    password: Optional[str] = Field(None, alias="password")
    name: Optional[str] = Field(None, alias="name")
    avatar: Optional[str] = Field(None, alias="avatar")
    position: Optional[str] = Field(None, alias="position")
    year_of_experience: Optional[int] = Field(None, alias="yearOfExperience")
    class Config:
        allow_population_by_field_name = True
        extra = "allow"

class UpdateProjectRequest(BaseModel):
    id: str = Field(..., description="Project ID phải có")
    key: Optional[str] = Field(None, alias="key")
    name: Optional[str] = Field(None, alias="name")
    description: Optional[str] = Field(None, alias="description")
    teamId: Optional[str] = Field(None, alias="teamId")
    leadId: Optional[str] = Field(None, alias="leadId")
    type: Optional[str] = Field(None, alias="type")
    status: Optional[str] = Field(None, alias="status")
    category: Optional[str] = Field(None, alias="category")
    avatarUrl: Optional[str] = Field(None, alias="avatarUrl")
    color: Optional[str] = Field(None, alias="color")
    startDate: Optional[str] = Field(None, alias="startDate")
    endDate: Optional[str] = Field(None, alias="endDate")
    visibility: Optional[str] = Field(None, alias="visibility")
    permissionSchemeId: Optional[str] = Field(None, alias="permissionSchemeId")
    workflowSchemeId: Optional[str] = Field(None, alias="workflowSchemeId")
    issueTypeSchemeId: Optional[str] = Field(None, alias="issueTypeSchemeId")
    defaultAssigneeType: Optional[str] = Field(None, alias="defaultAssigneeType")
    defaultAssigneeId: Optional[str] = Field(None, alias="defaultAssigneeId")
    lastIssueNumber: Optional[int] = Field(None, alias="lastIssueNumber")
    settings: Optional[dict] = Field(default_factory=dict, alias="settings")
    members: Optional[list] = Field(default_factory=list, alias="members")
    class Config:
        allow_population_by_field_name = True
        extra = "allow"


class SuggestTasksRequest(BaseModel):
    project_id: str = Field(..., description="ID của project là bắt buộc") 
    user_id: Optional[str] = Field(None, alias="user_id")
    k: Optional[int] = Field(3, alias="k")
    class Config:
        allow_population_by_field_name = True
        extra = "allow"

    
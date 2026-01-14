from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# DÙng cho chức năng tạo subtask và compose task
class SubtaskOut(BaseModel):
    title: str
    description: str = ""
    priority: str = "MEDIUM"

class ComposeOut(BaseModel):
    title: str
    description: str = ""
    priority: str = "MEDIUM"
    type: str = "FEATURE"
    due_date: Optional[str] = None



# Dùng cho chức năng duplicate task
class DuplicateDoc(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any]  

class DuplicateTaskOut(BaseModel):
    duplicates: List[DuplicateDoc]
    nearest_tasks: Optional[List[Dict[str, Any]]] = []



# Dùng cho chức năng assign task
class AssigneeUserOut(BaseModel):
    id: str
    email: str
    name: str
    position: str

class AssignOut(BaseModel):
    assignee: AssigneeUserOut
    reason: str


# Dùng cho chức năng Estimate Story Point 
class EstimateOut(BaseModel):
    model_estimate: float = Field(..., description="Giá trị Story Point thô được dự đoán bởi mô hình.")
    suggested_story_point: str = Field(..., description="Gợi ý Story Point theo chuẩn Planning Poker (ví dụ: '5' hoặc '5 hoặc 8').")
    

# Dùng cho chức năng Sprint Summary 
class SprintMetrics(BaseModel):
    velocity: Optional[float]
    burndown_status: Optional[str]
    average_lead_time: Optional[float]
    average_cycle_time: Optional[float]

class SprintSummary(BaseModel):
    summary: str
    metrics: SprintMetrics
    recommendations: Optional[str] = ""


# Dùng cho chức năng gợi ý task
class SuggestTasksOut(BaseModel):
    suggested_tasks: List[Dict[str, Any]]
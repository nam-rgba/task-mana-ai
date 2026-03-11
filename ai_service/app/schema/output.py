from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# Dùng cho chức năng assign task (Optional cho trường hợp không có assignee phù hợp)
class AssigneeUserOut(BaseModel):
    id: Optional[int]
    email: Optional[str]
    name: Optional[str]
    position: Optional[str]

class AssignOut(BaseModel):
    id: Optional[int]
    email: Optional[str]
    name: Optional[str]
    position: Optional[str]
    reason: Optional[str]

#Model tasks out
class ComposeOut(BaseModel):
    title: str
    description: str = ""
    priority: str = "MEDIUM"
    type: str = "FEATURE"
    story_point: Optional[float] = None
    start_date: Optional[str] = None
    due_date: Optional[str] = None
    assignee: Optional[int] = None
    todos: Optional[List[str]] = []
    

# Danh sách tasks out
class ListComposeOut(BaseModel):
    tasks: List[ComposeOut]
  
# Model schedule out (phase) 
class PhaseOut(BaseModel):
    title: str
    description: str
    phase_start: str
    phase_end: str
    

# Danh sách schedule out
class ListPhaseOut(BaseModel):
    phases: List[PhaseOut]


# Dùng cho chức năng duplicate task
class DuplicateDoc(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any]  

class DuplicateTaskOut(BaseModel):
    duplicates: List[DuplicateDoc]
    nearest_tasks: Optional[List[Dict[str, Any]]] = []


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
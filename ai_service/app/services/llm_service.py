import re
import json
import logging
import numpy as np  
import pandas as pd
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from langdetect import detect 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import  RunnablePassthrough, RunnableLambda, RunnableParallel
from app.services.vector_store import VectorStoreService
from app.schema.output import *
from app.schema.input import *
from app.services.models_loader import ModelsLoader
from app.services.xgb_service import get_xgb_service

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

device = "cpu"
print("Current device:", device)



class LLMService:
    def __init__(self, vector_store: Optional[VectorStoreService] = None, custom_api_key: str = None, custom_provider: str = "groq" , custom_model_name: Optional[str] = None):
        # Lưu trữ custom key & provider
        self.custom_api_key = custom_api_key
        self.custom_provider = custom_provider
        self.custom_model_name = custom_model_name

        # Load models
        self.llm = ModelsLoader.llm(api_key=self.custom_api_key, provider=self.custom_provider, model_name=self.custom_model_name)
        self.embed_model = ModelsLoader.embeddings()
        self.xgb_model = ModelsLoader.xgb_model()
        self.scaler = ModelsLoader.scaler()
        self.vector_store = vector_store or VectorStoreService()
        self.xgb_service = get_xgb_service() # Khởi tạo để gọi hàm dự đoán story point
        # LLM enabled flag
        self.enabled = self.llm is not None
        
    # HÀM TẠO TASK (COMPOSITION) - DÙNG RETRIEVER LẤY NGỮ CẢNH
    def compose_with_llm(
    self,
    user_input: str,
    project_id: Optional[int] = None,
    attach_context: Optional[str] = None
) -> ComposeOut:
        """
        Tạo task mới dựa trên project spec (RAG)
        """

        if not project_id:
            raise ValueError("Thiếu project_id để tạo task.")

        lang_name = _detect_language(user_input)
        today = date.today().isoformat()
        
        #Lấy thông tin project
        project_docs = self.vector_store.get_project_by_id(project_id)
        project_info = ""

        if project_docs:
            project_info = project_docs[0].page_content.strip()

        # Lấy project spec từ vector store
        project_spec = self.vector_store.retrieve_project_specs_by_query(
            project_id=project_id,
            query=user_input,
            k=3
        )

        context_parts = []

        # Thêm project info
        if project_info:
            context_parts.append(f"Mô tả:\n{project_info}")

        # Thêm spec từ vector search
        if project_spec:
            specs = "\n".join(
                f"- {doc['content'][:400]}" for doc in project_spec
            )
            context_parts.append(f"Đặc tả:\n{specs}")

        # Gộp lại
        project_context = "\n\n".join(context_parts) if context_parts else ""

        template = """
    Bạn là AI quản lý dự án phần mềm.

    Đặc tả dự án:
    {project_context}

    Mô tả nhiệm vụ:
    {user_input}

    Ngữ cảnh bổ sung:
    {attach_context}

    Hãy tạo nội dung nhiệm vụ phù hợp.
    Yêu cầu:
    - Viết bằng {lang}
    - Không markdown. Không giải thích. Trả JSON hợp lệ
    - priority thuộc: URGENT,HIGH,MEDIUM,LOW
    - type thuộc: FEATURE,BUG,IMPROVEMENT,RESEARCH,DOCUMENTATION,TESTING,DEPLOYMENT,ENHANCEMENT,MAINTENANCE,OTHER
    - thời gian hợp lý so với hôm nay ({date})

    Cấu trúc task:
    {{
    "title": {user_input},
    "description": "mô tả chi tiết",
    "priority": "MEDIUM",
    "type": "FEATURE",
    "story_point": number,
    "start_date": "YYYY-MM-DD",
    "due_date": "YYYY-MM-DD",
    "todos": ["step 1","step 2"]
    }}
    """

        prompt = PromptTemplate(
            template=template,
            input_variables=[
                "project_context",
                "user_input",
                "attach_context",
                "lang",
                "date",
            ],
        )

        parser = PydanticOutputParser(pydantic_object=ComposeOut)
        chain = prompt | self.llm

        try:
            ai_msg = chain.invoke({
                "project_context": project_context,
                "user_input": user_input,
                "attach_context": attach_context or "Không có",
                "lang": lang_name,
                "date": today,
            })
            response = parser.invoke(ai_msg)
            usage = ai_msg.response_metadata.get("token_usage", {})
            log.info("Tạo task thành công với LLM Compose.")
            return {
                "result": response,
                "usage": usage
            }

        except Exception as e:
            log.exception("LLM compose failed")
            return {"error": str(e)}

    # HÀM GỢI Ý GÁN NGƯỜI THỰC HIỆN (ASSIGNMENT) 
    def assign_candidate(
        self,
        task: dict,
        project_id: int,
        requirement_text: Optional[str] = None
    ) -> Dict[str, Any]:

        lang_name = _detect_language(requirement_text or task.get("title", ""))

        if not project_id:
            log.warning("Task không có project_id.")
            return {"error": "Missing project_id"}

        try:

            # Lấy danh sách user trong project
            if requirement_text:
                users = self.vector_store.retrieve_users_by_query(
                    task_text=requirement_text,
                    project_id=project_id,
                    k=6
                ) or []
            else:
                users = self.vector_store.retrieve_users_by_project(
                    project_id=project_id,
                    k=6
                ) or []

            available_users = []

            for user in users:

                user_meta = user.get("metadata", {})
                user_id = user_meta.get("user_id")

                # Lấy task history
                tasks = self.vector_store.retrieve_tasks_by_user(
                    user_id=user_id,
                    project_id=project_id,
                    k=10
                )

                pending = 0
                processing = 0
                done = 0

                for t in tasks:
                    status = (t.get("metadata", {}).get("status") or "").upper()

                    if status == "PENDING":
                        pending += 1
                    elif status == "PROCESSING":
                        processing += 1
                    elif status == "DONE":
                        done += 1

                available_users.append({
                    "id": user_id,
                    "email": user_meta.get("email"),
                    "name": user_meta.get("name"),
                    "position": user_meta.get("position"),
                    "experience_years": user_meta.get("experience_years"),
                    "pending_tasks": pending,
                    "processing_tasks": processing,
                    "done_tasks": done
                })

            log.info(f"Prepared {len(available_users)} available users")

            template = """
                Bạn là AI phân công task.
                Chọn 1 người phù hợp nhất dựa trên:Position phù hợp task. Ít pending + processing tasks. Nhiều done tasks. Kinh nghiệm
                Trả JSON duy nhất:
                {{
                "id": -1 nếu không phù hợp,
                "email": "",
                "name": "",
                "position": "",
                "reason": "giải thích ngắn"
                }}

                task:
                {task}

                available_users:
                {users}
                """

            prompt = PromptTemplate(
                template=template,
                input_variables=["task", "users"]
            )

            parser = PydanticOutputParser(pydantic_object=AssignOut)
            chain = prompt | self.llm

            ai_msg = chain.invoke({
                "task": json.dumps(task, ensure_ascii=False),
                "users": json.dumps(available_users, ensure_ascii=False)
            })
            assignment = parser.invoke(ai_msg)
            usage = ai_msg.response_metadata.get("token_usage", {})

            log.info(f"Assignment successful: {assignment.name}")

            return {
                "assignment": assignment.model_dump(),
                "available_users": available_users,
                "usage": usage
            }

        except Exception as e:
            log.exception("Assignment failed")
            return {"error": str(e)}
    
    # HÀM DUPLICATE FINDER
    def find_duplicate_tasks(
        self,
        task: dict,
        threshold: float = 0.2,
        k: int = 3,
        project_id: Optional[int] = None
    ) -> DuplicateTaskOut:
        """
        Tìm các nhiệm vụ trùng lặp dựa trên embedding similarity/distance.
        """

        try:
            # Build query text từ task
            title = task.get("title", "")
            description = task.get("description", "")
            tags_text = " ".join(task.get("tags", []))
            priority = task.get("priority", "")
            assignee_name = (task.get("assignee") or {}).get("name", "")

            query_text = " ".join([
                title,
                description,
                tags_text,
                priority,
                assignee_name
            ])

            log.info("Đang kiểm tra duplicate task...")

            retrieved = self.vector_store.retrieve_tasks_with_scores(
                query=query_text,
                k=k,
                project_id=project_id
            )

            duplicates = []

            for doc in retrieved:
                score = None

                if isinstance(doc, dict):
                    score = doc.get("score") or doc.get("similarity") or doc.get("distance")

                try:
                    if score is not None and score <= threshold:
                        duplicates.append({
                            "content": doc.get("content"),
                            "score": score,
                            "metadata": doc.get("metadata"),
                        })
                except Exception:
                    continue

            return DuplicateTaskOut(
                duplicates=[DuplicateDoc(**d) for d in duplicates],
                nearest_tasks=retrieved
            )

        except Exception as e:
            log.exception("Tìm kiếm trùng lặp thất bại")
            raise RuntimeError(f"Duplicate search failed: {e}")
        
    # DỰ ĐOÁN STORY POINT (GỌI XGB_SERVICE)
    def predict_story_point(self, title: str, desc:str, type_val: str = "FEATURE", priority_val: str = "MEDIUM") -> float:
        """
        Dự đoán story point cho task dựa trên text, type, priority.
        """
        xgb_service = get_xgb_service()
        return xgb_service.predict_story_point(title, desc, type_val, priority_val)

    def suggest_story_point(self, value: float) -> str:
        """
        Gợi ý story point gần nhất theo planning poker.
        """
        xgb_service = get_xgb_service()
        return xgb_service.suggest_story_point(value)

   # PIPELINE SINH TASK
    def generate_task(
        self,
        user_input: str,
        project_id: Optional[int] = None,
        requirement_text: Optional[str] = None,
        attach_context: Optional[str] = None,
        duplicate_threshold: float = 0.25,
        duplicate_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Pipeline hoàn chỉnh để sinh task, từ input của người dùng đến các gợi ý chi tiết.
        Sử dụng LangChain Expression Language (LCEL) để kết nối các bước.
        """

        if not project_id:
            log.warning("generate_task missing project_id.")
            return {"error": "Missing project_id for task generation."}
        
        # ĐỊnh nghĩa các runnable 

        # 1) Tạo task từ input của người dùng
        def _invoke_compose(x):
            res = self.compose_with_llm(
                user_input=x["user_input"],
                project_id=x.get("project_id"),
                attach_context=x.get("attach_context")
            )
            if "error" in res:
                return res
            task_dict = res["result"].model_dump() if hasattr(res["result"], "model_dump") else res["result"]
            task_dict["_llm_usage"] = res.get("usage", {})
            return task_dict

        compose_chain = RunnableLambda(_invoke_compose)

        # 2) Dự đoán Story Point
        story_point_chain = RunnableLambda(
            lambda x: self.predict_story_point(
                title=x["composed_task"]["title"],
                desc=x["composed_task"]["description"],
                type_val=x["composed_task"].get("type", "FEATURE"),
                priority_val=x["composed_task"].get("priority", "MEDIUM")
            )
        )

        # 3) Gợi ý Story Point
        sp_suggest_chain = RunnableLambda(
            lambda x: (
                self.suggest_story_point(float(x["estimated_story_point"]))
                if x.get("estimated_story_point") is not None
                else None
            )
        )
        
        # 4) Tìm kiếm các task trùng lặp
        find_duplicates_chain = RunnableLambda(
            lambda x: self.find_duplicate_tasks(
                task=x.get("composed_task"),
                threshold=duplicate_threshold,
                k=duplicate_k,
                project_id=x.get("project_id")
            ).model_dump() # Sử dụng .dict() nếu DuplicateTaskOut là Pydantic model
        )


        # 5. Gợi ý gán người thực hiện
        assign_chain = RunnableLambda(
            lambda x: self.assign_candidate(
                task=x["composed_task"],
                project_id=x.get("project_id"),
                requirement_text=x.get("requirement_text")
            )
        )

        # --- Xây dựng PIPELINE chính ---
        # Sử dụng RunnablePassthrough và RunnableMap (còn gọi là dictionary comprehension)
        # để truyền dữ liệu qua các bước và thêm kết quả mới vào.
        
        full_chain = (
            # Bước 1: Bắt đầu với input ban đầu và tạo task - đầu ra sẽ có thêm "composed_task" và truyền vào bước sau
            # Output: {"user_input", "project_id", "requirement_text", "composed_task"}
            RunnablePassthrough.assign(composed_task=compose_chain)
            
            # Bước 2: Dựa vào task đã tạo, dự đoán story point
            # Output: thêm key: "estimated_story_point"
            .assign(estimated_story_point=story_point_chain)
            
            # Bước 3: Dựa vào story point đã dự đoán, gợi ý giá trị planning poker
            # Output: thêm key: "story_point_suggestion"
            .assign(story_point_suggestion=sp_suggest_chain)
            
            # Bước 4: Tìm các task trùng lặp
            # Output: thêm key: "duplicates"
            .assign(duplicates=find_duplicates_chain)

            # Bước 5: Gợi ý gán người thực hiện
            # Output: thêm key: "assignment" --> Ta gọi tuần tự để tránh call API LLM liên tục
            .assign(assignment=assign_chain)


        )

        # --- Chạy PIPELINE ---
        input_data = {
            "user_input": user_input,
            "project_id": project_id,
            "attach_context": attach_context,
            "requirement_text": requirement_text,
        }

        # result sẽ là một dictionary chứa kết quả từ tất cả các bước
        result = full_chain.invoke(input_data)


        return {
            "user_input": {
                "user_input": user_input,
                "project_id": project_id,
                "attach_context": attach_context,
                "requirement_text": requirement_text
            },
            "composed_task": result.get("composed_task"),
            "estimated_story_point": result.get("estimated_story_point"),
            "story_point_suggestion": result.get("story_point_suggestion"),
            "duplicates": result.get("duplicates"),
            "assignment": result.get("assignment"),
            "usage": result.get("composed_task", {}).pop("_llm_usage", {}) if isinstance(result.get("composed_task"), dict) else {}
        }
    
    # GỢI Ý TASK HÔM NAY (THỦ CÔNG)
    def suggest_tasks_for_today(
            self,
            user_id: Optional[int] = None,
            project_id: Optional[int] = None, 
            k: int = 5) -> Dict[str, Any]:
        """
        Gợi ý các task cần làm trong ngày hôm nay dựa trên:
        - Priority và deadline gần nhất.
        - Status: (PENDING, PROCESSING).
        - Assignee là người hiện tại (nếu có).
        """
        try:
            if not project_id:
                raise ValueError("project_id is required for suggesting tasks.")
            # Lấy tasks từ vector store
            tasks = self.vector_store.retrieve_tasks_by_project(
                project_id=project_id,
                k=50  # Lấy nhiều hơn để lọc
            ) or []
            log.info(f"Retrieved {len(tasks)} tasks from project_id={project_id} for suggestion.")
            # Lọc tasks theo điều kiện
            filtered_tasks = []
            today = date.today()

            for t in tasks:
                metadata = t.get("metadata", {})
                status = metadata.get("status", "").upper()

                # Chỉ lấy những task chưa hoàn thành
                if status not in ["PENDING", "PROCESSING"]:
                    continue

                # Nếu có user_id, chỉ lấy task được gán
                assignee_id = metadata.get("assignee_id")
                if user_id and assignee_id != user_id:
                    continue

                filtered_tasks.append(t)

            log.info(f"Filtered {len(filtered_tasks)} eligible tasks from {len(tasks)} total tasks")

            # Hàm tính điểm ưu tiên cho mỗi task
            def calculate_task_score(t):
                metadata = t.get("metadata", {})
                
                # 1. Priority score (4 -> 1)
                priority_map = {"URGENT": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
                priority = metadata.get("priority", "MEDIUM").upper()
                priority_score = priority_map.get(priority, 2)
                
                # 2. Deadline urgency
                due_date_str = metadata.get("due_date")
                deadline_score = 0
                if due_date_str:
                    try:
                        due_date = datetime.fromisoformat(due_date_str).date()
                        days_left = (due_date - today).days
                        
                        # Overdue tasks = highest urgency
                        if days_left < 0:
                            deadline_score = 100  # Quá hạn → ưu tiên cao nhất
                        elif days_left == 0:
                            deadline_score = 50   # Hôm nay
                        elif days_left <= 3:
                            deadline_score = 20   # 1-3 ngày tới
                        elif days_left <= 7:
                            deadline_score = 10   # Tuần này
                        else:
                            deadline_score = max(0, 5 - days_left // 7)  # Giảm dần theo tuần
                    except Exception:
                        deadline_score = 0
                
                # 3. Story points (ưu tiên task nhỏ để quick wins, chia để ngược lại)
                estimate_effort = metadata.get("estimate_effort", 5)
                try:
                    sp_score = 1 / float(estimate_effort) if estimate_effort > 0 else 0
                except:
                    sp_score = 0
                
                # 4. Status bonus (PROCESSING > PENDING)
                status = metadata.get("status", "PENDING").upper()
                status_bonus = 5 if status == "PROCESSING" else 0
                
                # Tổng điểm (nhân thêm trọng số (weighted) để thể hiện độ ưu tiên)
                total_score = (
                    priority_score * 10 +      # Priority quan trọng nhất
                    deadline_score * 5 +       # Deadline cũng rất quan trọng
                    sp_score * 2 +             # Quick wins
                    status_bonus               # Ưu tiên task đang làm dở
                )
                
                return total_score

            # Tính và Gán điểm vào metadata để debug
            for t in filtered_tasks:
                score = calculate_task_score(t)
                t["score"] = score  

            # Sắp xếp filtered_tasks (QUAN TRỌNG!)
            filtered_tasks.sort(key=calculate_task_score, reverse=True)
            
            # Lấy top k
            suggested_tasks = filtered_tasks[:k]

            return {
                "today_date": today.isoformat(),
                "suggested_tasks": suggested_tasks,
                "total_eligible": len(filtered_tasks),
                "total_scanned": len(tasks),
                "criteria": {
                    "priority": "URGENT > HIGH > MEDIUM > LOW",
                    "deadline": "Overdue > Today > This week > Later",
                    "effort": "Smaller tasks preferred",
                    "status": "PROCESSING > PENDING"
                }
            }

        except Exception as e:
            log.exception("Gợi ý task cho hôm nay thất bại")
            return {"error": str(e)}

    # GỢI Ý TASK HÔM NAY (LLM XEM LỊCH SỬ TASK RỒI ĐỀ XUẤT)
    def suggest_tasks_for_today_llm(
            self,
            project_id: int,
            user_id: Optional[int] = None,
            k: int =5
        ) -> Dict[str, Any]:
        """
        Gợi ý các task nên làm hôm nay dựa trên lịch sử task của project và user (nếu có).
        Sử dụng LLM để phân tích và đề xuất.
        """
        try:
            if not self.enabled or not self.llm:
                raise RuntimeError("LLM service is not enabled or not set up.")

            if not project_id:
                raise ValueError("project_id is required for suggesting tasks.")

            # Lấy thông tin project
            project_content = self.vector_store.get_project_by_id(project_id)
            if project_content:
                project_info = project_content[0].page_content
            else:
                project_info = "Không có"

            # Lấy lịch sử task của project
            project_tasks = self.vector_store.retrieve_tasks_by_project(
                project_id=project_id,
                k=50
            ) or []

            # Lấy lịch sử task của user (nếu có)
            user_tasks = []
            if user_id:
                user_tasks = self.vector_store.retrieve_tasks_by_user(
                    user_id=user_id,
                    project_id=project_id,
                    k=50
                ) or []

            # Kết hợp dữ liệu
            combined_tasks = project_tasks + user_tasks

            # Tạo prompt cho LLM
            template = """
                Bạn là AI trợ lý quản lý dự án. Dựa trên lịch sử nhiệm vụ dưới đây, hãy đề xuất {k} nhiệm vụ quan trọng nhất mà người dùng nên tập trung hoàn thành trong ngày hôm nay.
                Thông tin dự án:
                {project_info}
                Lịch sử nhiệm vụ:
                {tasks_history}
                Yêu cầu:
                - Chọn nhiệm vụ có độ ưu tiên cao, deadline gần, và phù hợp với vai trò người dùng (nếu có).
                - Trả về danh sách nhiệm vụ dưới dạng JSON với cấu trúc:
                {{
                    "suggested_tasks": [
                        {{
                            "task_id": int (khớp với task_id trong lịch sử để dễ tra cứu),
                            "title": "string",
                            "priority": "string",
                            "due_date": "YYYY-MM-DD",
                            "reason": "string (lý do chọn nhiệm vụ này)"
                        }},
                        ...
                    ]
                }}
            """
            tasks_history_text = format_context_included_taskid(combined_tasks)

            prompt = PromptTemplate(
                template=template,
                input_variables=["project_info", "tasks_history", "k"]
            )

            parser = PydanticOutputParser(pydantic_object=SuggestTasksOut)
            chain = prompt | self.llm

            ai_msg = chain.invoke({
                "project_info": project_info,
                "tasks_history": tasks_history_text,
                "k": k
            })
            response = parser.invoke(ai_msg)
            usage = ai_msg.response_metadata.get("token_usage", {})

            result = response.model_dump()
            result["usage"] = usage
            return result

        except Exception as e:
            log.exception("Gợi ý task hôm nay bằng LLM thất bại")
            return {"error": str(e)}

    # GỢI Ý TASK MỚI HÔM NAY (DÙNG LLM TỰ ĐỘNG SINH TASK)
    def suggest_new_task_today(
            self,
            project_id: int,
            user_id: Optional[int] = None,
            k: int = 1 # số task mới cần gợi ý
        ) -> Dict[str, Any]:
        """
        Gợi ý các task mới nên tạo hôm nay dựa trên lịch sử task của project và user (nếu có).
        Sử dụng LLM để phân tích pattern và đề xuất task mới.
        """
        try:
            if not self.enabled or not self.llm:
                raise RuntimeError("LLM service is not enabled or not set up.")
            
            if not project_id:
                raise ValueError("project_id is required for suggesting new tasks.")

            # Lấy thông tin project
            project_content = self.vector_store.get_project_by_id(project_id)
            if project_content:
                project_info = project_content[0].page_content
            else:
                project_info = "Không có"
                
            # 1. Lấy lịch sử task của project
            project_tasks = self.vector_store.retrieve_tasks_by_project(
                project_id=project_id,
                k=50
            ) or []

            # Lấy thông tin project
            project_content = self.vector_store.get_project_by_id(project_id)
            # if project_content:
            #     project_info
            
            log.info(f"Retrieved {len(project_tasks)} project tasks for analysis")

            # 2. Lấy lịch sử task của user (nếu có)
            user_tasks = []
            user_context = "Không có thông tin người dùng cụ thể."
            if user_id:
                user_tasks = self.vector_store.retrieve_tasks_by_user(
                    user_id=user_id,
                    project_id=project_id,
                    k=10
                ) or []
                log.info(f"Retrieved {len(user_tasks)} user tasks for user_id={user_id}")
                
                # Lấy thông tin user
                users = self.vector_store.retrieve_users_by_project(
                    project_id=project_id,
                    k=10
                ) or []
                
                user_info = next((u for u in users if u.get("metadata", {}).get("user_id") == user_id), None)
                if user_info:
                    metadata = user_info.get("metadata", {})
                    user_context = f"Người dùng: {metadata.get('name', 'N/A')} - Vị trí: {metadata.get('position', 'N/A')}"

            # 3. Phân tích pattern từ lịch sử
            today = date.today()
            
            # Thống kê các loại task
            task_types = {}
            priorities = {}
            done_tasks = []
            todo_tasks = []
            
            for t in project_tasks:
                metadata = t.get("metadata", {})
                
                # Đếm type
                task_type = metadata.get("task_type", "OTHER")
                task_types[task_type] = task_types.get(task_type, 0) + 1
                
                # Đếm priority
                priority = metadata.get("priority", "MEDIUM")
                priorities[priority] = priorities.get(priority, 0) + 1
                
                # Phân loại theo status
                status = metadata.get("status", "").upper()
                if status == "DONE":
                    done_tasks.append(t)
                elif status in ["PENDING", "PROCESSING"]:
                    todo_tasks.append(t)
            
            # 4. Build context cho LLM
            project_context = format_context(project_tasks[:10])  # Top 10 tasks
            
            stats_text = f"""
            Thống kê dự án:
            - Tổng số task: {len(project_tasks)}
            - Task hoàn thành: {len(done_tasks)}
            - Task đang làm/chưa làm: {len(todo_tasks)}
            - Loại task phổ biến: {', '.join([f"{k}({v})" for k, v in sorted(task_types.items(), key=lambda x: x[1], reverse=True)[:3]])}
            - Mức độ ưu tiên: {', '.join([f"{k}({v})" for k, v in sorted(priorities.items(), key=lambda x: x[1], reverse=True)])} 
            """
            # sau khi format "BUG(12), FEATURE(8), DOC(3)"
            # sau khi format "HIGH(10), MEDIUM(5), LOW(3), URGENT(2)"

            # 5. Tạo prompt cho LLM
            template = """
            Bạn là AI trợ lý quản lý dự án Agile. Nhiệm vụ của bạn là phân tích lịch sử nhiệm vụ và ĐỀ XUẤT {k} NHIỆM VỤ MỚI nên được tạo trong ngày hôm nay ({today}).

            **Thông tin dự án**
            {project_info}

            **Ngữ cảnh dự án:**
            {project_context}

            **Thống kê:**
            {stats}

            **Thông tin người dùng:**
            {user_context}

            **Yêu cầu phân tích:**
            1. Xác định các gaps (khoảng trống) trong backlog hiện tại
            2. Dựa vào pattern của các task đã hoàn thành, đề xuất task tiếp theo hợp lý
            3. Cân nhắc dependencies và workflow tự nhiên của dự án
            4. Ưu tiên task có giá trị cao, rủi ro thấp

            **Lưu ý:**
            - KHÔNG đề xuất task trùng lặp với các task hiện có
            - Task phải phù hợp với ngữ cảnh dự án
            - Mỗi task trả về là 1 JSON riêng biệt, KHÔNG phải array

            Hãy trả về {k} task suggestions. Mỗi task là một JSON object riêng biệt theo format:

            {{
                "title": "string (ngắn gọn, rõ ràng)",
                "description": "string (mô tả chi tiết theo dạng Given-When-Then tự nhiên)",
                "priority": "URGENT | HIGH | MEDIUM | LOW",
                "type": "FEATURE | BUG | IMPROVEMENT | RESEARCH | DOCUMENTATION | TESTING | DEPLOYMENT | ENHANCEMENT | MAINTENANCE | OTHER",
                "due_date": "YYYY-MM-DD" (tính từ hôm nay {today}, nên trong vòng 3-7 ngày)
                "todos": [ "string (các bước cụ thể để hoàn thành task, nếu có)" ]
            }}

            Chỉ trả về JSON thuần, KHÔNG kèm markdown, giải thích hay text khác.
            """

            prompt = PromptTemplate(
                template=template,
                input_variables=["k", "today", "project_info","project_context", "stats", "user_context"]
            )

            # 6. Tạo chain để sinh nhiều tasks
            suggested_new_tasks = []
            
            total_usage = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

            for i in range(k):
                try:
                    parser = PydanticOutputParser(pydantic_object=ComposeOut)
                    chain = prompt | self.llm
                    
                    ai_msg = chain.invoke({
                        "k": k,
                        "today": today.isoformat(),
                        "project_info": project_info,
                        "project_context": project_context,
                        "stats": stats_text,
                        "user_context": user_context
                    })
                    response = parser.invoke(ai_msg)
                    usage = ai_msg.response_metadata.get("token_usage", {})
                    
                    # Accumulate usage
                    if usage:
                        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                        total_usage["total_tokens"] += usage.get("total_tokens", 0)
                    
                    # ComposeOut trả về Pydantic model
                    task_dict = response.model_dump() if hasattr(response, 'model_dump') else response
                    suggested_new_tasks.append(task_dict)
                    
                    log.info(f"Generated new task suggestion {i+1}/{k}: {task_dict.get('title', 'N/A')}")
                    
                except Exception as e:
                    log.warning(f"Failed to generate task {i+1}: {e}")
                    continue

            return {
                "suggested_new_tasks": suggested_new_tasks,
                "total_generated": len(suggested_new_tasks),
                "analysis": {
                    "project_tasks_analyzed": len(project_tasks),
                    "user_tasks_analyzed": len(user_tasks),
                    "done_tasks": len(done_tasks),
                    "pending_tasks": len(todo_tasks),
                    "top_task_types": list(sorted(task_types.items(), key=lambda x: x[1], reverse=True)[:3])
                },
                "generation_date": today.isoformat(),
                "usage": total_usage
            }

        except Exception as e:
            log.exception("Gợi ý task mới hôm nay thất bại")
            return {"error": str(e)}

    # SINH NHIỀU PHASES
    def generate_phases(
        self,
        project_id: int,
        project_spec: str,
    ) -> ListPhaseOut:
        """
        Sinh các phase chính của dự án.
        """

        if not project_spec:
            return {"error": "Thiếu đặc tả dự án (project_spec)."}

        lang_name = _detect_language(project_spec)
        log.info(f"Phát hiện ngôn ngữ: {lang_name}")

        today = date.today().isoformat()

        template = """
    Bạn là AI hỗ trợ quản lý dự án phần mềm.
    Đặc tả dự án:
    {project_spec}

    Hãy chia dự án thành các phase phát triển hợp lý.
    Yêu cầu:
    - Viết bằng {lang}
    - Không markdown. Không giải thích. Chỉ trả JSON hợp lệ
    - Khoảng 8-10 phase
    - Mỗi phase là một giai đoạn lớn của dự án
    - thời gian phải hợp lý so với hôm nay ({date})
    Cấu trúc phase:
    {{
    "title": "tên phase",
    "description": "mô tả phase",
    "phase_start": "YYYY-MM-DD",
    "phase_end": "YYYY-MM-DD"
    }}
    Trả về JSON:
    {{
    "phases":[...]
    }}
    """

        prompt = PromptTemplate(
            template=template,
            input_variables=["project_spec", "lang", "date"]
        )

        parser = PydanticOutputParser(pydantic_object=ListPhaseOut)
        chain = prompt | self.llm

        try:
            ai_msg = chain.invoke({
                "project_spec": project_spec,
                "lang": "Vietnamese" if lang_name == "Vietnamese" else "English",
                "date": today,
            })
            response = parser.invoke(ai_msg)
            usage = ai_msg.response_metadata.get("token_usage", {})

            log.info("Generate phases thành công")
            result = response.model_dump() if hasattr(response, 'model_dump') else response
            if isinstance(result, dict):
                result["usage"] = usage
            return result

        except Exception as e:
            log.exception("Generate phases failed")
            return {"error": str(e)}
    
    # SINH NHIỀU TASKS CHO MỖI PHASE
    def generate_tasks(
        self,
        project_id: int,
        phase_content: str,
        users: list
    ) -> ListComposeOut:
        """
        Sinh tasks cho 1 phase và tự động phân công user.
        """

        if not phase_content:
            return {"error": "Thiếu nội dung phase"}

        lang_name = _detect_language(phase_content)
        log.info(f"Phát hiện ngôn ngữ: {lang_name}")

        today = date.today().isoformat()

    
        # convert lại thành JSON gọn gàng
        users_clean = users 
        

        template = """
    Bạn là AI quản lý dự án phần mềm.
    Phase hiện tại:
    {phase_content}
    Danh sách thành viên dự án:
    {users}
    Hãy tạo danh sách tasks cho phase này và phân công cho thành viên phù hợp.
    Yêu cầu:
    - Viết bằng {lang}
    - Không markdown. Không giải thích. Trả JSON hợp lệ
    - Khoảng 8-10 tasks
    - priority thuộc: URGENT,HIGH,MEDIUM,LOW
    - type thuộc: FEATURE,BUG,IMPROVEMENT,RESEARCH,DOCUMENTATION,TESTING,DEPLOYMENT,ENHANCEMENT,MAINTENANCE,OTHER
    - assignee phải là id user từ danh sách trên.Phân công theo skills phù hợp
    - thời gian hợp lý so với hôm nay ({date})
    Cấu trúc task:
    {{
    "title": "string",
    "description": "mô tả chi tiết",
    "priority": "MEDIUM",
    "type": "FEATURE",
    "story_point": number(ước lượng bằng story point nếu có thể),
    "start_date": "YYYY-MM-DD",
    "due_date": "YYYY-MM-DD",
    "assignee": number,
    "todos": ["step 1","step 2"]
    }}
    Trả về JSON:
    {{
    "tasks":[...]
    }}
    """
        prompt = PromptTemplate(
            template=template,
            input_variables=[
                "phase_content",
                "users",
                "lang",
                "date"
            ]
        )

        parser = PydanticOutputParser(pydantic_object=ListComposeOut)
        chain = prompt | self.llm

        try:
            ai_msg = chain.invoke({
                "phase_content": phase_content,
                "users": users_clean,
                "lang": "Vietnamese" if lang_name == "Vietnamese" else "English",
                "date": today,
            })
            response = parser.invoke(ai_msg)
            usage = ai_msg.response_metadata.get("token_usage", {})

            log.info("Generate tasks thành công")
            result = response.model_dump() if hasattr(response, 'model_dump') else response
            if isinstance(result, dict):
                result["usage"] = usage
            return result

        except Exception as e:
            log.exception("Generate tasks failed")
            return {"error": str(e)}


    # NHẬN XÉT HIỆU SUẤT USER (ĐỌC TASKS + COMMENTS)
    def review_user_performance(
        self,
        context: str,
        lang: str = "Vietnamese"
    ) -> Dict[str, Any]:
        """
        Đánh giá hiệu suất user dựa trên context do backend cung cấp.
        Trả về:
        - performance_review
        - improvement_suggestions
        """

        if not context:
            return {"error": "Thiếu context để đánh giá."}

        if not self.enabled or not self.llm:
            return {"error": "LLM service chưa được khởi tạo."}

        template = """
    Bạn là AI chuyên đánh giá hiệu suất làm việc trong dự án phần mềm.
    Thông tin:
    {context}
    Hãy đưa ra đánh giá hiệu suất của thành viên.
    Yêu cầu:Viết bằng {lang}.Phân tích cả tasks và comments.Không markdown.Chỉ trả JSON hợp lệ
    JSON format:
    {{
    "performance_review": "Nhận xét tổng thể về hiệu suất làm việc",
    "improvement_suggestions": "Các hướng cải thiện cụ thể cho thành viên"
    }}
    """

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "lang"]
        )

        chain = prompt | self.llm

        try:
            ai_msg = chain.invoke({
                "context": context,
                "lang": lang,
            })

            text = ai_msg.content.strip()

            result = json.loads(text)

            usage = ai_msg.response_metadata.get("token_usage", {})
            result["usage"] = usage

            return result

        except Exception as e:
            log.exception("review_user_performance thất bại")
            return {"error": str(e)}

        
# ---------------------UTILS HỖ TRỢ---------------------

# Hàm phát hiện ngôn ngữ (dùng langdetect)
def _detect_language(text: str) -> str:
    """Phát hiện ngôn ngữ của text (Vietnamese hoặc English)."""
    try:
        lang_code = detect(text)
        return "Vietnamese" if lang_code.startswith("vi") else "English"
    except Exception:
        return "English"
    
# Hàm định dạng context cho mỗi task
def format_context(tasks):
    if not tasks:
        return "Không có ngữ cảnh liên quan."
    
    return "\n".join([
        f"- {t.get('metadata', {}).get('title', '')}: {t.get('content','')}"
        for t in tasks
    ])

# Hàm định dạng context cho mỗi task có kèm id
def format_context_included_taskid(tasks):
    if not tasks:
        return "Không có ngữ cảnh liên quan."
    
    return "\n".join([
        f"- [ID: {t.get('metadata', {}).get('task_id', 'N/A')}] {t.get('metadata', {}).get('title', '')}: {t.get('content','')}"
        for t in tasks
    ])

# Hàm loại bỏ trùng lặp trong danh sách task
def dedupe_tasks(tasks):
    seen = set()
    out = []
    for t in tasks:
        tid = t.get("metadata", {}).get("task_id")
        if tid not in seen:
            seen.add(tid)
            out.append(t)
    return out

# Hàm gộp ngữ cảnh từ project tasks và semantic tasks
def merge_context(project_tasks, semantic_tasks):
    # Ưu tiên semantic trước
    merged = semantic_tasks + project_tasks
    return dedupe_tasks(merged)

# Hàm hỗ trợ build doc từ task object để truyền vào estimate
def _build_full_text(task: dict, fallback: str = "") -> str:
    """
    Tạo text đầu vào cho model ML bằng cách gom toàn bộ thông tin cần thiết.
    """
    if not task:
        return fallback

    parts = []
    parts.append(task.get("title", ""))
    parts.append(task.get("description", ""))

    tags = task.get("tags", [])
    if tags:
        parts.append("Tags: " + ", ".join(tags))

    priority = task.get("priority")
    if priority:
        parts.append(f"Priority: {priority}")

    # Subtasks
    for st in task.get("subtasks", []):
        st_title = st.get("title", "")
        st_desc = st.get("description", "")
        parts.append(f"Subtask: {st_title} - {st_desc}")

    return " ".join(filter(None, parts)) or fallback

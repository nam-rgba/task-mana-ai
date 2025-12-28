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

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

device = "cpu"
print("Current device:", device)



class LLMService:
    def __init__(self, vector_store: Optional[VectorStoreService] = None):
        # Load models
        self.llm = ModelsLoader.llm()
        self.embed_model = ModelsLoader.embeddings()
        self.xgb_model = ModelsLoader.xgb_model()
        self.scaler = ModelsLoader.scaler()
        self.vector_store = vector_store or VectorStoreService()

        # LLM enabled flag
        self.enabled = self.llm is not None
        
    # HÀM TẠO TASK (COMPOSITION) - DÙNG RETRIEVER LẤY NGỮ CẢNH
    def compose_with_llm(self, user_input: str, project_id: Optional[str] = None) -> ComposeOut:
        """
        Tạo task mới. Ngữ cảnh kết hợp 2 cách:
        1) Các task thuộc project (ngữ cảnh nền)
        2) Các task liên quan đến truy vấn (nếu có)
        
        """
        
        lang_name = _detect_language(user_input)
        log.info(f"Phát hiện ngôn ngữ: {lang_name}")

        # 1. Lấy ngữ cảnh các task thuộc project
        project_tasks = self.vector_store.retrieve_tasks_by_project(
            k=10,
            project_id=project_id  
        ) or  []
    
        # 2. Lấy các task liên quan đến truy vấn (nếu có) thuộc cùng dự án
        semantic_tasks = self.vector_store.retrieve_tasks_by_query(
            query=user_input,
            project_id=project_id,     # lọc trong project
            k= 5
        ) or []

        log.info(f"Ngữ cảnh dự án: {len(project_tasks)} tasks | Ngữ cảnh mô tả: {len(semantic_tasks)} tasks")


        # Build context
        merged_tasks = merge_context(project_tasks, semantic_tasks)
        context_text = format_context(merged_tasks)

        # 3. Tạo prompt 
        template = """
            Bạn là AI hỗ trợ quản lý dự án phần mềm. Hãy tạo nhiệm vụ mới tuân thủ ngữ cảnh dự án.
            
            Ngữ cảnh của dự án (RẤT QUAN TRỌNG):
            {context}

            Mô tả nhiệm vụ mới của người dùng:
            {user_input}

            Yêu cầu đầu ra:
            - Tạo một nhiệm vụ mới dựa trên **Nội dung mô tả công việc mới** nhưng phải **phù hợp với Ngữ cảnh Dự án**.
            - Viết bằng ngôn ngữ **{lang}**.
            - Trả về **JSON hợp lệ** (đúng theo cấu trúc) không kèm lời giải thích, markdown (```json).
            - Subtasks tối đa 3, chỉ tạo khi thực sự cần thiết.

            Cấu trúc JSON bắt buộc:
                {{
                "title": "string",
                "description": "string (mô tả chi tiết, ưu tiên theo dạng Given–When–Then nhưng viết dưới dạng Ngôn Ngữ Tự Nhiên, KHÔNG kèm theo các chữ Given, When, Then)",
                "priority": "HIGH | MEDIUM | LOW",
                "due_date": "YYYY-MM-DD"(tính từ ngày {date}), 
                "subtasks": [
                    {{
                    "title": "string",
                    "description": "string (chi tiết)",
                    "priority": "HIGH | MEDIUM | LOW"
                    }}
                ]
                }}
    """
        prompt = PromptTemplate(
            template=template, 
            input_variables=["context", "user_input", "lang", "date"])
        
        # Tạo chain
        chain = ( prompt | self.llm | PydanticOutputParser(pydantic_object=ComposeOut) )

        # 4. Invoke LLM và Xử lý Kết quả
        response = ""
        try:
            response = chain.invoke({
                "context": context_text,
                "user_input": user_input, 
                "lang": lang_name,
                "date": date.today().isoformat(),
            })

            # Response is always: {"raw": AIMessage, "parsed": ComposeOut, "parsing_error": None/error}
            log.info("Tạo task thành công với LLM Compose.")
            return response
        
        except Exception as e:
            log.exception("LLM chain.invoke/run failed")
            return {"error": str(e)}

    # HÀM GỢI Ý GÁN NGƯỜI THỰC HIỆN (ASSIGNMENT) 
    def assign_candidate(self, task: dict, project_id: str, requirement_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Gợi ý gán người dùng phù hợp,
        Hàm này tự động lấy dữ liệu từ Vector Store.
        """
        lang_name = _detect_language(requirement_text)

        if not project_id:
            log.warning("Task không có project_id.")
            return {"error": "Missing project_id in task data."}

        try:
            # Lấy user theo requirement (nếu có). Các user này phải tham gia project (Hiện tại chưa có)
            related_users = []
            if requirement_text:
                related_users = self.vector_store.retrieve_users_by_query(
                    task_text=requirement_text,
                    project_id=project_id,
                    k=4
                ) or []
            else:
                related_users = self.vector_store.retrieve_users_by_project(
                    project_id=project_id,
                    k=4
                ) or []

            tasks_docs = []
            for user in related_users:
                user_id = user["metadata"].get("user_id")
                log.info(f"Lấy lịch sử task của user_id={user_id}")
                t = self.vector_store.retrieve_tasks_by_user(
                    user_id=user_id,
                    project_id=project_id,
                    k=4
                )
                log.info(f"Tìm thấy {len(t)} tasks cho user_id={user_id}")
                tasks_docs.extend(t)
            
            log.info(f"Đã truy xuất {len(related_users)} users liên quan và {len(tasks_docs)} tasks liên quan tương ứng.")


            # Template và Prompt (không đổi)
            template = """
                    Bạn là một AI có nhiệm vụ phân công người thực hiện phù hợp nhất cho một nhiệm vụ. 
                    Đầu vào:
                    - task: mô tả nhiệm vụ cần phân công.
                    - users: danh sách ứng viên liên quan (đã lọc theo yêu cầu và thuộc dự án)
                    - tasks:  lịch sử nhiệm vụ của các ứng viên (để tính tải công việc và thành tích)
                    - requirement: yêu cầu bổ sung (nếu có)

                    Tiêu chí chọn 1 ứng viên phù hợp:
                    1) Khớp vị trí/chuyên môn với nội dung task.
                    2) Tải công việc (ưu tiên người có ít nhiệm vụ TODO/IN_PROGRESS).
                    3) Thành tích (ưu tiên người có nhiều nhiệm vụ DONE).
                    4) Kinh nghiệm (năm làm việc — dùng khi điểm bằng nhau).

                    Yêu cầu trả về:
                    JSON DUY NHẤT, đúng cấu trúc:
                    {{
                        "assignee": {{
                            "id": "",
                            "email": "",
                            "name": "",
                            "position": ""
                            }},
                        "reason": "Giải thích ngắn bằng tiếng Vietnamese: lý do vì sao phù hợp dựa trên tải công việc, thành tích DONE, kinh nghiệm."
                    }}

                    Không trả lời ngoài JSON.

                    Dữ liệu ngữ cảnh:
                    task: {task}
                    users: {users}
                    tasks: {tasks}
                    requirement: {requirement}
                    """
            prompt = PromptTemplate(template=template, input_variables=["task", "users", "tasks", "requirement", "lang"])
            

            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=AssignOut)
            

            assignment = chain.invoke({
                "task": json.dumps(task, ensure_ascii=False),
                "users": json.dumps(related_users, ensure_ascii=False),
                "tasks": json.dumps(tasks_docs, ensure_ascii=False),
                "requirement": requirement_text or "None",
                "lang": lang_name
            })
            print (assignment)
            log.info(f"Assignment successful: {assignment.assignee.name}")

            # Return structured response
            return {
                "assignment": assignment.model_dump(),  # Already a dict from Pydantic
                "related_users": related_users
            }
            
        except Exception as e:
            log.exception("Gán người thực hiện thất bại")
            return {"error": str(e)}
    
    
    # HÀM DUPLICATE FINDER
    def find_duplicate_tasks(self, task: dict, threshold: float = 0.2, k: int = 3, project_id: Optional[str] = None) -> DuplicateTaskOut:
        """
        Tìm các nhiệm vụ trùng lặp dựa trên embedding similarity/distance.
        """
        try:
            
            subtasks_text = " ".join([f"{st.get('title','')} {st.get('description','')}" for st in task.get("subtasks", []) or []])
            assignee_name = (task.get("assignee") or {}).get("name", "")
            tags_text = " ".join(task.get("tags", []))
            priority = task.get("priority", "")
            
            query_text = " ".join([
                task.get("title", ""),
                task.get("description", ""),
                tags_text,
                priority,
                assignee_name,
                subtasks_text
            ])
            log.info(f"Đang kiểm tra duplicate...")
            retrieved = self.vector_store.retrieve_tasks_with_scores(query=query_text, k=k, project_id=project_id)
            #print(json.dumps(retrieved, indent=2, ensure_ascii=False))

            duplicates = []
            for doc in retrieved:
                score = None
                if isinstance(doc, dict):
                    score = doc.get("score", doc.get("similarity", doc.get("distance", None)))
                try:
                    if (score <= threshold):
                        duplicates.append({
                            "content": doc.get("content"),  
                            "score": score,          
                            "metadata": doc.get("metadata"),
                        })
                        
                except Exception:
                    # ignore malformed score and continue
                    continue

            return DuplicateTaskOut(
                duplicates=[DuplicateDoc(**d) for d in duplicates],
                nearest_tasks=retrieved
            )
        except Exception as e:
            log.exception("TÌm kiếm trùng lặp thất bại")
            return {"error": str(e)}
        
    # HÀM DỰ ĐOÁN STORY POINT
    def predict_story_point(self, text: str) -> float:
        """
        Dự đoán story point thô sử dụng XGBRegressor + embeddings + 2 feature scaled.
        - Nhận đầu vào json: 
            {
            text: "string" -- Nhớ combine title + description trước khi truyền vào
            }
        """
        
        if not self.xgb_model or not self.embed_model or not hasattr(self, "scaler"):
            raise RuntimeError("Model, embeddings hoặc scaler chưa được load!")

        # ---- Embedding ----
        # DÙng model embeđing đã load 
        emb_list = self.embed_model.embed_query(text)
        emb = np.array(emb_list, dtype=float)
        # normalize thành vector đơn vị để giống với behavior trước đây
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        # Tạo thêm đặc trưng (word count, char count)
        word_count = len(text.split())
        char_count = len(text)
        extra = pd.DataFrame([[word_count, char_count]], columns=["word_count", "char_count"])
        extra_scaled = self.scaler.transform(extra)  # MinMaxScaler transform

        # Kết hợp embedding + extra features 
        X_input = np.hstack([emb.reshape(1, -1), extra_scaled])  # shape (1, 386)

        # Dự đoán
        pred = self.xgb_model.predict(X_input)[0]

        return round(float(pred), 2)

    # HÀM GỢI Ý STORY POINT THEO PLANNING POKER
    def suggest_story_point(self, value: float) -> str:
        """
        Chuyển giá trị thô sang Story Point gần nhất theo chuẩn Planning Poker."""
        STORY_POINTS = [0.5, 1, 2, 3, 5, 8, 13]
        
        diffs = [(abs(value - sp), sp) for sp in STORY_POINTS]
        diffs.sort(key=lambda x: x[0])

        best = diffs[0][1]
        second = diffs[1][1]

        # Nếu giá trị nằm giữa 2 story point gần nhau
        if abs(value - best) < 0.4 and abs(value - second) < 0.4:
            return f"{best} - {second}"
        return str(best)

    # REPORT / METRICS CHAIN & LOCAL CALC
    # --- GENERATE SPRINT REPORT
    def generate_sprint_report(self, tasks: List[Dict[str, Any]], use_llm_summary: bool = True) -> Dict[str, Any]:
        """
        Tạo sprint report: flatten subtasks, tính metrics local, và tóm tắt bằng LLM nếu yêu cầu.
        """
        all_tasks = []

        def flatten(task_list, parent_title=None):
            for t in task_list:
                flat_task = t.copy()
                flat_task['parent_task'] = parent_title
                # parse dates
                for date_field in ['start_date', 'end_date', 'completed_at']:
                    if flat_task.get(date_field):
                        try:
                            flat_task[date_field] = datetime.fromisoformat(flat_task[date_field])
                        except Exception as e:
                            log.warning(f"Cannot parse {date_field}={flat_task[date_field]}: {e}")
                            flat_task[date_field] = None
                    else:
                        flat_task[date_field] = None
                # ensure story_points is numeric
                flat_task['story_points'] = float(flat_task.get('story_points', 0))
                all_tasks.append(flat_task)
                # recursively flatten subtasks
                subtasks = t.get('subtasks', [])
                if subtasks:
                    flatten(subtasks, parent_title=t.get('title'))

        flatten(tasks)

        # --- Metrics Local
        total_tasks = len(all_tasks)
        completed_tasks = len([t for t in all_tasks if str(t.get('status')).lower() == 'done'])
        pending_tasks = total_tasks - completed_tasks
        total_story_points = sum(t['story_points'] for t in all_tasks)
        avg_story_points = total_story_points / total_tasks if total_tasks else 0
        progress_percent = (completed_tasks / total_tasks * 100) if total_tasks else 0

        metrics = {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "pending_tasks": pending_tasks,
            "total_story_points": total_story_points,
            "average_story_points": round(avg_story_points, 2),
            "progress_percent": round(progress_percent, 2),
        }

        report = {
            "metrics": metrics,
            "tasks": all_tasks
        }

        # --- Optional: LLM Summary of tasks & metrics ---
        if use_llm_summary:
            try:
                report['llm_summary'] = self.summarize_with_llm(all_tasks)
            except Exception as e:
                log.warning(f"LLM summary failed: {e}")
                report['llm_summary'] = "LLM summary failed."

        return report

    # LLM SUMMARY - Test
    def summarize_with_llm(self, text: str) -> Dict[str, Any]:
        """
        Tóm tắt text bằng LLM nếu khả dụng, fallback bằng extractive summarization.
        Luôn trả về dict chuẩn JSON.
        """
        # --- 1) LLM summary ---
        if hasattr(self, "enabled") and self.enabled and hasattr(self, "llm") and self.llm:
            try:
                prompt_template = """
                Bạn là một AI Agile assistant chuyên về quản lý Sprint. 
                Bạn nhận vào dữ liệu các task của một Sprint dưới dạng JSON:
                {text}

                Các yêu cầu khi trả về:
                1. Viết **summary tự nhiên**, ngắn gọn, dễ hiểu cho PM / stakeholder.
                2. Highlight:
                - Trends (xu hướng tiến độ)
                - Risks (rủi ro)
                - Bottlenecks (nút thắt)
                3. Trả về **CHỈ JSON**, KHÔNG kèm markdown hay giải thích nào khác.
                4. JSON phải có cấu trúc sau (bắt buộc):

                {{
                "summary": "string (tóm tắt ngắn gọn về sprint)", 
                "metrics": {{
                    "velocity": number (tổng story points hoàn thành trong sprint),
                    "burndown_status": "string (on track, behind schedule, ahead of schedule)",
                    "average_lead_time": number (nếu có thể tính, phút/giờ/ngày),
                    "average_cycle_time": number (nếu có thể tính, phút/giờ/ngày)
                }},
                "recommendations": "string optional (khuyến nghị cho sprint tiếp theo)"
                }}

                **Lưu ý quan trọng**:
                - Nếu một số giá trị metrics không thể tính, trả về null.
                - Không thêm bất kỳ ký tự, markdown, giải thích hay code block nào xung quanh JSON.
                - Chỉ trả về **JSON hợp lệ**.
                """

                prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                chain = prompt | self.llm | PydanticOutputParser(pydantic_object=SprintSummary)

                resp = chain.invoke({"text": text})
                return resp
        

                

            except Exception as e:
                log.warning(f"LLM summarization failed: {e}")
                return {"summary": "LLM summary failed.", "metrics": {}, "recommendations": ""}

        else:
            log.warning("LLM not enabled or not set up; skipping LLM summarization.")
            # fallback
            try:
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
                return {"summary": " ".join(sentences[:3]), "metrics": {}, "recommendations": ""}
            except Exception as e:
                log.warning(f"Fallback summarization failed: {e}")
                return {"summary": "Summary failed due to internal error.", "metrics": {}, "recommendations": ""}

    # PIPELINE SINH TASK
    def generate_task(
        self,
        user_input: str,
        project_id: Optional[str] = None,
        requirement_text: Optional[str] = None,
        duplicate_threshold: float = 0.25,
        duplicate_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Pipeline hoàn chỉnh để sinh task, từ input của người dùng đến các gợi ý chi tiết.
        Sử dụng LangChain Expression Language (LCEL) để kết nối các bước.
        """

        # ĐỊnh nghĩa các runnable 

        # 1) Tạo task từ input của người dùng
        compose_chain = RunnableLambda(
            lambda x: self.compose_with_llm(
                user_input=x["user_input"],
                project_id=x.get("project_id")
            ).model_dump() # Sử dụng .model_dump() để chuyển Pydantic model thành dict
        )

        # 2) Dự đoán Story Point
        story_point_chain = RunnableLambda(
            lambda x: self.predict_story_point(
                text=_build_full_text(x.get("composed_task"), fallback=x["user_input"])
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
            "requirement_text": requirement_text,
        }

        # result sẽ là một dictionary chứa kết quả từ tất cả các bước
        result = full_chain.invoke(input_data)


        return {
            "user_input": {
                "user_input": user_input,
                "project_id": project_id,
                "requirement_text": requirement_text
            },
            "composed_task": result.get("composed_task"),
            "estimated_story_point": result.get("estimated_story_point"),
            "story_point_suggestion": result.get("story_point_suggestion"),
            "duplicates": result.get("duplicates"),
            "assignment": result.get("assignment"),
        }
    




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

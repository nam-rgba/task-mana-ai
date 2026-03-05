import os
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from app.services.models_loader import ModelsLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.fetch_data import FetchData  # Sẽ dùng sau khi đã có dữ liệu thật


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ID Prefixes để tránh xung đột giữa các collection (Do các id là số tự nhiên từ 0,1,2,3...)
USER_PREFIX = "user_"
TASK_PREFIX = "task_"
PROJECT_PREFIX = "project_"
TEAM_PREFIX = "team_"

def make_user_id(user_id: int) -> str:
    """Chuyển đổi user_id số thành string có tiền tố."""
    return f"{USER_PREFIX}{user_id}"


def make_task_id(task_id: int) -> str:
    """Chuyển đổi task_id số thành string có tiền tố."""
    return f"{TASK_PREFIX}{task_id}"


def make_project_id(project_id: int) -> str:
    """Chuyển đổi project_id số thành string có tiền tố."""
    return f"{PROJECT_PREFIX}{project_id}"

def make_team_id(team_id: int) -> str:
    """Chuyển đổi team_id số thành string có tiền tố."""
    return f"{TEAM_PREFIX}{team_id}"

# Không còn dùng các file mock này nữa
USERS_FILE = "./app/data/users.json"
TASKS_FILE = "./app/data/tasks_numeric.json"
PROJECTS_FILE = "./app/data/projects.json"  # Thêm đường dẫn file projects
GUIDES_FOLDER = "./app/data/guides"  # Thư mục chứa các file PDF hướng dẫn sử dụng

# Hỗ trợ tăng ngữ nghĩa cho tìm kiếm 1
ROLE_CONFIGS = {
    "MANAGER": {
        "description": "lãnh đạo nhóm, phân công nhiệm vụ, giám sát tiến độ dự án",
        "skills": ["team leadership", "project planning", "risk management", "communication"],
    },
    "DEV_FE": {
        "description": "phát triển giao diện người dùng và tối ưu trải nghiệm",
        "skills": ["React", "Vue", "HTML", "CSS", "JavaScript", "UI optimization", "animation", "responsive layout"],
    },
    "DEV_BE": {
        "description": "xây dựng API, xử lý logic phía server và kết nối cơ sở dữ liệu",
        "skills": ["Node.js", "Python", "SQL", "REST API", "authentication", "security", "scalability", "microservices"],
    },
    "DEV_FULLSTACK": {
        "description": "làm việc cả frontend và backend, có khả năng triển khai toàn bộ sản phẩm",
        "skills": ["React", "Node.js", "SQL", "API design", "CI/CD", "testing"],
    },
    "DEV_MOBILE": {
        "description": "phát triển ứng dụng di động Android/iOS",
        "skills": ["Android", "iOS", "Flutter", "React Native", "mobile UI", "mobile performance"],
    },
    "DEV_OPS": {
        "description": "quản lý hạ tầng, CI/CD và vận hành hệ thống",
        "skills": ["Docker", "Kubernetes", "CI/CD", "Linux", "Networking", "Monitoring", "Cloud AWS/GCP"],
    },
    "TESTER": {
        "description": "kiểm thử phần mềm, đảm bảo chất lượng sản phẩm",
        "skills": ["manual testing", "automation testing", "Selenium", "QA process"],
    },
    "DESIGNER": {
        "description": "thiết kế giao diện và trải nghiệm người dùng",
        "skills": ["Figma", "UI/UX", "wireframe", "prototype", "illustration", "design system"],
    },
    "BUSINESS_ANALYST": {
        "description": "phân tích yêu cầu nghiệp vụ và kết nối giữa team kỹ thuật và khách hàng",
        "skills": ["requirements analysis", "diagramming UML/BPMN", "stakeholder communication", "documentation"],
    },
}

# Hàm tạo Document gọn gàng hơn
def create_user_document(user):
    pos_key = user.get("position")
    # Lấy config theo role, nếu không thấy thì dùng default
    config = ROLE_CONFIGS.get(pos_key, {
        "description": "thực hiện các nhiệm vụ phù hợp với vai trò của mình",
        "skills": []
    })
    
    description = config["description"]
    skills_str = ", ".join(config["skills"])
    years = user.get('yearOfExperience', 0)

    # Tạo nội dung text
    text_content = (
        f"Tên: {user.get('name')}. "
        f"Vị trí: {description}. "
        f"Kỹ năng: {skills_str}. "
        f"Kinh nghiệm: {years} năm."
    )

    # Metadata
    metadata = {
        "user_id": user["id"],
        "email": user.get("email"),
        "name": user.get("name"),
        "position": pos_key,
        "year_of_experience": years,
        "type": "user",
    }
    
    return Document(
        page_content=text_content, 
        metadata=metadata, 
        id=make_user_id(user["id"])
    )


def create_task_document(task):
    # Nội dung văn bản (để AI thực hiện tìm kiếm ngữ nghĩa)
    text_content = (
        f"Tiêu đề: {task.get('title')}\n"
        f"Mô tả: {task.get('description')}\n"
        f"Trạng thái: {task.get('status')}\n"
        f"Ưu tiên: {task.get('priority')}\n"
        f"Loại task: {task.get('type')}\n"
        f"Dự án ID: {task.get('projectId')}\n"
    )

    # 3. Metadata (Chỉ lấy các trường cần thiết)
    metadata = {
        "task_id": task.get("id"),
        "title": task.get("title"),
        "description": task.get("description"),
        "status": task.get("status"),
        "priority": task.get("priority"),
        "project_id": task.get("projectId"),
        "assignee_id": task.get("assigneeId"),  # JSON dùng assigneeId
        "reviewer_id": task.get("reviewerId"),
        "due_date": task.get("dueDate"),
        "estimate_effort": task.get("estimateEffort"),
        "actual_effort": task.get("actualEffort"),
        "created_at": task.get("createdAt"),
        "updated_at": task.get("updatedAt"),
        "completed_at": task.get("completedAt"),
        "qc_review_status": task.get("qcReviewStatus"),
        "task_type": task.get("type"),
        "type": "task",  # Phân biệt với user/guide trong vector store
    }


    return Document(
        page_content=text_content, 
        metadata=metadata, 
        id=make_task_id(task["id"])
    )


def create_project_document(project):
    """
    Tạo Document cho Project từ dữ liệu API thực tế.
    Chỉ giữ lại các thông tin quan trọng cho Semantic Search.
    """
    # 1. Tạo nội dung văn bản để Vectorize (Dùng cho tìm kiếm ngữ nghĩa)
    # Tập trung vào những gì người dùng thường hỏi: tên dự án là gì, làm về cái gì, trạng thái ra sao.
    name = project.get("name", "N/A")
    description = project.get("description", "Không có mô tả")
    p_type = project.get("type", "N/A")
    status = project.get("status", "N/A")
    visibility = project.get("visibility", "N/A")

    text_content = (
        f"Dự án: {name}. "
        f"Mô tả: {description}. "
        f"Loại hình: {p_type}. "
        f"Trạng thái: {status}. "
        f"Chế độ hiển thị: {visibility}."
    )

    # 2. Metadata để lọc (Filter) và hiển thị kết quả (chỉ lấy các trường cần thiết)
    metadata = {
        "project_id": project.get("id"),
        "name": name,
        "description": description,
        "status": status,
        "project_type": p_type,
        "team_id": project.get("teamId"),
        "lead_id": project.get("leadId"),
        "visibility": visibility,
        "start_date": project.get("startDate"),
        "end_date": project.get("endDate"),
        "created_at": project.get("createdAt"),
        "updated_at": project.get("updatedAt"),
        "type": "project",  # Dùng để phân biệt trong Vector Store
    }

    return Document(
        page_content=text_content,
        metadata=metadata,
        id=make_project_id(project.get("id")),
    )


class VectorStoreService:
    def __init__(
        self,
        connection: Optional[str] = None,
        device: str = "cpu",
        init_stores: bool = True,
    ):

        self.connection = connection or os.getenv("DB_CONNECT_STRING")
        self.device = device
        self.embedding_adapter = ModelsLoader.embeddings()

        # Vector stores bắt đầu là None và sẽ được tạo khi cần thiết
        self.users_store: Optional[PGVector] = None
        self.tasks_store: Optional[PGVector] = None
        self.projects_store: Optional[PGVector] = None
        self.guides_store: Optional[PGVector] = None
        self.members_store: Optional[PGVector] = None

        # Test với file JSON
        self.users_file = USERS_FILE
        self.tasks_file = TASKS_FILE
        self.projects_file = PROJECTS_FILE
        self.guides_folder = GUIDES_FOLDER

        if not self.connection:
            log.warning("DB_CONNECTION_STRING missing. Vector stores disabled.")
        else:
            self._init_stores()

        # self.sync_data(force=True) # Đã sync từ api, ko call hàm này nữa nha
        self.sync_guides(force=True)  # Sync dữ liệu hướng dẫn từ PDFs

    def _init_stores(self):
        """Initialize PGVector instances."""
        try:
            self.users_store = PGVector(
                embeddings=self.embedding_adapter,
                collection_name="users",
                connection=self.connection,
                use_jsonb=True,
            )
            self.tasks_store = PGVector(
                embeddings=self.embedding_adapter,
                collection_name="tasks",
                connection=self.connection,
                use_jsonb=True,
            )

            self.projects_store = PGVector(
                embeddings=self.embedding_adapter,
                collection_name="projects",
                connection=self.connection,
                use_jsonb=True,
            )

            self.guides_store = PGVector(
                embeddings=self.embedding_adapter,
                collection_name="guides",
                connection=self.connection,
                use_jsonb=True,
            )

            self.members_store = PGVector(
                embeddings=self.embedding_adapter,
                collection_name="team_members",
                connection=self.connection,
                use_jsonb=True,
            )

            log.info("PGVector stores initialized successfully.")
        except Exception as e:
            log.exception(f"Failed to initialize PGVector: {e}")

    def _ensure_stores(self) -> bool:
        """Kiểm tra các vector store đã được khởi tạo"""
        if self.users_store and self.tasks_store and self.projects_store and self.guides_store and self.members_store:
            return True
        if not self.connection:
            log.warning(
                "Không thể kết nối Databse: connection string chưa được cấu hình."
            )
            return False
        try:
            self._init_stores()
            return True
        except Exception as e:
            log.exception("Lỗi: %s", e)
            return False

    def _load_json(self, path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            log.warning(f"File not found: {path}")
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Load và xử lý file PDF hướng dẫn
    def _load_guides(self) -> List[Document]:
        """Load PDF từ thư mục hoặc file, loại bỏ ký tự NUL, split thành chunks và gắn metadata."""

        if not os.path.exists(self.guides_folder):
            log.warning(f"Không tìm thấy: {self.guides_folder}")
            return []

        documents = []

        try:
            # Trường hợp guides_folder là thư mục
            if os.path.isdir(self.guides_folder):
                pdf_files = [
                    f for f in os.listdir(self.guides_folder) if f.endswith(".pdf")
                ]
                for file in pdf_files:
                    loader = PyPDFLoader(os.path.join(self.guides_folder, file))
                    docs = loader.load()
                    # Loại bỏ ký tự NUL
                    for d in docs:
                        d.page_content = d.page_content.replace("\x00", "")
                    documents.extend(docs)

            # Trường hợp guides_folder là 1 file PDF
            elif self.guides_folder.endswith(".pdf"):
                loader = PyPDFLoader(self.guides_folder)
                docs = loader.load()
                for d in docs:
                    d.page_content = d.page_content.replace("\x00", "")
                documents.extend(docs)

            if not documents:
                return []

            # Split văn bản
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
            )
            splits = text_splitter.split_documents(documents)

            # Loại bỏ các chunk rỗng
            safe_splits = []
            for doc in splits:
                if not doc.page_content.strip():
                    continue
                source = doc.metadata.get("source") or "unknown.pdf"
                doc.metadata["type"] = "guide"
                doc.metadata["file_name"] = os.path.basename(source)
                # Id cố định: file name + index trong file
                idx = len(
                    [
                        d
                        for d in safe_splits
                        if d.metadata["file_name"] == os.path.basename(source)
                    ]
                )
                doc.id = f"{os.path.basename(source)}_{idx}"
                safe_splits.append(doc)

            log.info(f"Đã tạo {len(safe_splits)} guide chunks từ PDF.")
            return safe_splits

        except Exception as e:
            log.error(f"Error processing PDFs: {str(e)}")
            return []

    # Đây là hàm sync dữ liệu từ file JSON mock nha
    # def sync_data(self, force: bool = False):
    #     if not self._ensure_stores():
    #         log.warning("Vector stores chưa khởi tạo, bỏ qua sync_data")
    #         return

    #     # --- SYNC USERS ---
    #     users_data = self._load_json(USERS_FILE)
    #     if users_data and self.users_store:
    #         if force:
    #             # Xóa toàn bộ nếu force=True
    #             self.users_store.delete(ids=None)

    #         user_docs = [create_user_document(u) for u in users_data]
    #         # Sử dụng ids= để PGVector biết chính xác ID nào cần quản lý
    #         ids = [doc.id for doc in user_docs]
    #         self.users_store.add_documents(user_docs, ids=ids)
    #         log.info(f"Synced {len(user_docs)} users.")

    #     # --- SYNC TASKS ---
    #     tasks_data = self._load_json(TASKS_FILE)
    #     if tasks_data and self.tasks_store:
    #         if force:
    #             self.tasks_store.delete(ids=None)

    #         subtasks_by_parent = {}
    #         parent_tasks = []
    #         for t in tasks_data:
    #             pid = t.get("parentTaskId")
    #             if pid:
    #                 subtasks_by_parent.setdefault(pid, []).append(t)
    #             else:
    #                 parent_tasks.append(t)

    #         task_docs = []
    #         for t in parent_tasks:
    #             t["subtasks"] = subtasks_by_parent.get(t["id"], [])
    #             task_docs.append(create_task_document(t))

    #         ids = [doc.id for doc in task_docs]
    #         self.tasks_store.add_documents(task_docs, ids=ids)
    #         log.info(f"Synced {len(task_docs)} parent tasks.")

    #     # --- SYNC PROJECTS ---
    #     projects_data = self._load_json(self.projects_file)
    #     if projects_data and self.projects_store:
    #         if force:
    #             self.projects_store.delete(ids=None)
    #         project_docs = [create_project_document(p) for p in projects_data]
    #         ids = [doc.id for doc in project_docs]
    #         self.projects_store.add_documents(project_docs, ids=ids)
    #         log.info(f"Synced {len(project_docs)} projects.")

    def sync_guides(self, force: bool = False):
        # --- SYNC GUIDES (PDFs) ---
        if self.guides_store:
            if force:
                self.guides_store.delete(ids=None)

            guide_docs = self._load_guides()
            if guide_docs:
                # Với guides, vì ID được tạo từ file_name_index, ta nên check kỹ hơn
                # Lấy các ids hiện có trong DB
                # Chú ý: get_by_ids có thể trả về None hoặc lỗi nếu DB rỗng
                try:
                    existing_ids = set()
                    # Chỉ check nếu không phải force
                    if not force:
                        check_ids = [doc.id for doc in guide_docs]
                        res = self.guides_store.get_by_ids(check_ids)
                        existing_ids = set(d.id for d in res if d is not None)

                    to_add = [doc for doc in guide_docs if doc.id not in existing_ids]
                    if to_add:
                        self.guides_store.add_documents(
                            to_add, ids=[d.id for d in to_add]
                        )
                        log.info(f"Synced {len(to_add)} new guide chunks.")
                except Exception as e:
                    # Nếu bảng rỗng hoàn toàn, get_by_ids có thể lỗi, ta add hết
                    self.guides_store.add_documents(
                        guide_docs, ids=[d.id for d in guide_docs]
                    )

    # TRUY VẤN ------------------
    def users_retriever(
        self, k: int = 5, filters: Optional[dict] = None
    ) -> VectorStoreRetriever:
        """Trả về một LangChain Retriever object cho Users."""
        store = self.users_store
        if not store:
            raise ValueError("User Store not initialized")

        return store.as_retriever(
            search_type="similarity", search_kwargs={"k": k, "filter": filters}
        )
 
    def tasks_retriever(
        self, k: int = 5, project_id: Optional[int] = None
    ) -> VectorStoreRetriever:
        """Trả về một LangChain Retriever object cho Tasks."""
        store = self.tasks_store
        if not store:
            raise ValueError("Task Store not initialized")

        search_kwargs = {"k": k}
        if project_id:
            search_kwargs["filter"] = {"project_id": project_id}

        return store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

    def guides_retriever(self, k: int = 5) -> VectorStoreRetriever:
        """Trả về một LangChain Retriever object cho Guides."""
        store = self.guides_store
        if not store:
            raise ValueError("Guide Store not initialized")

        return store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    # Hàm tìm kiếm task liên quan
    def retrieve_tasks_by_query(
        self, query: str, k: int = 5, project_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if not self.tasks_store:
            log.warning(
                "tasks_store not initialized. retrieve_tasks_for_query returns empty list."
            )
            return []
        # try to include project_id filter; still use metadata-level filtering afterward to remove tombstones
        filters = {"project_id": project_id} if project_id else None
        try:
            results = self.tasks_store.similarity_search(query, k=k, filter=filters)
        except TypeError:
            results = self.tasks_store.similarity_search(query, k=k)
        except Exception as e:
            log.exception("similarity_search failed: %s", e)
            return []
        seen = set()
        unique = []
        for r in results:
            meta = getattr(r, "metadata", {}) or {}
            # skip tombstones
            if meta.get("is_deleted"):
                continue
            orig = meta.get("task_id")
            if orig and orig in seen:
                continue
            if orig:
                seen.add(orig)
            unique.append({"content": r.page_content, "metadata": meta})
            if len(unique) >= k:
                break
        return unique
    
    # Hàm tìm kiếm task liên quan kèm score
    def retrieve_tasks_with_scores(
        self, query: str, k: int = 5, project_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Tìm kiếm các task liên quan kèm score (cosine distance - Càng thấp -> Khoảng cách càng ngắn --> Càng giống) tương tự dựa trên truy vấn."""
        if not self.tasks_store:
            log.warning(
                "tasks_store not initialized. retrieve_tasks_with_scores returns empty list."
            )
            return []
        filters = {"project_id": project_id} if project_id else None
        try:
            results = self.tasks_store.similarity_search_with_score(
                query, k=k, filter=filters
            )
        except TypeError:
            results = self.tasks_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            log.exception("similarity_search_with_score failed: %s", e)
            return []

        seen = set()
        unique = []
        for item in results:
            doc, score = item[0], item[1]
            meta = getattr(doc, "metadata", {}) or {}
            if meta.get("is_deleted"):
                continue
            orig = meta.get("task_id")
            if orig and orig in seen:
                continue
            if orig:
                seen.add(orig)
            unique.append(
                {"content": doc.page_content, "metadata": meta, "score": score}
            )
            if len(unique) >= k:
                break
        return unique

    # Hàm lấy các task trong một project
    def retrieve_tasks_by_project(
        self, project_id: int, k: int = 10
    ) -> List[Dict[str, Any]]:
        if not self.tasks_store:
            log.warning(
                "tasks_store not initialized. retrieve_tasks_for_project returns empty list."
            )
            return []
        filters = {"project_id": project_id}
        try:
            results = self.tasks_store.similarity_search("", k=k, filter=filters)
        except TypeError:
            results = self.tasks_store.similarity_search("", k=k)
        except Exception as e:
            log.exception("similarity_search failed: %s", e)
            return []
        return [
            {"content": r.page_content, "metadata": r.metadata}
            for r in results
            if not (getattr(r, "metadata", {}) or {}).get("is_deleted")
        ]
    
    # Hàm lấy task theo user_id
    def retrieve_tasks_by_user(
        self, user_id: int, project_id: str, k: int = 10
    ) -> List[Dict[str, Any]]:
        if not self.tasks_store:
            log.warning(
                "tasks_store not initialized. retrieve_tasks_for_user returns empty list."
            )
            return []
        filters = {"assignee_id": user_id, "project_id": project_id}
        try:
            results = self.tasks_store.similarity_search("", k=k, filter=filters)
        except TypeError:
            results = self.tasks_store.similarity_search("", k=k)
        except Exception as e:
            log.exception("similarity_search failed: %s", e)
            return []
        return [
            {"content": r.page_content, "metadata": r.metadata}
            for r in results
            if not (getattr(r, "metadata", {}) or {}).get("is_deleted")
        ]

    #HELPERS TÌM KIẾM USER ------------------
    def _get_user_roles_map_by_project(self, project_id: int) -> Dict[str, str]:
        """Helper: Trả về Map { 'user_7': 'LEAD', 'user_2': 'QC' } của một project."""
        if not all([self.projects_store, self.members_store]):
            return {}
            
        # 1. Lấy team_id từ project
        project_docs = self.projects_store.get_by_ids([make_project_id(project_id)])
        if not project_docs:
            return {}
        
        team_id = project_docs[0].metadata.get("team_id")
        
        # 2. Lấy members từ team
        team_docs = self.members_store.get_by_ids([make_team_id(team_id)])
        if not team_docs:
            return {}

        members = team_docs[0].metadata.get("members", [])
        
        # Tạo dictionary: { "user_7": "LEAD", "user_2": "QC", ... }
        return {make_user_id(m["user_id"]): m.get("role", "MEMBER") for m in members if "user_id" in m}
        
    # Hàm lấy các user tham gia trong một project
    def retrieve_users_by_project(self, project_id: int, k: int = 10) -> List[Dict[str, Any]]:
        if not self.users_store:
            return []

        try:
            # Lấy map roles: { "user_7": "LEAD", ... }
            user_roles_map = self._get_user_roles_map_by_project(project_id)
            if not user_roles_map:
                return []

            user_ids = list(user_roles_map.keys())
            users = self.users_store.get_by_ids(user_ids) #Lấy thông tin các user từ vector store
            
            out = []
            for user in users:
                meta = (getattr(user, "metadata", {}) or {}).copy()
                if meta.get("is_deleted"):
                    continue
                
                # Lấy role từ map đã tạo ở trên
                meta["role_in_team"] = user_roles_map.get(user.id, "MEMBER")
                    
                out.append({
                    "content": user.page_content, 
                    "metadata": meta
                })
                if len(out) >= k:
                    break
            return out
        except Exception as e:
            log.exception(f"Lỗi retrieve_users_by_project: {e}")
            return []
    
    # Hàm lấy các user thuộc project nhưng liên quan đến yêu cầu cụ thể
    def retrieve_users_by_query(self, task_text: str, project_id: Optional[int] = None, k: int = 5) -> List[Dict[str, Any]]:
        if not self.users_store:
            return []

        try:
            if project_id:
                # Lấy map roles của project
                user_roles_map = self._get_user_roles_map_by_project(project_id)
                if not user_roles_map:
                    return []

                # Tìm kiếm similarity
                results = self.users_store.similarity_search(task_text, k=k*3) # Tăng số lượng k để tránh thiếu user do filter role
                
                filtered = []
                for r in results:
                    # Nếu user search được nằm trong team của project
                    if r.id in user_roles_map:
                        meta = (getattr(r, "metadata", {}) or {}).copy()
                        if meta.get("is_deleted"):
                            continue
                            
                        # Gán role tương ứng
                        meta["role_in_team"] = user_roles_map[r.id]
                        
                        filtered.append({"content": r.page_content, "metadata": meta})
                    
                    if len(filtered) >= k:
                        break
                return filtered

            else:
                # Nếu không có project_id, search bình thường và không có role_in_team
                results = self.users_store.similarity_search(task_text, k=k)
                return [{"content": r.page_content, "metadata": r.metadata} for r in results if not r.metadata.get("is_deleted")]

        except Exception as e:
            log.exception(f"Lỗi retrieve_users_by_query: {e}")
            return []
        

    # CRUD document ------------------
    def get_task_by_id(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Lấy thông tin chi tiết của một task từ vector store bằng ID."""
        if not self._ensure_stores():
            return None
        try:
            # Truy vấn trực tiếp theo id document, hàm get_by_ids yêu cầu danh sách các ids và trả về danh sách các docs
            docs = self.tasks_store.get_by_ids([make_task_id(task_id)])
            if not docs:
                return None

            docs = [d for d in docs if not d.metadata.get("is_deleted", False)]

            return docs
        except Exception as e:
            log.exception(f"Failed to get task by id {task_id}: {e}")
            return None

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Lấy thông tin chi tiết của một user từ vector store bằng ID."""
        if not self._ensure_stores():
            return None
        try:
            docs = self.users_store.get_by_ids(
                [make_user_id(user_id)]
            )  # Hàm yêu cầu danh sách nên phải đưa vào là list dù chỉ 1 id
            if not docs:
                return None
            docs = [d for d in docs if not d.metadata.get("is_deleted", False)]
            return docs

        except Exception as e:
            log.exception(f"Failed to get task by id {user_id}: {e}")
            return None

    def get_project_by_id(self, project_id: int) -> Optional[Dict[str, Any]]:
        """Lấy thông tin chi tiết của một project từ vector store bằng ID."""
        if not self._ensure_stores():
            return None
        try:
            docs = self.projects_store.get_by_ids(
                [make_project_id(project_id)]
            )  # Hàm yêu cầu danh sách nên phải đưa vào là list dù chỉ 1 id
            if not docs:
                return None
            docs = [d for d in docs if not d.metadata.get("is_deleted", False)]
            return docs

        except Exception as e:
            log.exception(f"Failed to get project by id {project_id}: {e}")
            return None

    def get_team_by_id(self, team_id: int) -> Optional[Dict[str, Any]]:
        """Lấy thông tin chi tiết của một team từ vector store bằng ID."""
        if not self._ensure_stores():
            return None
        try:
            docs = self.members_store.get_by_ids(
                [make_team_id(team_id)]
            )  # Hàm yêu cầu danh sách nên phải đưa vào là list dù chỉ 1 id
            if not docs:
                return None
            return docs

        except Exception as e:
            log.exception(f"Failed to get team by id {team_id}: {e}")
            return None
        
    def upsert_task(self, task: dict, force: bool = False):
        if not self._ensure_stores() or "id" not in task:
            return False
        try:
            task_id = task["id"]
            task_id_str = make_task_id(task_id)
            if force:
                existing_docs = self.tasks_store.get_by_ids([task_id_str])
                if existing_docs:
                    self.tasks_store.delete(ids=[task_id_str])
                    log.info(f"Xóa task {task_id} cũ trước khi upsert do force=True.")
            # Thêm doc mới
            doc = create_task_document(task)
            self.tasks_store.add_documents([doc], ids = [doc.id])
            log.info(f"Upsert task {task_id} thành công.")
            return True
        except Exception as e:
            log.exception(f"Upsert task {task_id} thất bại: {e}")
            return False

    def upsert_user(self, user: dict, force: bool = False):
        if not self._ensure_stores() or "id" not in user:
            return False
        try:
            user_id = user["id"]
            user_id_str = make_user_id(user_id)
            if force:
                existing_docs = self.users_store.get_by_ids([user_id_str])
                if existing_docs:
                    self.users_store.delete(ids=[user_id_str])
                    log.info(f"Xóa user {user_id} cũ trước khi upsert do force=True.")
            # Thêm doc mới
            doc = create_user_document(user)
            self.users_store.add_documents([doc], ids = [doc.id])
            log.info(f"Upsert task {user_id} thành công.")
            return True
        except Exception as e:
            log.exception(f"Upsert user {user_id} thất bại: {e}")
            return False

    def upsert_team(self, team: dict, force: bool = False):
        if not self._ensure_stores() or "id" not in team:
            return False
        try:
            team_id = team["id"]
            team_id_str = make_team_id(team_id)
            if force:
                existing_docs = self.members_store.get_by_ids([team_id_str])
                if existing_docs:
                    self.members_store.delete(ids=[team_id_str])
                    log.info(f"Xóa team {team_id} cũ trước khi upsert do force=True.")
            # Thêm doc mới
            doc = Document(
                page_content=f"Team ID: {team_id}, Team Name: {team.get('name', 'N/A')}",
                metadata={
                    "team_id": team_id,
                    "name": team.get("name"),
                    "members": team.get("members", []),
                    "type": "team",
                },
                id=team_id_str,
            )
            self.members_store.add_documents([doc], ids = team_id_str)
            log.info(f"Upsert team {team_id} thành công.")
            return True
        except Exception as e:
            log.exception(f"Upsert team {team_id} thất bại: {e}")
            return False

    def upsert_project(self, project: dict, force: bool = False):
        if not self._ensure_stores() or "id" not in project:
            return False
        try:
            project_id = project["id"]
            project_id_str = make_project_id(project_id)
            if force:
                existing_docs = self.projects_store.get_by_ids([project_id_str])
                if existing_docs:
                    self.projects_store.delete(ids=[project_id_str])
                    log.info(
                        f"Xóa project {project_id} cũ trước khi upsert do force=True."
                    )
            # Thêm doc mới
            doc = create_project_document(project)
            self.projects_store.add_documents([doc], ids = [doc.id])
            log.info(f"Upsert project {project_id} thành công.")
            return True
        except Exception as e:
            log.exception(f"Upsert project {project_id} thất bại: {e}")
            return False

    def delete_task_by_id(self, task_id: int):
        if not self._ensure_stores():
            return False
        try:
            self.tasks_store.delete(
                ids=[make_task_id(task_id)]
            )  # Delete theo danh sách ids
            log.info(f"Đã xoá task {task_id} khỏi vector store.")
            return True
        except Exception as e:
            log.exception(f"Xóa task {task_id} thất bại: {e}")
            return False

    def delete_user_by_id(self, user_id: int):
        if not self._ensure_stores():
            return False
        try:
            self.users_store.delete(
                ids=[make_user_id(user_id)]
            )  # Delete theo danh sách ids
            log.info(f"Đã xoá user {user_id} khỏi vector store.")
            return True
        except Exception as e:
            log.exception(f"Xóa user {user_id} thất bại: {e}")
            return False

    def delete_project_by_id(self, project_id: int):
        if not self._ensure_stores():
            return False
        try:
            self.projects_store.delete(
                ids=[make_project_id(project_id)]
            )  # Delete theo danh sách ids
            log.info(f"Đã xoá project {project_id} khỏi vector store.")
            return True
        except Exception as e:
            log.exception(f"Xóa project {project_id} thất bại: {e}")
            return False

    def delete_team_by_id(self, team_id: int):
        if not self._ensure_stores():
            return False
        try:
            self.members_store.delete(
                ids=[make_team_id(team_id)]
            )  # Delete theo danh sách ids
            log.info(f"Đã xoá team {team_id} khỏi vector store.")
            return True
        except Exception as e:
            log.exception(f"Xóa team {team_id} thất bại: {e}")
            return False
        
    # SYN DATA TỪ API ------------------
    async def sync_tasks_from_api(self, force: bool = False):
        """
        Đồng bộ toàn bộ tasks từ API vào vector store.
        """
        if not self._ensure_stores():
            log.warning("Vector stores chưa khởi tạo, bỏ qua sync_from_api")
            return

        # --- SYNC TASKS FROM API ---
        try:
            tasks_data = await FetchData.fetch_all_tasks_from_api()
            if not tasks_data:
                log.warning("Không lấy được dữ liệu tasks từ API.")
                return

            if force:
                self.tasks_store.delete(ids=None)
                
            task_docs = [create_task_document(t) for t in tasks_data]
            ids = [doc.id for doc in task_docs]
            self.tasks_store.add_documents(task_docs, ids=ids)
                
            log.info(f"Đã đồng bộ {len(task_docs)} tasks từ API vào vector store.")
        except Exception as e:
            log.exception(f"Lỗi khi đồng bộ tasks từ API: {e}")

    async def sync_projects_from_api(self, force: bool = False):
        """
        Đồng bộ toàn bộ projects từ API vào vector store.
        """
        if not self._ensure_stores():
            log.warning("Vector stores chưa khởi tạo, bỏ qua sync_from_api")
            return

        # --- SYNC PROJECTS FROM API ---
        try:
            projects_data = await FetchData.fetch_all_projects_from_api()
            if not projects_data:
                log.warning("Không lấy được dữ liệu projects từ API.")
                return

            if force:
                self.projects_store.delete(ids=None)

            project_docs = [create_project_document(p) for p in projects_data]
            ids = [doc.id for doc in project_docs]
            self.projects_store.add_documents(project_docs, ids=ids)
            log.info(
                f"Đã đồng bộ {len(project_docs)} projects từ API vào vector store."
            )
        except Exception as e:
            log.exception(f"Lỗi khi đồng bộ projects từ API: {e}")

    async def sync_users_from_api(self, force: bool = False):
        """
        Đồng bộ toàn bộ users từ API vào vector store.
        """
        if not self._ensure_stores():
            log.warning("Vector stores chưa khởi tạo, bỏ qua sync_from_api")
            return

        # --- SYNC PROJECTS FROM API ---
        try:
            users_data = await FetchData.fetch_all_users_from_api()
            if not users_data:
                log.warning("Không lấy được dữ liệu users từ API.")
                return

            if force:
                self.users_store.delete(ids=None)

            users_docs = [create_user_document(p) for p in users_data]
            ids = [doc.id for doc in users_docs]
            self.users_store.add_documents(users_docs, ids=ids)
            log.info(
                f"Đã đồng bộ {len(users_docs)} users từ API vào vector store."
            )
        except Exception as e:
            log.exception(f"Lỗi khi đồng bộ users từ API: {e}")
    
    async def sync_team_member_from_api(self, force: bool = False):
        """
        Đồng bộ toàn bộ team members từ API vào vector store.
        Mỗi document là 1 team, metadata chứa danh sách user_id kèm role.
        """
        if not self._ensure_stores():
            log.warning("Vector stores chưa khởi tạo, bỏ qua sync_from_api")
            return

        try:
            team_map = await FetchData.fetch_all_team_members_map()  # {teamId: [{"userId":..., "role":...}, ...]}
            if not team_map:
                log.warning("Không lấy được dữ liệu team members từ API.")
                return

            if force:
                self.members_store.delete(ids=None)

            member_docs = []
            for team_id, members in team_map.items():
                # Tạo list các dict user_id/role
                user_roles = [{"user_id": m["userId"], "role": m.get("role")} for m in members]
                doc_id = f"team_{team_id}"
                text_content = f"Team {team_id} with users: {user_roles}"
                metadata = {
                    "team_id": team_id,
                    "members": user_roles,
                    "type": "team"
                }
                member_docs.append(Document(page_content=text_content, metadata=metadata, id=doc_id))

            ids = [doc.id for doc in member_docs]
            self.members_store.add_documents(member_docs, ids=ids)
            log.info(
                f"Đã đồng bộ {len(member_docs)} team từ API vào vector store."
            )
        except Exception as e:
            log.exception(f"Lỗi khi đồng bộ team members từ API: {e}")



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
from app.services.fetch_data import FetchData #Sẽ dùng sau khi đã có dữ liệu thật


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

USERS_FILE = "./app/data/users.json"
TASKS_FILE = "./app/data/tasks.json"
PROJECTS_FILE = "./app/data/projects.json"  # Thêm đường dẫn file projects
GUIDES_FOLDER = "./app/data/guides" # Thư mục chứa các file PDF hướng dẫn sử dụng
USER_SKILLS = {
    "MANAGER": ["team leadership", "project planning", "risk management", "communication"],
    "DEV_FE": ["React", "Vue", "HTML", "CSS", "JavaScript", "UI optimization", "animation", "responsive layout"],
    "DEV_BE": ["Node.js", "Python", "SQL", "REST API", "authentication", "security", "scalability", "microservices"],
    "DEV_FULLSTACK": ["React", "Node.js", "SQL", "API design", "CI/CD", "testing"],
    "DEV_MOBILE": ["Android", "iOS", "Flutter", "React Native", "mobile UI", "mobile performance"],
    "DEV_OPS": ["Docker", "Kubernetes", "CI/CD", "Linux", "Networking", "Monitoring", "Cloud AWS/GCP"],
    "TESTER": ["manual testing", "automation testing", "Selenium", "QA process"],
    "DESIGNER": ["Figma", "UI/UX", "wireframe", "prototype", "illustration", "design system"],
    "BUSINESS_ANALYST": ["requirements analysis", "diagramming UML/BPMN", "stakeholder communication", "documentation"]
}

def user_role_description(position: str) -> str:
    mapping = {
        "MANAGER": "lãnh đạo nhóm, phân công nhiệm vụ, giám sát tiến độ dự án",
        "DEV_FE": "phát triển giao diện người dùng và tối ưu trải nghiệm",
        "DEV_BE": "xây dựng API, xử lý logic phía server và kết nối cơ sở dữ liệu",
        "DEV_FULLSTACK": "làm việc cả frontend và backend, có khả năng triển khai toàn bộ sản phẩm",
        "DEV_MOBILE": "phát triển ứng dụng di động Android/iOS",
        "DEV_OPS": "quản lý hạ tầng, CI/CD và vận hành hệ thống",
        "TESTER": "kiểm thử phần mềm, đảm bảo chất lượng sản phẩm",
        "DESIGNER": "thiết kế giao diện và trải nghiệm người dùng",
        "BUSINESS_ANALYST": "phân tích yêu cầu nghiệp vụ và kết nối giữa team kỹ thuật và khách hàng",
    }
    return mapping.get(position, "thực hiện các nhiệm vụ phù hợp với vai trò của mình")

def create_user_document(user):
    text_content = (
        f"Tên: {user.get('name')}. "
        f"Vị trí: {user_role_description(user.get('position'))}. "
        f"Kỹ năng: {', '.join(USER_SKILLS.get(user.get('position'), []))}. "
        f"Kinh nghiệm: {user.get('yearOfExperience', 0)} năm."
    )

    metadata = {
        "user_id": user["id"],
        "email": user.get("email"),
        "name": user.get("name"),
        "position": user.get("position"),
        "year_of_experience": user.get("yearOfExperience", 0),
        "avatar": user.get("avatar"),
        "created_at": user.get("createdAt"),
        "type": "user"
    }
    return Document(page_content=text_content, metadata=metadata, id=str(user["id"]))

def create_task_document(task):
    subtasks_text = "Không có subtasks."
    if task.get("subtasks"):
        subtasks_text = "\n".join([f"- {s['title']}: {s['description']}" for s in task["subtasks"]])

    text_content = (
        f"{task.get('title')}\n"
        f"{task.get('description')}\n"
        f"Trạng thái: {task.get('status')}\n"
        f"Ưu tiên: {task.get('priority')}\n"
        f"Dự án: {task.get('projectId')}\n"
        f"Nhiệm vụ con:\n{subtasks_text}"
    )

    metadata = {
        "task_id": task["id"],
        "title": task.get("title"),
        "description": task.get("description"),
        "status": task.get("status"),
        "priority": task.get("priority"),
        "project_id": task.get("projectId"),
        "implementor_id": task.get("implementorId"),
        "reviewer_id": task.get("reviewerId"),
        "due_date": task.get("dueDate"),
        "estimate_effort": task.get("estimateEffort"),
        "actual_effort": task.get("actualEffort"),
        "created_at": task.get("createdAt"),
        "updated_at": task.get("updatedAt"),
        "parent_task_id": task.get("parentTaskId"),
        "task_type": task.get("type"), # FEATURE, BUG, IMPROVEMENT, ...
        "is_deleted": False,
        "type": "task" # Type này là phân biệt giữa user/task/guide trong cùng một vector store
    }
    return Document(page_content=text_content, metadata=metadata, id=str(task["id"]))

def create_project_document(project):
    text_content = (
        f"Tên dự án: {project.get('name')}. "
        f"Mô tả: {project.get('description')}. "
        f"Trạng thái: {project.get('status')}. "
        f"Loại: {project.get('type')}. "
        f"Danh mục: {project.get('category')}. "
        f"Nhóm: {project.get('teamId')}. "
        f"Ngày bắt đầu: {project.get('startDate')}, ngày kết thúc: {project.get('endDate')}."
    )
    metadata = {
        "project_id": project["id"],
        "key": project.get("key"),
        "name": project.get("name"),
        "description": project.get("description"),
        "status": project.get("status"),
        "type": project.get("type"),
        "team_id": project.get("teamId"),
        "lead_id": project.get("leadId"),
        "category": project.get("category"),
        "color": project.get("color"),
        "start_date": project.get("startDate"),
        "end_date": project.get("endDate"),
        "created_at": project.get("createdAt"),
        "updated_at": project.get("updatedAt"),
        "type": "project"
    }
    return Document(page_content=text_content, metadata=metadata, id=str(project["id"]))



class VectorStoreService:
    def __init__(self, connection: Optional[str] = None, device: str = "cpu", init_stores: bool = True):

        self.connection = connection or os.getenv("DB_CONNECT_STRING")
        self.device = device
        self.embedding_adapter = ModelsLoader.embeddings()

        # Vector stores bắt đầu là None và sẽ được tạo khi cần thiết
        self.users_store: Optional[PGVector] = None
        self.tasks_store: Optional[PGVector] = None
        self.projects_store: Optional[PGVector] = None
        self.guides_store: Optional[PGVector] = None

        # Test với file JSON
        self.users_file = USERS_FILE
        self.tasks_file = TASKS_FILE
        self.projects_file = PROJECTS_FILE
        self.guides_folder = GUIDES_FOLDER

        if not self.connection:
            log.warning("DB_CONNECTION_STRING missing. Vector stores disabled.")
        else:
            self._init_stores()

        self.sync_data(force=False)
    
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


            log.info("PGVector stores initialized successfully.")
        except Exception as e:
            log.exception(f"Failed to initialize PGVector: {e}")
    
    def _ensure_stores(self) -> bool:
        """ Kiểm tra các vector store đã được khởi tạo """
        if self.users_store and self.tasks_store and self.projects_store:
            return True
        if not self.connection:
            log.warning("Không thể kết nối Databse: connection string chưa được cấu hình.")
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
                pdf_files = [f for f in os.listdir(self.guides_folder) if f.endswith('.pdf')]
                for file in pdf_files:
                    loader = PyPDFLoader(os.path.join(self.guides_folder, file))
                    docs = loader.load()
                    # Loại bỏ ký tự NUL
                    for d in docs:
                        d.page_content = d.page_content.replace('\x00', '')
                    documents.extend(docs)

            # Trường hợp guides_folder là 1 file PDF
            elif self.guides_folder.endswith('.pdf'):
                loader = PyPDFLoader(self.guides_folder)
                docs = loader.load()
                for d in docs:
                    d.page_content = d.page_content.replace('\x00', '')
                documents.extend(docs)

            if not documents:
                return []

            # Split văn bản
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
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
                idx = len([d for d in safe_splits if d.metadata["file_name"] == os.path.basename(source)])
                doc.id = f"{os.path.basename(source)}_{idx}"
                safe_splits.append(doc)


            log.info(f"Đã tạo {len(safe_splits)} guide chunks từ PDF.")
            return safe_splits

        except Exception as e:
            log.error(f"Error processing PDFs: {str(e)}")
            return []
        
    def sync_data(self, force: bool = False):
        if not self._ensure_stores():
            log.warning("Vector stores chưa khởi tạo, bỏ qua sync_data")
            return

        # --- SYNC USERS ---
        users_data = self._load_json(USERS_FILE)
        if users_data and self.users_store:
            if force:
                # Xóa toàn bộ nếu force=True
                self.users_store.delete(ids=None) 
            
            user_docs = [create_user_document(u) for u in users_data]
            # Sử dụng ids= để PGVector biết chính xác ID nào cần quản lý
            ids = [doc.id for doc in user_docs]
            self.users_store.add_documents(user_docs, ids=ids)
            log.info(f"Synced {len(user_docs)} users.")

        # --- SYNC TASKS ---
        tasks_data = self._load_json(TASKS_FILE)
        if tasks_data and self.tasks_store:
            if force:
                self.tasks_store.delete(ids=None)

            subtasks_by_parent = {}
            parent_tasks = []
            for t in tasks_data:
                pid = t.get("parentTaskId")
                if pid: subtasks_by_parent.setdefault(pid, []).append(t)
                else: parent_tasks.append(t)

            task_docs = []
            for t in parent_tasks:
                t["subtasks"] = subtasks_by_parent.get(t["id"], [])
                task_docs.append(create_task_document(t))
            
            ids = [doc.id for doc in task_docs]
            self.tasks_store.add_documents(task_docs, ids=ids)
            log.info(f"Synced {len(task_docs)} parent tasks.")

        # --- SYNC PROJECTS ---
        projects_data = self._load_json(self.projects_file)
        if projects_data and self.projects_store:
            if force:
                self.projects_store.delete(ids=None)
            project_docs = [create_project_document(p) for p in projects_data]
            ids = [doc.id for doc in project_docs]
            self.projects_store.add_documents(project_docs, ids=ids)
            log.info(f"Synced {len(project_docs)} projects.")

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
                        self.guides_store.add_documents(to_add, ids=[d.id for d in to_add])
                        log.info(f"Synced {len(to_add)} new guide chunks.")
                except Exception as e:
                    # Nếu bảng rỗng hoàn toàn, get_by_ids có thể lỗi, ta add hết
                    self.guides_store.add_documents(guide_docs, ids=[d.id for d in guide_docs])



    # TRUY VẤN ------------------
    def users_retriever(self, k: int = 5, filters: Optional[dict] = None) -> VectorStoreRetriever:
        """Trả về một LangChain Retriever object cho Users."""
        store = self.users_store
        if not store:
            raise ValueError("User Store not initialized")
        
        return store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "filter": filters}
        )

    def tasks_retriever(self, k: int = 5, project_id: Optional[str] = None) -> VectorStoreRetriever:
        """Trả về một LangChain Retriever object cho Tasks."""
        store = self.tasks_store
        if not store:
            raise ValueError("Task Store not initialized")
        
        search_kwargs = {"k": k}
        if project_id:
            search_kwargs["filter"] = {"project_id": project_id}

        return store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
    
    def guides_retriever(self, k: int = 5) -> VectorStoreRetriever:
        """Trả về một LangChain Retriever object cho Guides."""
        store = self.guides_store
        if not store:
            raise ValueError("Guide Store not initialized")
        
        return store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    

    # Hàm tìm kiếm task liên quan 
    def retrieve_tasks_by_query(self, query: str, k: int = 5, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.tasks_store:
            log.warning("tasks_store not initialized. retrieve_tasks_for_query returns empty list.")
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
    def retrieve_tasks_with_scores(self, query: str, k: int = 5, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """ Tìm kiếm các task liên quan kèm score (cosine distance - Càng thấp -> Khoảng cách càng ngắn --> Càng giống) tương tự dựa trên truy vấn. """
        if not self.tasks_store:
            log.warning("tasks_store not initialized. retrieve_tasks_with_scores returns empty list.")
            return []
        filters = {"project_id": project_id} if project_id else None
        try:
            results = self.tasks_store.similarity_search_with_score(query, k=k, filter=filters)
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
            unique.append({"content": doc.page_content, "metadata": meta, "score": score})
            if len(unique) >= k:
                break
        return unique

    # Hàm lấy các task trong một project 
    def retrieve_tasks_by_project(self, project_id: str, k: int = 10) -> List[Dict[str, Any]]:
        if not self.tasks_store:
            log.warning("tasks_store not initialized. retrieve_tasks_for_project returns empty list.")
            return []
        filters = {"project_id": project_id}
        try:
            results = self.tasks_store.similarity_search("", k=k, filter=filters)
        except TypeError:
            results = self.tasks_store.similarity_search("", k=k)
        except Exception as e:
            log.exception("similarity_search failed: %s", e)
            return []
        return [{"content": r.page_content, "metadata": r.metadata} for r in results if not (getattr(r, "metadata", {}) or {}).get("is_deleted")]
    
    # Hàm lấy task theo user_id
    def retrieve_tasks_by_user(self, user_id: str, project_id:str, k: int = 10) -> List[Dict[str, Any]]:
        if not self.tasks_store:
            log.warning("tasks_store not initialized. retrieve_tasks_for_user returns empty list.")
            return []
        filters = {"implementor_id": user_id, "project_id": project_id}
        try:
            results = self.tasks_store.similarity_search("", k=k, filter=filters)
        except TypeError:
            results = self.tasks_store.similarity_search("", k=k)
        except Exception as e:
            log.exception("similarity_search failed: %s", e)
            return []
        return [{"content": r.page_content, "metadata": r.metadata} for r in results if not (getattr(r, "metadata", {}) or {}).get("is_deleted")]
    

    # Hàm lấy các user tham gia trong một project
    def retrieve_users_by_project(self, project_id: str, k: int = 10) -> List[Dict[str, Any]]:
        if not self.users_store:
            log.warning("users_store not initialized. retrieve_users_for_project returns empty list.")
            return []
        try:
            results = self.users_store.similarity_search("", k=k, filter=None)
        except TypeError:
            results = self.users_store.similarity_search("", k=k)
        except Exception as e:
            log.exception("users similarity_search failed: %s", e)
            return []
        # use getattr safely for metadata access and filter out tombstones
        out = []
        for r in results:
            meta = getattr(r, "metadata", {}) or {}
            if meta.get("is_deleted"):
                continue
            position = meta.get('position', '')
            content = f"{r.page_content}. {position}"
            out.append({"content": content, "metadata": meta})
            if len(out) >= k:
                break
        return out

    # Hàm tìm người dùng phù hợp cho văn bản nhiệm vụ
    def retrieve_users_by_query(self, task_text: str, k: int = 5, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.users_store:
            log.warning("users_store not initialized. retrieve_users_for_task_text returns empty list.")
            return []
        filters = None # filters = {"project_id": project_id} Hiện tại chưa cung cấp
        try:
            results = self.users_store.similarity_search(task_text, k=k, filter=filters)
        except TypeError:
            results = self.users_store.similarity_search(task_text, k=k)
        except Exception as e:
            log.exception("users similarity_search failed: %s", e)
            return []
        return [{"content": f"Tên: {r.page_content}. Vị trí: {r.metadata.get('position','')}", "metadata": r.metadata} for r in results if not (getattr(r, "metadata", {}) or {}).get("is_deleted")]


    # CRUD document ------------------
    def get_task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Lấy thông tin chi tiết của một task từ vector store bằng ID."""
        if not self._ensure_stores():
            return None
        try:
            # Truy vấn trực tiếp theo id document, hàm get_by_ids yêu cầu danh sách các ids và trả về danh sách các docs
            docs = self.tasks_store.get_by_ids([task_id])
            if not docs:
                return None
            
            docs = [d for d in docs if not d.metadata.get("is_deleted", False)]
            
            return docs
        except Exception as e:
            log.exception(f"Failed to get task by id {task_id}: {e}")
            return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Lấy thông tin chi tiết của một user từ vector store bằng ID."""
        if not self._ensure_stores():
            return None
        try:
            docs = self.users_store.get_by_ids([user_id]) #Hàm yêu cầu danh sách nên phải đưa vào là list dù chỉ 1 id
            if not docs:
                return None
            docs = [d for d in docs if not d.metadata.get("is_deleted", False)]
            return docs
        
        except Exception as e:
            log.exception(f"Failed to get task by id {user_id}: {e}")
            return None

    def get_project_by_id(self, project_id: str) -> Optional[Dict[str, Any]]:   
        """Lấy thông tin chi tiết của một project từ vector store bằng ID."""
        if not self._ensure_stores():
            return None
        try:
            docs = self.projects_store.get_by_ids([project_id]) #Hàm yêu cầu danh sách nên phải đưa vào là list dù chỉ 1 id
            if not docs:
                return None
            docs = [d for d in docs if not d.metadata.get("is_deleted", False)]
            return docs
        
        except Exception as e:
            log.exception(f"Failed to get project by id {project_id}: {e}")
            return None
        

    def upsert_task(self, task: dict, force: bool = False):
        if not self._ensure_stores() or "id" not in task:
            return False
        try:
            task_id = task["id"]
            if force:
                existing_docs = self.tasks_store.get_by_ids([task_id])
                if existing_docs:
                    self.tasks_store.delete(ids=[task_id])
                    log.info(f"Xóa task {task_id} cũ trước khi upsert do force=True.")
            # Thêm doc mới
            doc = create_task_document(task)
            self.tasks_store.add_documents([doc])
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
            if force:
                existing_docs  = self.users_store.get_by_ids([user_id])
                if existing_docs:
                    self.users_store.delete(ids = [user_id])
                    log.info(f"Xóa user {user_id} cũ trước khi upsert do force=True.")
            # Thêm doc mới
            doc = create_user_document(user)
            self.users_store.add_documents([doc])
            log.info(f"Upsert task {user_id} thành công.")
            return True
        except Exception as e:
            log.exception(f"Upsert user {user_id} thất bại: {e}")
            return False

    def upsert_project(self, project: dict, force: bool = False):
        if not self._ensure_stores() or "id" not in project:
            return False
        try:
            project_id = project["id"]
            if force:
                existing_docs  = self.projects_store.get_by_ids([project_id])
                if existing_docs:
                    self.projects_store.delete(ids = [project_id])
                    log.info(f"Xóa project {project_id} cũ trước khi upsert do force=True.")
            # Thêm doc mới
            doc = create_project_document(project)
            self.projects_store.add_documents([doc])
            log.info(f"Upsert project {project_id} thành công.")
            return True
        except Exception as e:
            log.exception(f"Upsert project {project_id} thất bại: {e}")
            return False
    
    def delete_task_by_id(self, task_id: str):
        if not self._ensure_stores():
            return False
        try:
            self.tasks_store.delete(ids=[task_id]) # Delete theo danh sách ids 
            log.info(f"Đã xoá task {task_id} khỏi vector store.")
            return True
        except Exception as e:
            log.exception(f"Xóa task {task_id} thất bại: {e}")
            return False

    def delete_user_by_id(self, user_id: str):
        if not self._ensure_stores():
            return False
        try:
            self.users_store.delete(ids=[user_id]) # Delete theo danh sách ids 
            log.info(f"Đã xoá user {user_id} khỏi vector store.")
            return True
        except Exception as e:
            log.exception(f"Xóa user {user_id} thất bại: {e}")
            return False
    
    def delete_project_by_id(self, project_id: str):
        if not self._ensure_stores():
            return False
        try:
            self.projects_store.delete(ids=[project_id]) # Delete theo danh sách ids 
            log.info(f"Đã xoá project {project_id} khỏi vector store.")
            return True
        except Exception as e:
            log.exception(f"Xóa project {project_id} thất bại: {e}")
            return False
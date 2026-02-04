from app.utils.api import fetch_from_api

class FetchData:
    @staticmethod
    async def fetch_all_done_tasks_and_export(export_format: str = "csv"):
        """
        Lấy toàn bộ tasks có status DONE và export ra file csv.
        File sẽ được lưu vào thư mục data/train trong workspace Docker.
        """
        import os
        import pandas as pd
        import json
        out_dir = os.path.join(os.getcwd(), "app/data/train") #Lưu ý direct này nha
        os.makedirs(out_dir, exist_ok=True)
        tasks = []
        page = 1
        limit = 100
        total_pages = 1
        while True:
            params = {"status": "DONE", "limit": limit, "page": page}
            resp = await fetch_from_api("/tasks", params=params)

            meta = resp.get("metadata", {})
            page_info = meta.get("page", {})
            page_tasks = meta.get("tasks", [])
            
            tasks.extend(page_tasks)
            total_pages = page_info.get("pages", 1)
            #Tiếp tục đọc trang kế nếu còn
            if page >= total_pages:
                break
            page += 1
        # Chỉ lấy các trường cần thiết
        processed = [
            {
                "Title": t["title"],
                "Description": t["description"],
                "Type": t["type"],
                "Priority": t["priority"],
                "Story_Point": t["actualEffort"]
            }
            for t in tasks if t.get("status") == "DONE"
        ]
        out_path = os.path.join(out_dir, f"tasks_done_train.{export_format}")
        if export_format == "csv":
            df = pd.DataFrame(processed)
            df.to_csv(out_path, index=False, encoding="utf-8")
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(processed, f, ensure_ascii=False, indent=2)
        return out_path
        
    @staticmethod
    async def fetch_all_tasks_from_api(params: dict = None):
        """
        Lấy toàn bộ tasks từ API.
        Trả về list các task (không lọc status).
        """
        tasks = []
        page = 1
        limit = 100
        total_pages = 1
        params = params.copy() if params else {}
        while True:
            params.update({"limit": limit, "page": page})
            resp = await fetch_from_api("/tasks", params=params)
            meta = resp.get("metadata", {})
            page_info = meta.get("page", {})
            page_tasks = meta.get("tasks", [])
            tasks.extend(page_tasks)
            total_pages = page_info.get("pages", 1)
            if page >= total_pages:
                break
            page += 1
        return tasks
    
    @staticmethod
    async def fetch_all_projects_from_api():
        """
        Lấy toàn bộ projects từ API.
        Trả về list các project.
        """
        projects = []
        resp = await fetch_from_api("/projects") #APi này không có params phân trang
        # Metadata là list luôn
        if isinstance(resp.get("metadata"), list):
            projects = resp["metadata"]
        else:
            projects = []
        return projects
    
    # @staticmethod
    # async def fetch_all_users_from_api():
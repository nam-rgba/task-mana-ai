from app.utils.api import fetch_from_api

from app.utils.api import fetch_from_api

class FetchData:
    @staticmethod
    async def get_projects(params: dict = None):
        """Lấy danh sách projects từ API ngoài."""
        return await fetch_from_api("/projects", params=params)

    @staticmethod
    async def get_tasks(params: dict = None):
        """Lấy danh sách tasks từ API ngoài."""
        return await fetch_from_api("/tasks?limit=100", params=params)

    @staticmethod
    async def get_members(params: dict = None):
        """Lấy danh sách members từ API ngoài."""
        return await fetch_from_api("/members", params=params)
import httpx
import os
from dotenv import load_dotenv

load_dotenv()  # Tải biến môi trường từ file .env

EXTERNAL_API_BASE_URL = os.getenv("EXTERNAL_API_BASE_URL", "https://taskee.codes/api/aidata")

async def fetch_from_api(endpoint: str, params: dict = None):
    url = EXTERNAL_API_BASE_URL + endpoint
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()
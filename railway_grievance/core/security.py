from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader
import secrets
from .config import settings

API_KEY_NAME = "RAILWAY-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    if not secrets.compare_digest(api_key, settings.SERVER_API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key
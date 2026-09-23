"""演示入口：.venv/bin/uvicorn app:app --host 127.0.0.1 --port 5347"""
from adcp.web import app

__all__ = ["app"]

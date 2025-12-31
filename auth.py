from functools import wraps
from flask import request, jsonify
from config import API_KEY

def require_api_key(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not API_KEY:
            return jsonify({"error": "Server misconfigured: API_KEY missing"}), 500

        # OpenAI-style: Authorization: Bearer <key>
        auth = request.headers.get("Authorization", "")
        token = ""
        if auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1].strip()

        # Support X-API-Key too
        if not token:
            token = request.headers.get("X-API-Key", "").strip()

        if token != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

        return fn(*args, **kwargs)
    return wrapper

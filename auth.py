"""Bearer token authentication middleware."""

from __future__ import annotations

import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from config import settings

log = logging.getLogger("rlm_proxy.auth")

# Paths that skip auth (monitoring/health)
_OPEN_PATHS = frozenset({"/health", "/v1/rlm/metrics", "/docs", "/openapi.json", "/redoc"})


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Require Bearer token on /v1/* endpoints when RLM_API_KEY is set."""

    async def dispatch(self, request: Request, call_next):
        # Skip auth if no key configured
        if not settings.api_key:
            return await call_next(request)

        # Skip auth for open paths
        if request.url.path in _OPEN_PATHS:
            return await call_next(request)

        # Check Authorization header
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Missing Bearer token", "type": "auth_error"}},
            )

        token = auth_header[7:]  # Strip "Bearer "
        if token != settings.api_key:
            log.warning("Invalid API key from %s", request.client.host if request.client else "unknown")
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Invalid API key", "type": "auth_error"}},
            )

        return await call_next(request)

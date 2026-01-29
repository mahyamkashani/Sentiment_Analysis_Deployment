from fastapi import Request
from typing import Optional, Dict, Any


class BaseController:
    """
    Base controller class providing common utilities for all controllers.
    Similar to SoftRemedyBaseApiController from softremedy_report.

    This class provides helper methods for:
    - Request data extraction
    - Query parameter access
    - Header access
    - Common HTTP operations
    """

    def __init__(self):
        self.request: Optional[Request] = None

    def set_request(self, request: Request):
        """Set the current request context"""
        self.request = request

    def get_request_body(self) -> Dict[str, Any]:
        """
        Get parsed JSON body from request.
        Note: In FastAPI, body is typically already parsed by Pydantic models.
        """
        if not self.request:
            raise ValueError("Request context not set")
        return {}

    def get_query_param(self, key: str, default: Any = None) -> Any:
        """Get query parameter by key"""
        if not self.request:
            raise ValueError("Request context not set")
        return self.request.query_params.get(key, default)

    def get_header(self, key: str, default: Any = None) -> Any:
        """Get header by key"""
        if not self.request:
            raise ValueError("Request context not set")
        return self.request.headers.get(key, default)

    def get_client_ip(self) -> Optional[str]:
        """Get client IP address"""
        if not self.request:
            raise ValueError("Request context not set")
        return self.request.client.host if self.request.client else None

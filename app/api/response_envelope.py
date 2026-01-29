from typing import Any, Optional
from pydantic import BaseModel
from fastapi.responses import JSONResponse


class ResponseEnvelope(BaseModel):
    """
    Standard response wrapper for all API endpoints.
    Ensures consistent response format across the entire API.
    """
    status: str  # "success" or "error"
    code: int    # HTTP status code
    message: str
    data: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def success(cls, data: Any, message: str = "OK", code: int = 200):
        """Create a success response envelope"""
        return cls(
            status="success",
            code=code,
            message=message,
            data=data
        )

    @classmethod
    def error(cls, message: str, code: int = 400, data: Any = None):
        """Create an error response envelope"""
        return cls(
            status="error",
            code=code,
            message=message,
            data=data
        )

    def to_json_response(self) -> JSONResponse:
        """Convert envelope to FastAPI JSONResponse"""
        return JSONResponse(
            status_code=self.code,
            content=self.model_dump()
        )

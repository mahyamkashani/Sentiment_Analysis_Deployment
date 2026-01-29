import asyncio
from functools import wraps
from typing import Callable
from fastapi import HTTPException
from app.api.response_envelope import ResponseEnvelope


def put_in_envelope(func: Callable):
    """
    Decorator that wraps controller method responses in ResponseEnvelope.

    Similar to the @put_in_envelope decorator from softremedy_report,
    but adapted for FastAPI instead of Flask.

    Usage:
        @put_in_envelope
        def my_controller_method(self):
            return {"key": "value"}

    The controller method should return raw data (dict, list, etc.).
    The decorator will wrap it in a ResponseEnvelope and return JSONResponse.

    Exception Handling:
    - HTTPException: Preserves status code and message
    - ValueError: Returns 422 validation error
    - Other exceptions: Returns 500 internal server error
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            # Execute the controller method
            result = await func(*args, **kwargs)

            # Wrap in success envelope
            envelope = ResponseEnvelope.success(data=result)
            return envelope.to_json_response()

        except HTTPException as e:
            # FastAPI HTTPException - preserve status code
            envelope = ResponseEnvelope.error(
                message=e.detail,
                code=e.status_code
            )
            return envelope.to_json_response()

        except ValueError as e:
            # Validation errors
            envelope = ResponseEnvelope.error(
                message=f"Validation error: {str(e)}",
                code=422
            )
            return envelope.to_json_response()

        except Exception as e:
            # Unexpected errors
            envelope = ResponseEnvelope.error(
                message=f"Internal server error: {str(e)}",
                code=500
            )
            return envelope.to_json_response()

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            # Execute the controller method
            result = func(*args, **kwargs)

            # Wrap in success envelope
            envelope = ResponseEnvelope.success(data=result)
            return envelope.to_json_response()

        except HTTPException as e:
            # FastAPI HTTPException - preserve status code
            envelope = ResponseEnvelope.error(
                message=e.detail,
                code=e.status_code
            )
            return envelope.to_json_response()

        except ValueError as e:
            # Validation errors
            envelope = ResponseEnvelope.error(
                message=f"Validation error: {str(e)}",
                code=422
            )
            return envelope.to_json_response()

        except Exception as e:
            # Unexpected errors
            envelope = ResponseEnvelope.error(
                message=f"Internal server error: {str(e)}",
                code=500
            )
            return envelope.to_json_response()

    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

"""
DAF SDK Exceptions

Custom exception classes for handling API errors and SDK-specific issues.
"""

from typing import Any, Dict, Optional


class DAFError(Exception):
    """Base exception for all DAF SDK errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class APIError(DAFError):
    """
    Raised when the API returns a non-success status code (4xx or 5xx).

    Attributes:
        status_code: HTTP status code from the response
        message: Error message
        body: Raw response body
        headers: Response headers
    """

    def __init__(
        self,
        message: str,
        status_code: int,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}
        super().__init__(message)

    def __str__(self) -> str:
        return f"APIError(status_code={self.status_code}): {self.message}"


class BadRequestError(APIError):
    """Raised for 400 Bad Request responses."""

    def __init__(
        self, message: str, body: Optional[Any] = None, headers: Optional[Dict[str, str]] = None
    ):
        super().__init__(message, status_code=400, body=body, headers=headers)


class AuthenticationError(APIError):
    """Raised for 401 Unauthorized responses."""

    def __init__(
        self, message: str, body: Optional[Any] = None, headers: Optional[Dict[str, str]] = None
    ):
        super().__init__(message, status_code=401, body=body, headers=headers)


class PermissionDeniedError(APIError):
    """Raised for 403 Forbidden responses."""

    def __init__(
        self, message: str, body: Optional[Any] = None, headers: Optional[Dict[str, str]] = None
    ):
        super().__init__(message, status_code=403, body=body, headers=headers)


class NotFoundError(APIError):
    """Raised for 404 Not Found responses."""

    def __init__(
        self, message: str, body: Optional[Any] = None, headers: Optional[Dict[str, str]] = None
    ):
        super().__init__(message, status_code=404, body=body, headers=headers)


class ConflictError(APIError):
    """Raised for 409 Conflict responses."""

    def __init__(
        self, message: str, body: Optional[Any] = None, headers: Optional[Dict[str, str]] = None
    ):
        super().__init__(message, status_code=409, body=body, headers=headers)


class RateLimitError(APIError):
    """Raised for 429 Too Many Requests responses."""

    def __init__(
        self, message: str, body: Optional[Any] = None, headers: Optional[Dict[str, str]] = None
    ):
        super().__init__(message, status_code=429, body=body, headers=headers)


class InternalServerError(APIError):
    """Raised for 500+ Internal Server Error responses."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(message, status_code=status_code, body=body, headers=headers)


class ConnectionError(DAFError):
    """Raised when unable to connect to the DAF server."""

    pass


class TimeoutError(DAFError):
    """Raised when a request times out."""

    pass


class ValidationError(DAFError):
    """Raised when input validation fails."""

    pass


class StreamError(DAFError):
    """Raised when streaming response encounters an error."""

    pass


def raise_for_status(
    status_code: int,
    message: str,
    body: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
) -> None:
    """
    Raise appropriate exception based on HTTP status code.

    Args:
        status_code: HTTP status code
        message: Error message
        body: Response body
        headers: Response headers
    """
    if status_code < 400:
        return

    error_map = {
        400: BadRequestError,
        401: AuthenticationError,
        403: PermissionDeniedError,
        404: NotFoundError,
        409: ConflictError,
        429: RateLimitError,
    }

    if status_code in error_map:
        raise error_map[status_code](message, body=body, headers=headers)
    elif status_code >= 500:
        raise InternalServerError(message, status_code=status_code, body=body, headers=headers)
    else:
        raise APIError(message, status_code=status_code, body=body, headers=headers)

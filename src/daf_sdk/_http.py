"""
DAF SDK HTTP Client

Low-level HTTP client wrapper with authentication, retry logic, and streaming support.
"""

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, Generator, Optional
from urllib.parse import urljoin

import httpx

from .exceptions import APIError, ConnectionError, TimeoutError, raise_for_status
from .models import RawResponse, RequestOptions


class HTTPClient:
    """
    Synchronous HTTP client for DAF API.

    Handles request/response processing, authentication, retries, and error handling.
    """

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._default_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **(headers or {}),
        }
        if token:
            self._default_headers["Authorization"] = f"Bearer {token}"
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        """Lazy initialization of httpx client."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout, headers=self._default_headers)
        return self._client

    def set_auth_token(self, token: str) -> None:
        """Set authentication token."""
        self._default_headers["Authorization"] = f"Bearer {token}"
        # Update existing client if any
        if self._client is not None:
            self._client.headers["Authorization"] = f"Bearer {token}"

    def clear_auth_token(self) -> None:
        """Clear authentication token."""
        self._default_headers.pop("Authorization", None)
        if self._client is not None:
            self._client.headers.pop("Authorization", None)

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "HTTPClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _build_url(self, path: str) -> str:
        """Build full URL from path."""
        if path.startswith("http"):
            return path
        return urljoin(self.base_url + "/", path.lstrip("/"))

    def _merge_headers(self, headers: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Merge custom headers with defaults."""
        merged = self._default_headers.copy()
        if headers:
            merged.update(headers)
        return merged

    def _handle_response(self, response: httpx.Response) -> Any:
        """Process response and handle errors."""
        headers = dict(response.headers)

        try:
            body = response.json() if response.content else None
        except json.JSONDecodeError:
            body = response.text

        if response.status_code >= 400:
            message = body.get("detail", str(body)) if isinstance(body, dict) else str(body)
            raise_for_status(response.status_code, message, body=body, headers=headers)

        return body

    def request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        files: Optional[Dict[str, Any]] = None,
        options: Optional[RequestOptions] = None,
    ) -> Any:
        """
        Make an HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            path: API endpoint path
            json_data: JSON body data
            params: Query parameters
            headers: Additional headers
            files: Files to upload
            options: Request options (timeout, retries)

        Returns:
            Parsed JSON response
        """
        # DEBUG ALWAYS
        if json_data:
            print(f"!!!SDK REQUEST: method={method}, path={path}")
            print(f"!!!SDK REQUEST: has json_data={json_data is not None}")
            if "api_key" in json_data:
                print("!!!SDK REQUEST: HAS API_KEY!")
                print(f"!!!SDK REQUEST: api_key length={len(json_data['api_key'])}")

        url = self._build_url(path)
        merged_headers = self._merge_headers(headers)

        timeout = options.timeout if options else self.timeout
        max_retries = options.max_retries if options else self.max_retries

        if options and options.additional_headers:
            merged_headers.update(options.additional_headers)

        # Remove Content-Type for file uploads
        if files:
            merged_headers.pop("Content-Type", None)

        last_exception: Optional[Exception] = None
        retries = max_retries or 0
        for attempt in range(retries + 1):
            try:
                response = self.client.request(
                    method=method,
                    url=url,
                    json=json_data,
                    params=params,
                    headers=merged_headers,
                    files=files,
                    timeout=timeout,
                )
                return self._handle_response(response)

            except httpx.ConnectError as e:
                last_exception = ConnectionError(f"Failed to connect to {url}: {e}")
            except httpx.TimeoutException as e:
                last_exception = TimeoutError(f"Request timed out: {e}")
            except APIError:
                raise
            except Exception as e:
                last_exception = APIError(str(e), status_code=0)

            if attempt < retries:
                import time

                time.sleep(2**attempt * 0.5)

        if last_exception is not None:
            raise last_exception
        raise ConnectionError(f"Request to {url} failed after {retries + 1} attempts")

    def request_raw(self, method: str, path: str, **kwargs) -> RawResponse:
        """Make request and return raw response with headers."""
        url = self._build_url(path)
        merged_headers = self._merge_headers(kwargs.pop("headers", None))

        response = self.client.request(method=method, url=url, headers=merged_headers, **kwargs)

        try:
            data = response.json() if response.content else None
        except json.JSONDecodeError:
            data = response.text

        return RawResponse(
            data=data, headers=dict(response.headers), status_code=response.status_code
        )

    def stream(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 300.0,
    ) -> Generator[str, None, None]:
        """
        Make a streaming HTTP request.

        Yields:
            Chunks of the response as strings
        """
        url = self._build_url(path)
        merged_headers = self._merge_headers(headers)
        merged_headers["Accept"] = "text/event-stream"

        try:
            with self.client.stream(
                method=method,
                url=url,
                json=json_data,
                params=params,
                headers=merged_headers,
                timeout=timeout,
            ) as response:
                if response.status_code >= 400:
                    body = response.read()
                    try:
                        error_data = json.loads(body)
                        message = error_data.get("detail", str(error_data))
                    except Exception:
                        message = body.decode() if isinstance(body, bytes) else str(body)
                    raise_for_status(response.status_code, message)

                for line in response.iter_lines():
                    if line:
                        yield line

        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect: {e}")
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Stream timed out: {e}")

    # Convenience methods
    def get(self, path: str, **kwargs) -> Any:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, json_data: Optional[Dict] = None, **kwargs) -> Any:
        return self.request("POST", path, json_data=json_data, **kwargs)

    def put(self, path: str, json_data: Optional[Dict] = None, **kwargs) -> Any:
        return self.request("PUT", path, json_data=json_data, **kwargs)

    def patch(self, path: str, json_data: Optional[Dict] = None, **kwargs) -> Any:
        return self.request("PATCH", path, json_data=json_data, **kwargs)

    def delete(self, path: str, **kwargs) -> Any:
        return self.request("DELETE", path, **kwargs)


class AsyncHTTPClient:
    """
    Asynchronous HTTP client for DAF API.

    Provides async/await interface for all HTTP operations.
    """

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._default_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **(headers or {}),
        }
        if token:
            self._default_headers["Authorization"] = f"Bearer {token}"
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy initialization of async httpx client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout, headers=self._default_headers)
        return self._client

    def set_auth_token(self, token: str) -> None:
        """Set authentication token."""
        self._default_headers["Authorization"] = f"Bearer {token}"
        if self._client is not None:
            self._client.headers["Authorization"] = f"Bearer {token}"

    def clear_auth_token(self) -> None:
        """Clear authentication token."""
        self._default_headers.pop("Authorization", None)
        if self._client is not None:
            self._client.headers.pop("Authorization", None)

    async def close(self) -> None:
        """Close the async HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "AsyncHTTPClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    def _build_url(self, path: str) -> str:
        """Build full URL from path."""
        if path.startswith("http"):
            return path
        return urljoin(self.base_url + "/", path.lstrip("/"))

    def _merge_headers(self, headers: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Merge custom headers with defaults."""
        merged = self._default_headers.copy()
        if headers:
            merged.update(headers)
        return merged

    def _handle_response(self, response: httpx.Response) -> Any:
        """Process response and handle errors."""
        headers = dict(response.headers)

        try:
            body = response.json() if response.content else None
        except json.JSONDecodeError:
            body = response.text

        if response.status_code >= 400:
            message = body.get("detail", str(body)) if isinstance(body, dict) else str(body)
            raise_for_status(response.status_code, message, body=body, headers=headers)

        return body

    async def request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        files: Optional[Dict[str, Any]] = None,
        options: Optional[RequestOptions] = None,
    ) -> Any:
        """Make an async HTTP request with retry logic."""
        url = self._build_url(path)
        merged_headers = self._merge_headers(headers)

        timeout = options.timeout if options else self.timeout
        max_retries = options.max_retries if options else self.max_retries

        if options and options.additional_headers:
            merged_headers.update(options.additional_headers)

        if files:
            merged_headers.pop("Content-Type", None)

        last_exception: Optional[Exception] = None
        retries = max_retries or 0
        for attempt in range(retries + 1):
            try:
                response = await self.client.request(
                    method=method,
                    url=url,
                    json=json_data,
                    params=params,
                    headers=merged_headers,
                    files=files,
                    timeout=timeout,
                )
                return self._handle_response(response)

            except httpx.ConnectError as e:
                last_exception = ConnectionError(f"Failed to connect to {url}: {e}")
            except httpx.TimeoutException as e:
                last_exception = TimeoutError(f"Request timed out: {e}")
            except APIError:
                raise
            except Exception as e:
                last_exception = APIError(str(e), status_code=0)

            if attempt < retries:
                await asyncio.sleep(2**attempt * 0.5)

        if last_exception is not None:
            raise last_exception
        raise ConnectionError(f"Request to {url} failed after {retries + 1} attempts")

    async def request_raw(self, method: str, path: str, **kwargs) -> RawResponse:
        """Make async request and return raw response with headers."""
        url = self._build_url(path)
        merged_headers = self._merge_headers(kwargs.pop("headers", None))

        response = await self.client.request(
            method=method, url=url, headers=merged_headers, **kwargs
        )

        try:
            data = response.json() if response.content else None
        except json.JSONDecodeError:
            data = response.text

        return RawResponse(
            data=data, headers=dict(response.headers), status_code=response.status_code
        )

    async def stream(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 300.0,
    ) -> AsyncGenerator[str, None]:
        """Make an async streaming HTTP request."""
        url = self._build_url(path)
        merged_headers = self._merge_headers(headers)
        merged_headers["Accept"] = "text/event-stream"

        try:
            async with self.client.stream(
                method=method,
                url=url,
                json=json_data,
                params=params,
                headers=merged_headers,
                timeout=timeout,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    try:
                        error_data = json.loads(body)
                        message = error_data.get("detail", str(error_data))
                    except Exception:
                        message = body.decode() if isinstance(body, bytes) else str(body)
                    raise_for_status(response.status_code, message)

                async for line in response.aiter_lines():
                    if line:
                        yield line

        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect: {e}")
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Stream timed out: {e}")

    # Convenience methods
    async def get(self, path: str, **kwargs) -> Any:
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, json_data: Optional[Dict] = None, **kwargs) -> Any:
        return await self.request("POST", path, json_data=json_data, **kwargs)

    async def put(self, path: str, json_data: Optional[Dict] = None, **kwargs) -> Any:
        return await self.request("PUT", path, json_data=json_data, **kwargs)

    async def patch(self, path: str, json_data: Optional[Dict] = None, **kwargs) -> Any:
        return await self.request("PATCH", path, json_data=json_data, **kwargs)

    async def delete(self, path: str, **kwargs) -> Any:
        return await self.request("DELETE", path, **kwargs)

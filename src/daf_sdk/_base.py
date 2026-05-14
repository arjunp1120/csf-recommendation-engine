"""
DAF SDK Base Classes

Base classes for API resources providing common functionality.
"""

from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, Type, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from ._http import AsyncHTTPClient, HTTPClient

T = TypeVar("T", bound=BaseModel)


class SyncResource:
    """
    Base class for synchronous API resources.

    Provides common methods for CRUD operations and request handling.
    """

    def __init__(self, client: "HTTPClient"):
        self._client = client

    def _get(self, path: str, **kwargs) -> Any:
        """Make a GET request."""
        return self._client.get(path, **kwargs)

    def _post(self, path: str, json_data: Optional[Dict] = None, **kwargs) -> Any:
        """Make a POST request."""
        return self._client.post(path, json_data=json_data, **kwargs)

    def _put(self, path: str, json_data: Optional[Dict] = None, **kwargs) -> Any:
        """Make a PUT request."""
        return self._client.put(path, json_data=json_data, **kwargs)

    def _patch(self, path: str, json_data: Optional[Dict] = None, **kwargs) -> Any:
        """Make a PATCH request."""
        return self._client.patch(path, json_data=json_data, **kwargs)

    def _delete(self, path: str, **kwargs) -> Any:
        """Make a DELETE request."""
        return self._client.delete(path, **kwargs)

    def _request(self, method: str, path: str, **kwargs) -> Any:
        """Make a generic request."""
        return self._client.request(method, path, **kwargs)

    def _stream(self, method: str, path: str, **kwargs):
        """Make a streaming request."""
        return self._client.stream(method, path, **kwargs)


class AsyncResource:
    """
    Base class for asynchronous API resources.

    Provides async methods for CRUD operations and request handling.
    """

    def __init__(self, client: "AsyncHTTPClient"):
        self._client = client

    async def _get(self, path: str, **kwargs) -> Any:
        """Make an async GET request."""
        return await self._client.get(path, **kwargs)

    async def _post(self, path: str, json_data: Optional[Dict] = None, **kwargs) -> Any:
        """Make an async POST request."""
        return await self._client.post(path, json_data=json_data, **kwargs)

    async def _put(self, path: str, json_data: Optional[Dict] = None, **kwargs) -> Any:
        """Make an async PUT request."""
        return await self._client.put(path, json_data=json_data, **kwargs)

    async def _patch(self, path: str, json_data: Optional[Dict] = None, **kwargs) -> Any:
        """Make an async PATCH request."""
        return await self._client.patch(path, json_data=json_data, **kwargs)

    async def _delete(self, path: str, **kwargs) -> Any:
        """Make an async DELETE request."""
        return await self._client.delete(path, **kwargs)

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        """Make a generic async request."""
        return await self._client.request(method, path, **kwargs)

    def _stream(self, method: str, path: str, **kwargs):
        """Make an async streaming request."""
        return self._client.stream(method, path, **kwargs)


class ResourceCollection(Generic[T]):
    """
    Generic collection for list responses with pagination support.
    """

    def __init__(
        self, items: List[T], total: Optional[int] = None, skip: int = 0, limit: int = 100
    ):
        self.items = items
        self.total = total or len(items)
        self.skip = skip
        self.limit = limit

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    @property
    def has_more(self) -> bool:
        """Check if there are more items to fetch."""
        return self.skip + len(self.items) < self.total


def parse_response(data: Dict[str, Any], model: Type[T]) -> T:
    """
    Parse API response into a Pydantic model.

    Args:
        data: Raw response data
        model: Pydantic model class

    Returns:
        Parsed model instance
    """
    return model.model_validate(data)


def parse_list_response(data: List[Dict[str, Any]], model: Type[T]) -> List[T]:
    """
    Parse list API response into Pydantic models.

    Args:
        data: List of raw response data
        model: Pydantic model class

    Returns:
        List of parsed model instances
    """
    return [model.model_validate(item) for item in data]

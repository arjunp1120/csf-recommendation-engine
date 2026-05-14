"""LLM Endpoints resource for DAF SDK."""

from typing import Any, Dict, List, Optional

from .._base import AsyncResource, SyncResource
from ..models import LLMEndpoint


class LLMEndpoints(SyncResource):
    """
    LLM Endpoints - Saved LLM configuration management.

    Allows you to save and reuse LLM configurations (provider, model, API keys)
    across multiple agents and teams.
    """

    def create(
        self,
        name: str,
        provider_type: str,
        model_name: str,
        api_key: str,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        custom_base_url: Optional[str] = None,
        is_default: bool = False,
    ) -> LLMEndpoint:
        """
        Create a new LLM endpoint configuration.

        Args:
            name: Endpoint name
            provider_type: Provider (Azure, OpenAI, Anthropic, etc.)
            model_name: Model identifier
            api_key: API key for the provider
            azure_endpoint: Azure OpenAI endpoint URL (required for Azure)
            azure_deployment: Azure deployment name (required for Azure)
            api_version: Azure API version (optional)
            custom_base_url: Custom base URL for OpenAI-compatible APIs
            is_default: Set as default endpoint for this provider

        Returns:
            Created LLM endpoint

        Example:
            >>> endpoint = client.llm_endpoints.create(
            ...     name="Production Azure GPT-4o",
            ...     provider_type="Azure",
            ...     model_name="gpt-4o",
            ...     api_key="your-key",
            ...     azure_endpoint="https://your-resource.openai.azure.com",
            ...     azure_deployment="gpt-4o",
            ...     is_default=True
            ... )
        """
        data = {
            "name": name,
            "provider_type": provider_type,
            "model_name": model_name,
            "api_key": api_key,
            "is_default": is_default,
        }

        if azure_endpoint:
            data["azure_endpoint"] = azure_endpoint
        if azure_deployment:
            data["azure_deployment"] = azure_deployment
        if api_version:
            data["api_version"] = api_version
        if custom_base_url:
            data["custom_base_url"] = custom_base_url

        result = self._post("/api/llm-endpoints", json_data=data)
        return LLMEndpoint(**result)

    def list(self) -> List[LLMEndpoint]:
        """
        List all saved LLM endpoints.

        Returns:
            List of LLM endpoints

        Example:
            >>> endpoints = client.llm_endpoints.list()
            >>> for ep in endpoints:
            ...     print(f"{ep.name}: {ep.provider_type}/{ep.model_name}")
        """
        result = self._get("/api/llm-endpoints")
        return [LLMEndpoint(**ep) for ep in result]

    def get(self, endpoint_id: str) -> LLMEndpoint:
        """
        Get a specific LLM endpoint.

        Args:
            endpoint_id: Endpoint ID

        Returns:
            LLM endpoint

        Raises:
            NotFoundError: If endpoint not found

        Example:
            >>> endpoint = client.llm_endpoints.get("endpoint-id")
            >>> print(f"Model: {endpoint.model_name}")
        """
        result = self._get(f"/api/llm-endpoints/{endpoint_id}")
        return LLMEndpoint(**result)

    def update(
        self,
        endpoint_id: str,
        name: Optional[str] = None,
        provider_type: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        custom_base_url: Optional[str] = None,
        is_default: Optional[bool] = None,
    ) -> LLMEndpoint:
        """
        Update an LLM endpoint.

        Args:
            endpoint_id: Endpoint ID
            name: New name (optional)
            provider_type: New provider (optional)
            model_name: New model (optional)
            api_key: New API key (optional)
            azure_endpoint: New Azure endpoint (optional)
            azure_deployment: New Azure deployment (optional)
            api_version: New API version (optional)
            custom_base_url: New custom base URL (optional)
            is_default: Set as default (optional)

        Returns:
            Updated LLM endpoint

        Example:
            >>> # Update model
            >>> updated = client.llm_endpoints.update(
            ...     endpoint_id="ep-123",
            ...     model_name="gpt-4o-mini"
            ... )

            >>> # Set as default
            >>> updated = client.llm_endpoints.update(
            ...     endpoint_id="ep-123",
            ...     is_default=True
            ... )
        """
        data: Dict[str, Any] = {}

        if name is not None:
            data["name"] = name
        if provider_type is not None:
            data["provider_type"] = provider_type
        if model_name is not None:
            data["model_name"] = model_name
        if api_key is not None:
            data["api_key"] = api_key
        if azure_endpoint is not None:
            data["azure_endpoint"] = azure_endpoint
        if azure_deployment is not None:
            data["azure_deployment"] = azure_deployment
        if api_version is not None:
            data["api_version"] = api_version
        if custom_base_url is not None:
            data["custom_base_url"] = custom_base_url
        if is_default is not None:
            data["is_default"] = is_default

        result = self._patch(f"/api/llm-endpoints/{endpoint_id}", json_data=data)
        return LLMEndpoint(**result)

    def delete(self, endpoint_id: str) -> Dict[str, Any]:
        """
        Delete an LLM endpoint.

        Args:
            endpoint_id: Endpoint ID

        Returns:
            Deletion result

        Raises:
            NotFoundError: If endpoint not found

        Example:
            >>> client.llm_endpoints.delete("endpoint-id")
        """
        return self._delete(f"/api/llm-endpoints/{endpoint_id}")

    def get_providers(self) -> List[Dict[str, str]]:
        """
        List available LLM providers.

        Returns:
            List of providers with id and name

        Example:
            >>> providers = client.llm_endpoints.get_providers()
            >>> for p in providers:
            ...     print(f"{p['id']}: {p['name']}")
        """
        result = self._get("/api/llm-endpoints/providers")
        return result.get("providers", result)

    def get_models(self, provider: str, chat_only: bool = True) -> Dict[str, Any]:
        """
        List available models for a provider.

        Args:
            provider: Provider ID (openai, azure, anthropic, etc.)
            chat_only: Return only chat/completion models (default: True)

        Returns:
            Dict with provider, models list, and metadata

        Example:
            >>> result = client.llm_endpoints.get_models("openai")
            >>> for model in result["models"]:
            ...     print(model["id"])
        """
        params = {"chat_only": str(chat_only).lower()}
        return self._get(f"/api/llm-endpoints/models/{provider}", params=params)

    def get_model_params(self, provider: str, model_name: str) -> Dict[str, Any]:
        """
        Get parameters for a specific model.

        Args:
            provider: Provider ID
            model_name: Model name

        Returns:
            Dict with model parameters

        Example:
            >>> params = client.llm_endpoints.get_model_params("openai", "gpt-4o")
            >>> for p in params["params"]:
            ...     print(f"{p['key']}: {p.get('minValue')} - {p.get('maxValue')}")
        """
        return self._get(f"/api/llm-endpoints/models/{provider}/{model_name}/params")

    def test(self, endpoint_id: str) -> Dict[str, Any]:
        """
        Test a saved LLM endpoint connection.

        Args:
            endpoint_id: Endpoint ID

        Returns:
            Test result

        Example:
            >>> result = client.llm_endpoints.test("endpoint-id")
        """
        return self._post(f"/api/llm-endpoints/{endpoint_id}/test")

    def test_inline(
        self,
        provider_type: str,
        model_name: str,
        api_key: str,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        custom_base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Test an LLM configuration without saving it.

        Args:
            provider_type: Provider (Azure, OpenAI, etc.)
            model_name: Model identifier
            api_key: API key
            azure_endpoint: Azure endpoint URL (for Azure)
            azure_deployment: Azure deployment (for Azure)
            api_version: Azure API version (optional)
            custom_base_url: Custom base URL (optional)

        Returns:
            Test result

        Example:
            >>> result = client.llm_endpoints.test_inline(
            ...     provider_type="Azure",
            ...     model_name="gpt-4o",
            ...     api_key="key",
            ...     azure_endpoint="https://...",
            ...     azure_deployment="gpt-4o"
            ... )
        """
        data = {"provider_type": provider_type, "model_name": model_name, "api_key": api_key}

        if azure_endpoint:
            data["azure_endpoint"] = azure_endpoint
        if azure_deployment:
            data["azure_deployment"] = azure_deployment
        if api_version:
            data["api_version"] = api_version
        if custom_base_url:
            data["custom_base_url"] = custom_base_url

        return self._post("/api/llm-endpoints/test-inline", json_data=data)

    def clear_cache(self) -> Dict[str, Any]:
        """
        Clear the LLM model cache.

        Returns:
            Cache clear result

        Example:
            >>> client.llm_endpoints.clear_cache()
        """
        return self._delete("/api/llm-endpoints/cache")


class AsyncLLMEndpoints(AsyncResource):
    """Async version of LLMEndpoints."""

    async def create(
        self,
        name: str,
        provider_type: str,
        model_name: str,
        api_key: str,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        custom_base_url: Optional[str] = None,
        is_default: bool = False,
    ) -> LLMEndpoint:
        """Async create LLM endpoint."""
        data = {
            "name": name,
            "provider_type": provider_type,
            "model_name": model_name,
            "api_key": api_key,
            "is_default": is_default,
        }

        if azure_endpoint:
            data["azure_endpoint"] = azure_endpoint
        if azure_deployment:
            data["azure_deployment"] = azure_deployment
        if api_version:
            data["api_version"] = api_version
        if custom_base_url:
            data["custom_base_url"] = custom_base_url

        result = await self._post("/api/llm-endpoints", json_data=data)
        return LLMEndpoint(**result)

    async def list(self) -> List[LLMEndpoint]:
        """Async list endpoints."""
        result = await self._get("/api/llm-endpoints")
        return [LLMEndpoint(**ep) for ep in result]

    async def get(self, endpoint_id: str) -> LLMEndpoint:
        """Async get endpoint."""
        result = await self._get(f"/api/llm-endpoints/{endpoint_id}")
        return LLMEndpoint(**result)

    async def update(
        self,
        endpoint_id: str,
        name: Optional[str] = None,
        provider_type: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        custom_base_url: Optional[str] = None,
        is_default: Optional[bool] = None,
    ) -> LLMEndpoint:
        """Async update endpoint."""
        data: Dict[str, Any] = {}

        if name is not None:
            data["name"] = name
        if provider_type is not None:
            data["provider_type"] = provider_type
        if model_name is not None:
            data["model_name"] = model_name
        if api_key is not None:
            data["api_key"] = api_key
        if azure_endpoint is not None:
            data["azure_endpoint"] = azure_endpoint
        if azure_deployment is not None:
            data["azure_deployment"] = azure_deployment
        if api_version is not None:
            data["api_version"] = api_version
        if custom_base_url is not None:
            data["custom_base_url"] = custom_base_url
        if is_default is not None:
            data["is_default"] = is_default

        result = await self._patch(f"/api/llm-endpoints/{endpoint_id}", json_data=data)
        return LLMEndpoint(**result)

    async def delete(self, endpoint_id: str) -> Dict[str, Any]:
        """Async delete endpoint."""
        return await self._delete(f"/api/llm-endpoints/{endpoint_id}")

    async def get_providers(self) -> List[Dict[str, str]]:
        """Async get providers."""
        result = await self._get("/api/llm-endpoints/providers")
        return result.get("providers", result)

    async def get_models(self, provider: str, chat_only: bool = True) -> Dict[str, Any]:
        """Async get models."""
        params = {"chat_only": str(chat_only).lower()}
        return await self._get(f"/api/llm-endpoints/models/{provider}", params=params)

    async def get_model_params(self, provider: str, model_name: str) -> Dict[str, Any]:
        """Async get model params."""
        return await self._get(f"/api/llm-endpoints/models/{provider}/{model_name}/params")

    async def test(self, endpoint_id: str) -> Dict[str, Any]:
        """Async test endpoint."""
        return await self._post(f"/api/llm-endpoints/{endpoint_id}/test")

    async def test_inline(
        self,
        provider_type: str,
        model_name: str,
        api_key: str,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        custom_base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async test inline."""
        data = {"provider_type": provider_type, "model_name": model_name, "api_key": api_key}

        if azure_endpoint:
            data["azure_endpoint"] = azure_endpoint
        if azure_deployment:
            data["azure_deployment"] = azure_deployment
        if api_version:
            data["api_version"] = api_version
        if custom_base_url:
            data["custom_base_url"] = custom_base_url

        return await self._post("/api/llm-endpoints/test-inline", json_data=data)

    async def clear_cache(self) -> Dict[str, Any]:
        """Async clear cache."""
        return await self._delete("/api/llm-endpoints/cache")

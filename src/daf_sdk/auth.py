"""
DAF SDK Authentication Resource

Handles login, registration, and authentication status.
"""

from typing import TYPE_CHECKING, Optional

from ._base import AsyncResource, SyncResource, parse_response
from .models import AuthStatusResponse, TokenResponse, UserResponse

if TYPE_CHECKING:
    from ._http import AsyncHTTPClient, HTTPClient


class Auth(SyncResource):
    """
    Resource for authentication operations.

    Usage:
        # Login
        client.auth.login(email="user@example.com", password="secret")

        # Get current user
        user = client.auth.me()

        # Check auth status
        status = client.auth.status()

        # Logout
        client.auth.logout()
    """

    def __init__(self, client: "HTTPClient", parent_client=None):
        super().__init__(client)
        self._parent_client = parent_client

    def login(self, email: str, password: str) -> TokenResponse:
        """
        Login with email and password.

        Args:
            email: User email
            password: User password

        Returns:
            TokenResponse with access_token
        """
        data = self._post("/api/auth/login", json_data={"email": email, "password": password})
        token_response = parse_response(data, TokenResponse)

        # Automatically set token in client
        self._client.set_auth_token(token_response.access_token)

        return token_response

    def register(
        self, email: str, password: str, display_name: Optional[str] = None
    ) -> TokenResponse:
        """
        Register a new user account.

        Args:
            email: User email
            password: User password
            display_name: Optional display name

        Returns:
            TokenResponse with access_token
        """
        payload = {"email": email, "password": password}
        if display_name:
            payload["display_name"] = display_name

        data = self._post("/api/auth/register", json_data=payload)
        token_response = parse_response(data, TokenResponse)

        # Automatically set token in client
        self._client.set_auth_token(token_response.access_token)

        return token_response

    def logout(self) -> dict:
        """
        Logout current user.

        Clears the authentication token from the client.

        Returns:
            Confirmation message
        """
        result = self._post("/api/auth/logout")
        self._client.clear_auth_token()
        return result

    def me(self) -> UserResponse:
        """
        Get current authenticated user.

        Returns:
            UserResponse with user details
        """
        data = self._get("/api/auth/me")
        return parse_response(data, UserResponse)

    def status(self) -> AuthStatusResponse:
        """
        Get authentication system status.

        Returns:
            AuthStatusResponse with auth configuration
        """
        data = self._get("/api/auth/status")
        return parse_response(data, AuthStatusResponse)

    def change_password(self, current_password: str, new_password: str) -> dict:
        """
        Change current user's password.

        Args:
            current_password: Current password
            new_password: New password

        Returns:
            Confirmation message
        """
        return self._post(
            "/api/auth/change-password",
            json_data={"current_password": current_password, "new_password": new_password},
        )

    def update_profile(
        self, display_name: Optional[str] = None, email: Optional[str] = None
    ) -> UserResponse:
        """
        Update current user's profile.

        Args:
            display_name: New display name
            email: New email

        Returns:
            Updated UserResponse
        """
        payload = {}
        if display_name is not None:
            payload["display_name"] = display_name
        if email is not None:
            payload["email"] = email

        data = self._put("/api/auth/me", json_data=payload)
        return parse_response(data, UserResponse)


class AsyncAuth(AsyncResource):
    """
    Async resource for authentication operations.
    """

    def __init__(self, client: "AsyncHTTPClient", parent_client=None):
        super().__init__(client)
        self._parent_client = parent_client

    async def login(self, email: str, password: str) -> TokenResponse:
        """Login with email and password."""
        data = await self._post("/api/auth/login", json_data={"email": email, "password": password})
        token_response = parse_response(data, TokenResponse)
        self._client.set_auth_token(token_response.access_token)
        return token_response

    async def register(
        self, email: str, password: str, display_name: Optional[str] = None
    ) -> TokenResponse:
        """Register a new user account."""
        payload = {"email": email, "password": password}
        if display_name:
            payload["display_name"] = display_name

        data = await self._post("/api/auth/register", json_data=payload)
        token_response = parse_response(data, TokenResponse)
        self._client.set_auth_token(token_response.access_token)
        return token_response

    async def logout(self) -> dict:
        """Logout current user."""
        result = await self._post("/api/auth/logout")
        self._client.clear_auth_token()
        return result

    async def me(self) -> UserResponse:
        """Get current authenticated user."""
        data = await self._get("/api/auth/me")
        return parse_response(data, UserResponse)

    async def status(self) -> AuthStatusResponse:
        """Get authentication system status."""
        data = await self._get("/api/auth/status")
        return parse_response(data, AuthStatusResponse)

    async def change_password(self, current_password: str, new_password: str) -> dict:
        """Change current user's password."""
        return await self._post(
            "/api/auth/change-password",
            json_data={"current_password": current_password, "new_password": new_password},
        )

    async def update_profile(
        self, display_name: Optional[str] = None, email: Optional[str] = None
    ) -> UserResponse:
        """Update current user's profile."""
        payload = {}
        if display_name is not None:
            payload["display_name"] = display_name
        if email is not None:
            payload["email"] = email

        data = await self._put("/api/auth/me", json_data=payload)
        return parse_response(data, UserResponse)

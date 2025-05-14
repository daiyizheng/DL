#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :mcp_package_test.py
@Description  : 对MCP包的源码调试
@Time         :2025/05/14 08:40:14
@Author       :flow-laic
@Version      :1.0
'''
import sys
sys.path.insert(0, "/Users/a1-6/Documents/projects/DL/08-llms/MCP_/package_")
import time
import secrets
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    construct_redirect_uri,
)

from mcp.server.auth.routes import (
    ClientRegistrationOptions,
    RevocationOptions,
    create_auth_routes,
)
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthToken,
)

## 验证
# 想要公开访问受保护资源的工具的服务器可以使用身份验证。
# mcp.server.auth实现 OAuth 2.0 服务器接口，服务器可以通过提供协议的实现来使用该OAuthServerProvider接口。

from mcp.server.auth.settings  import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions


class MyOAuthServerProvider(OAuthAuthorizationServerProvider):
    def __init__(self):
        self.clients = {}
        self.auth_codes = {}  # code -> {client_id, code_challenge, redirect_uri}
        self.tokens = {}  # token -> {client_id, scopes, expires_at}
        self.refresh_tokens = {}  # refresh_token -> access_token

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self.clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull):
        self.clients[client_info.client_id] = client_info

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        # toy authorize implementation which just immediately generates an authorization
        # code and completes the redirect
        code = AuthorizationCode(
            code=f"code_{int(time.time())}",
            client_id=client.client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            expires_at=time.time() + 300,
            scopes=params.scopes or ["read", "write"],
        )
        self.auth_codes[code.code] = code

        return construct_redirect_uri(
            str(params.redirect_uri), code=code.code, state=params.state
        )

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        return self.auth_codes.get(authorization_code)

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        assert authorization_code.code in self.auth_codes

        # Generate an access token and refresh token
        access_token = f"access_{secrets.token_hex(32)}"
        refresh_token = f"refresh_{secrets.token_hex(32)}"

        # Store the tokens
        self.tokens[access_token] = AccessToken(
            token=access_token,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=int(time.time()) + 3600,
        )

        self.refresh_tokens[refresh_token] = access_token

        # Remove the used code
        del self.auth_codes[authorization_code.code]

        return OAuthToken(
            access_token=access_token,
            token_type="bearer",
            expires_in=3600,
            scope="read write",
            refresh_token=refresh_token,
        )

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        old_access_token = self.refresh_tokens.get(refresh_token)
        if old_access_token is None:
            return None
        token_info = self.tokens.get(old_access_token)
        if token_info is None:
            return None

        # Create a RefreshToken object that matches what is expected in later code
        refresh_obj = RefreshToken(
            token=refresh_token,
            client_id=token_info.client_id,
            scopes=token_info.scopes,
            expires_at=token_info.expires_at,
        )

        return refresh_obj

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        # Check if refresh token exists
        assert refresh_token.token in self.refresh_tokens

        old_access_token = self.refresh_tokens[refresh_token.token]

        # Check if the access token exists
        assert old_access_token in self.tokens

        # Check if the token was issued to this client
        token_info = self.tokens[old_access_token]
        assert token_info.client_id == client.client_id

        # Generate a new access token and refresh token
        new_access_token = f"access_{secrets.token_hex(32)}"
        new_refresh_token = f"refresh_{secrets.token_hex(32)}"

        # Store the new tokens
        self.tokens[new_access_token] = AccessToken(
            token=new_access_token,
            client_id=client.client_id,
            scopes=scopes or token_info.scopes,
            expires_at=int(time.time()) + 3600,
        )

        self.refresh_tokens[new_refresh_token] = new_access_token

        # Remove the old tokens
        del self.refresh_tokens[refresh_token.token]
        del self.tokens[old_access_token]

        return OAuthToken(
            access_token=new_access_token,
            token_type="bearer",
            expires_in=3600,
            scope=" ".join(scopes) if scopes else " ".join(token_info.scopes),
            refresh_token=new_refresh_token,
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        token_info = self.tokens.get(token)

        # Check if token is expired
        # if token_info.expires_at < int(time.time()):
        #     raise InvalidTokenError("Access token has expired")

        return token_info and AccessToken(
            token=token,
            client_id=token_info.client_id,
            scopes=token_info.scopes,
            expires_at=token_info.expires_at,
        )

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        match token:
            case RefreshToken():
                # Remove the refresh token
                del self.refresh_tokens[token.token]

            case AccessToken():
                # Remove the access token
                del self.tokens[token.token]

                # Also remove any refresh tokens that point to this access token
                for refresh_token, access_token in list(self.refresh_tokens.items()):
                    if access_token == token.token:
                        del self.refresh_tokens[refresh_token]




mcp = FastMCP("My App",
        auth_server_provider=MyOAuthServerProvider(), ## 权限服务逻辑类
        auth=AuthSettings(
            issuer_url="https://myapp.com",
            revocation_options=RevocationOptions(
                enabled=True,
            ),
            client_registration_options=ClientRegistrationOptions(
                enabled=True,
                valid_scopes=["myscope", "myotherscope"],
                default_scopes=["myscope"],
            ),
            required_scopes=["myscope"],
        ),
)
# 带有 GitHub OAuth 身份验证的简单 MCP 服务器

这是一个使用 GitHub OAuth 身份验证的 MCP 服务器的简单示例。它演示了如何使用单个工具集成 OAuth 所需的基本组件。

这只是一个使用 auth 的服务器示例，官方的 GitHub mcp 服务器在[这里](https://github.com/github/github-mcp-server)


## 概述
这个简单的演示展示了如何设置服务器：

GitHub OAuth2 授权流程
单一工具：`get_user_profile`检索 GitHub 用户信息

## 先决条件
1. 创建 GitHub OAuth 应用程序：
- 转到 GitHub 设置 > 开发者设置 > OAuth 应用 > 新建 OAuth 应用
- 应用程序名称：任意名称（例如“Simple MCP Auth Demo”）
- 主页网址：`http://localhost:8000`
- 授权回调URL：`http://localhost:8000/github/callback`
- 点击“注册应用程序”
- 记下您的客户端 ID 和客户端密钥


## 必需的环境变量
运行服务器之前必须设置这些环境变量：
```shell
export MCP_GITHUB_GITHUB_CLIENT_ID="1290206"
export MCP_GITHUB_GITHUB_CLIENT_SECRET="Iv23liE88J3oe0LD4UMP"
```
如果没有正确设置这些环境变量，服务器将无法启动。

## 运行服务器
```bash
# Set environment variables first (see above)

# Run the server
uv run mcp-simple-auth
```
服务器将在 `http://localhost:8000` 上启动。

## 工具选择
该服务器支持可在同一端口上运行的多种传输协议：

### SSE（服务器发送事件）- 默认

```bash
uv run mcp-simple-auth
# or explicitly:
uv run mcp-simple-auth --transport sse
```

SSE 传输提供端点：
- /sse
### 可流式传输的 HTTP
```bash
uv run mcp-simple-auth --transport streamable-http
```
可流式传输的 HTTP 传输提供端点：
- /mcp

这确保了向后兼容性，而无需多个服务器实例。使用 SSE 传输 ( --transport sse) 时，只有/sse端点可用。\

## 可用工具
### 获取用户配置文件
此简单示例中的唯一工具。返回经过身份验证的用户的 GitHub 个人资料信息。
- 所需范围：user
- 返回：GitHub 用户个人资料数据，包括用户名、电子邮件、简历等。

## 故障排除
如果服务器启动失败，请检查：
1. 环境变量MCP_GITHUB_GITHUB_CLIENT_ID并MCP_GITHUB_GITHUB_CLIENT_SECRET设置
2. GitHub OAuth 应用回调 URL 匹配http://localhost:8000/github/callback
3. 没有其他服务正在使用端口 8000
4. 指定的传输有效（sse或streamable-http）

您可以使用Inspector来测试 Auth



整个源码的流程
AUTHORIZATION_PATH = "/authorize" 授权
TOKEN_PATH = "/token"   获得token
REGISTRATION_PATH = "/register" 注册
REVOCATION_PATH = "/revoke"  注销

第三方授权流程包括以下步骤：

- MCP 客户端向 MCP 服务器发起标准 OAuth 流程
- MCP 服务器将用户重定向到第三方授权服务器
- 用户向第三方服务器授权
- 第三方服务器使用授权码重定向回 MCP 服务器
- MCP 服务器交换第三方访问令牌的代码
- MCP 服务器生成自己的访问令牌并绑定到第三方会话
- MCP 服务器与 MCP 客户端完成原始 OAuth 流程




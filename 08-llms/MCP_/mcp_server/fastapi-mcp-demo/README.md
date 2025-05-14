# fastapi-mcp 实例教程
## FastAPI-MCP: 为AI模型赋能的零配置API工具
FastAPI-MCP是一个零配置工具，能够自动将FastAPI应用的所有端点转换为符合Model Context Protocol（MCP）规范的工具。它的主要特点包括：

零配置：只需几行代码即可集成，无需额外开发工作
自动发现：自动识别并转换FastAPI应用中的所有端点
完整兼容：保留原有API的请求模型、响应模型和文档
灵活部署：可以直接挂载到FastAPI应用中，也可以单独部署
MCP协议允许AI模型（如Claude、GPT等）调用外部工具来完成复杂任务，而FastAPI-MCP则简化了这些工具的创建过程，让开发者能够快速将现有服务暴露给AI模型使用。

## 安装与基本使用
### 安装
推荐使用uv（一个快速的Python包安装工具）进行安装：
```shell
uv init fastapi-mcp-demo
cd fastapi-mcp-demo
uv venv --python=3.10
source .venv/bin/activate


```
然后修改内容阿里源
export UV_INDEX_URL="xxx"


```shell
uv add fastapi-mcp
```

或者使用传统的pip安装：
```shell
pip install fastapi-mcp
```

### 基本使用
FastAPI-MCP的基本使用非常简单，只需几行代码即可完成：
```python
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

app = FastAPI()

mcp = FastApiMCP(app)

# 直接将 MCP 服务器挂载到您的 FastAPI 应用
mcp.mount()

```
完成上述配置后，一个自动生成的MCP服务器将在/mcp路径上可用。值得注意的是，虽然base_url参数是可选的，但强烈建议提供，因为它告诉MCP服务器在调用工具时要发送API请求的位置。


## 工具命名

FastAPI-MCP 使用 FastAPI 路由中的`operation_id`作为 MCP 工具的名称。如果您不指定`operation_id`，FastAPI 会自动生成一个，但这些名称可能比较晦涩。

比较以下两个端点定义：

```python
# 自动生成的 operation_id（类似于 "read_user_users__user_id__get"）
@app.get("/users/{user_id}")
async def read_user(user_id: int):
    return {"user_id": user_id}

# 显式 operation_id（工具将被命名为 "get_user_info"）
@app.get("/users/{user_id}", operation_id="get_user_info")
async def read_user(user_id: int):
    return {"user_id": user_id}
```

为了获得更清晰、更直观的工具名称，我们建议在 FastAPI 路由定义中添加显式的`operation_id`参数。

要了解更多信息，请阅读 FastAPI 官方文档中关于 [路径操作的高级配置](https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/) 的部分。

## 高级用法

FastAPI-MCP 提供了多种方式来自定义和控制 MCP 服务器的创建和配置。以下是一些高级用法模式：

### 自定义模式描述

```python
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

app = FastAPI()

mcp = FastApiMCP(
    app,
    name="我的 API MCP",
    describe_all_responses=True,     # 在工具描述中包含所有可能的响应模式
    describe_full_response_schema=True  # 在工具描述中包含完整的 JSON 模式
)

mcp.mount()
```

### 自定义公开的端点

您可以使用 Open API 操作 ID 或标签来控制哪些 FastAPI 端点暴露为 MCP 工具：

```python
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

app = FastAPI()

# 仅包含特定操作
mcp = FastApiMCP(
    app,
    include_operations=["get_user", "create_user"]
)

# 排除特定操作
mcp = FastApiMCP(
    app,
    exclude_operations=["delete_user"]
)

# 仅包含具有特定标签的操作
mcp = FastApiMCP(
    app,
    include_tags=["users", "public"]
)

# 排除具有特定标签的操作
mcp = FastApiMCP(
    app,
    exclude_tags=["admin", "internal"]
)

# 结合操作 ID 和标签（包含模式）
mcp = FastApiMCP(
    app,
    include_operations=["user_login"],
    include_tags=["public"]
)

mcp.mount()
```

关于过滤的注意事项：
- 您不能同时使用`include_operations`和`exclude_operations`
- 您不能同时使用`include_tags`和`exclude_tags`
- 您可以将操作过滤与标签过滤结合使用（例如，使用`include_operations`和`include_tags`）
- 当结合过滤器时，将采取贪婪方法。匹配任一标准的端点都将被包含

### 与原始 FastAPI 应用分开部署

您不限于在创建 MCP 的同一个 FastAPI 应用上提供 MCP 服务。

您可以从一个 FastAPI 应用创建 MCP 服务器，并将其挂载到另一个应用上：

```python
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

# 您的 API 应用
api_app = FastAPI()
# ... 在 api_app 上定义您的 API 端点 ...

# 一个单独的 MCP 服务器应用
mcp_app = FastAPI()

# 从 API 应用创建 MCP 服务器
mcp = FastApiMCP(api_app)

# 将 MCP 服务器挂载到单独的应用
mcp.mount(mcp_app)

# 现在您可以分别运行两个应用：
# uvicorn main:api_app --host api-host --port 8001
# uvicorn main:mcp_app --host mcp-host --port 8000

## uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 在 MCP 服务器创建后添加端点

如果您在创建 MCP 服务器后向 FastAPI 应用添加端点，您需要刷新服务器以包含它们：

```python
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

app = FastAPI()
# ... 定义初始端点 ...

# 创建 MCP 服务器
mcp = FastApiMCP(app)
mcp.mount()

# 在 MCP 服务器创建后添加新端点
@app.get("/new/endpoint/", operation_id="new_endpoint")
async def new_endpoint():
    return {"message": "Hello, world!"}

# 刷新 MCP 服务器以包含新端点
mcp.setup_server()
```

### 与 FastAPI 应用的通信

FastAPI-MCP 默认使用 ASGI 传输，这意味着它直接与您的 FastAPI 应用通信，而不需要发送 HTTP 请求。这样更高效，也不需要基础 URL。

如果您需要指定自定义基础 URL 或使用不同的传输方法，您可以提供自己的 `httpx.AsyncClient`：

```python
import httpx
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

app = FastAPI()

# 使用带有特定基础 URL 的自定义 HTTP 客户端
custom_client = httpx.AsyncClient(
    base_url="https://api.example.com",
    timeout=30.0
)

mcp = FastApiMCP(
    app,
    http_client=custom_client
)

mcp.mount()
```

## 示例

请参阅 [examples](examples) 目录以获取完整示例。

## 使用 SSE 连接到 MCP 服务器

一旦您的集成了 MCP 的 FastAPI 应用运行，您可以使用任何支持 SSE 的 MCP 客户端连接到它，例如 Cursor：

1. 运行您的应用。

2. 在 Cursor -> 设置 -> MCP 中，使用您的 MCP 服务器端点的URL（例如，`http://localhost:8000/mcp`）作为 sse。

3. Cursor 将自动发现所有可用的工具和资源。

## 使用 [mcp-proxy stdio](https://github.com/sparfenyuk/mcp-proxy?tab=readme-ov-file#1-stdio-to-sse) 连接到 MCP 服务器

如果您的 MCP 客户端不支持 SSE，例如 Claude Desktop：

1. 运行您的应用。

2. 安装 [mcp-proxy](https://github.com/sparfenyuk/mcp-proxy?tab=readme-ov-file#installing-via-pypi)，例如：`uv tool install mcp-proxy`。

3. 在 Claude Desktop 的 MCP 配置文件（`claude_desktop_config.json`）中添加：

在 Windows 上：
```json
{
  "mcpServers": {
    "my-api-mcp-proxy": {
        "command": "mcp-proxy",
        "args": ["http://127.0.0.1:8000/mcp"]
    }
  }
}
```
在 MacOS 上：
```json
{
  "mcpServers": {
    "my-api-mcp-proxy": {
        "command": "/Full/Path/To/Your/Executable/mcp-proxy",
        "args": ["http://127.0.0.1:8000/mcp"]
    }
  }
}
```
通过在终端运行`which mcp-proxy`来找到 mcp-proxy 的路径。

4. Claude Desktop 将自动发现所有可用的工具和资源

### 在DIFY中使用
安装插件 MCP SEE 
```shell
{
   "my-api-mcp": {    
         "url": "http://192.16.19.229:8000/mcp", 
         "headers": {},  
         "timeout": 5,       
         "sse_read_timeout": 300     
    } 
```

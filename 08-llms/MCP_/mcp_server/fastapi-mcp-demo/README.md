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
然后修改内容


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


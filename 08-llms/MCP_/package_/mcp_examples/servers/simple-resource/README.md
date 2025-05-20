# MCP 简单资源
一个简单的 MCP 服务器，将示例文本文件作为资源公开。

## 用法

使用 stdio（默认）或 SSE 传输启动服务器：

```bash
# Using stdio transport (default)
uv run mcp-simple-resource

# Using SSE transport on custom port
uv run mcp-simple-resource --transport sse --port 8000
```

服务器公开一些可供客户端读取的基本文本文件资源。

## 例子
使用 MCP 客户端，您可以使用 STDIO 传输检索如下资源：

```python
import asyncio
from mcp.types import AnyUrl
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main():
    async with stdio_client(
        StdioServerParameters(command="uv", args=["run", "mcp-simple-resource"])
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available resources
            resources = await session.list_resources()
            print(resources)

            # Get a specific resource
            resource = await session.read_resource(AnyUrl("file:///greeting.txt"))
            print(resource)


asyncio.run(main())

```

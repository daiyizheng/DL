# MCP 简单提示
一个简单的 MCP 服务器，它公开一个可定制的提示模板，其中包含可选的上下文和主题参数。

## 用法
使用 stdio（默认）或 SSE 传输启动服务器：
```bash
# Using stdio transport (default)
uv run mcp-simple-prompt

# Using SSE transport on custom port
uv run mcp-simple-prompt --transport sse --port 800
```

服务器公开一个名为`simple`的提示，它接受两个可选参数：
- context：需要考虑的其他背景信息
- topic：需要关注的具体主题

## 例子
使用 MCP 客户端，您可以使用 STDIO 传输检索如下提示：

```python
import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main():
    async with stdio_client(
        StdioServerParameters(command="uv", args=["run", "mcp-simple-prompt"])
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available prompts
            prompts = await session.list_prompts()
            print(prompts)

            # Get the prompt with arguments
            prompt = await session.get_prompt(
                "simple",
                {
                    "context": "User is a software developer",
                    "topic": "Python async programming",
                },
            )
            print(prompt)


asyncio.run(main())
```

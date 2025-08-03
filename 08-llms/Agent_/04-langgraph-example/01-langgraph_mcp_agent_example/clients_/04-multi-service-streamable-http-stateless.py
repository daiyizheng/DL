from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import asyncio
from dotenv import load_dotenv
load_dotenv()
# 使用流式 HTTP 协议的多服务客户端
# 该客户端连接到一个运行在 http://localhost:3000/mcp 的 MCP 服务器
# 该服务器提供了一个数学服务，可以处理数学计算请求

async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "streamable_http",
                "url": "http://localhost:3000/mcp"
            },
        }
    )
    agent = create_react_agent("openai:gpt-4.1", await client.get_tools())
    math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})

    print("Math Agent response:", math_response)

if __name__ == '__main__':
    asyncio.run(main())
    print("所有操作完成！")
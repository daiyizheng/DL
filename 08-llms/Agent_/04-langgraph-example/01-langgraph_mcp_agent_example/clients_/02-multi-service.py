from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

import asyncio


async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["mcp_servers/math_server.py"],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
            })
    print("Available tools:", await client.get_tools())

    # agent = create_react_agent("openai:gpt-4.1", client.get_tools())
    # math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
    # print("Math Agent response:", math_response)
    # weather_response = await agent.ainvoke({"messages": "what is the weather in nyc?"})
    # print("Weather Agent response:", weather_response)

if __name__ == '__main__':
    
    asyncio.run(main())

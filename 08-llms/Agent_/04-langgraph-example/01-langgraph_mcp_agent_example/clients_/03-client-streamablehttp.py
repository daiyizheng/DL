# Use server from examples/servers/streamable-http-stateless/

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from dotenv import load_dotenv
load_dotenv()


async def main():
    async with streamablehttp_client("http://localhost:3000/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize() 

            # Get tools
            tools = await load_mcp_tools(session)
            agent = create_react_agent("openai:gpt-4.1", tools)
            math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})

            print("Math Agent response:", math_response)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())



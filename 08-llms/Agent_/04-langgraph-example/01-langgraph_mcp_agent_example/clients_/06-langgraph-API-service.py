from contextlib import asynccontextmanager
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4.1")

@asynccontextmanager
async def make_graph():
    client =  MultiServerMCPClient(
        {
            "math": {
                "command": "python", 
                # Make sure to update to the full absolute path to your math_server.py file
                "args": ["mcp_servers/math_server.py"],
                "transport": "stdio",
            },
            "weather": { 
                # make sure you start your weather server on port 8000
                "url": "http://localhost:8000/sse",  
                "transport": "sse",  
            }
        }
    )
    agent = create_react_agent(model, await client.get_tools()) 
    yield agent

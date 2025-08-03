from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """获取指定地点天气"""
    return "It's always sunny in New York"

if __name__ == "__main__":
    mcp.run(transport="sse")

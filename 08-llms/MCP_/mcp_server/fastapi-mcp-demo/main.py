from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

# Your existing FastAPI application
app = FastAPI()

# Define your endpoints as you normally would
@app.get("/items/{item_id}", operation_id="get_item")
async def read_item(item_id: int):
    return {"item_id": item_id, "name": f"Item {item_id}"}

# Add the MCP server to your FastAPI app
mcp = FastApiMCP(
    app,  
    name="My API MCP",  # Name for your MCP server
    description="MCP server for my API",  # Description
)

# Mount the MCP server to your FastAPI app
mcp.mount()

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

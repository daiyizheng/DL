#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :14_mcp_package_clients_with_stdio.py
@Description  :
@Time         :2025/05/14 16:26:44
@Author       :flow-laic
@Version      :1.0
'''

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

# 为stdio连接创建服务器参数
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["example_server.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)


# 可选：创建采样回调
async def handle_sampling_message(
    message: types.CreateMessageRequestParams,
) -> types.CreateMessageResult:
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(
            type="text",
            text="Hello, world! from model",
        ),
        model="gpt-3.5-turbo",
        stopReason="endTurn",
    )


async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write, sampling_callback=handle_sampling_message
        ) as session:
            # 初始化连接
            await session.initialize()

            # 列出可用提示
            prompts = await session.list_prompts()

            # 获得提示
            prompt = await session.get_prompt(
                "example-prompt", arguments={"arg1": "value"}
            )

            # 列出可用资源
            resources = await session.list_resources()

            # 列出可用工具
            tools = await session.list_tools()

            # 阅读资源
            content, mime_type = await session.read_resource("file://some/path")

            # 调用工具
            result = await session.call_tool("tool-name", arguments={"arg1": "value"})


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
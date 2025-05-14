#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :15_mcp_package_clients_with_streamablehttp.py
@Description  :
@Time         :2025/05/14 16:33:46
@Author       :flow-laic
@Version      :1.0
'''

from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession


async def main():
    # Connect to a streamable HTTP server
    async with streamablehttp_client("example/mcp") as (
        read_stream,
        write_stream,
        _,
    ):
        # Create a session using the client streams
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()
            # Call a tool
            tool_result = await session.call_tool("echo", {"message": "hello"})

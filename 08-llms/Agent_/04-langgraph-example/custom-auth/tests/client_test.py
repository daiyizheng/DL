#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :client_test.py
@Description  :
@Time         :2025/08/01 12:00:40
@Author       :flow-laic
@Version      :1.0
'''

from langgraph_sdk import get_client
import asyncio

async def main():
    # 尝试不带令牌访问（应失败）
    client = get_client(url="http://localhost:2024")
    try:
        thread = await client.threads.create()
        print("❌ 没有令牌时应该失败！")
    except Exception as e:
        print("✅ 正确阻止了访问：", e)

    # 使用有效令牌尝试
    client = get_client(
        url="http://localhost:2024", headers={"Authorization": "Bearer user1-token"}
    )

    # 创建线程并聊天
    thread = await client.threads.create()
    print(f"✅ 以 Alice 身份创建了线程：{thread['thread_id']}")

    response = await client.runs.create(
        thread_id=thread["thread_id"],
        assistant_id="agent",
        input={"messages": [{"role": "user", "content": "Hello!"}]},
    )
    print("✅ 机器人响应：")
    print(response)



if __name__ == '__main__':
    asyncio.run(main())
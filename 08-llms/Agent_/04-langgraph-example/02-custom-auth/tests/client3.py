#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :client3.py
@Description  :
@Time         :2025/08/01 16:06:46
@Author       :flow-laic
@Version      :1.0
'''

import asyncio
import os
import httpx
from langgraph_sdk import get_client


async def login(email: str, password: str):
    """为现有用户获取访问令牌。"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
            json={
                "email": email,
                "password": password
            },
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Content-Type": "application/json"
            },
        )
        assert response.status_code == 200
        return response.json()["access_token"]


async def main(email1, password, email2):
    # 以用户 1 身份登录
    user1_token = await login(email1, password)
    user1_client = get_client(
        url="http://localhost:2024", headers={"Authorization": f"Bearer {user1_token}"}
    )

    # 以用户 1 身份创建线程
    thread = await user1_client.threads.create()
    print(f"✅ 用户 1 创建了线程：{thread['thread_id']}")

    # 尝试不带令牌访问
    unauthenticated_client = get_client(url="http://localhost:2024")
    try:
        await unauthenticated_client.threads.create()
        print("❌ 未认证访问应该失败！")
    except Exception as e:
        print("✅ 未认证访问被阻止：", e)

    # 尝试以用户 2 身份访问用户 1 的线程
    user2_token = await login(email2, password)
    user2_client = get_client(
        url="http://localhost:2024", headers={"Authorization": f"Bearer {user2_token}"}
    )

    try:
        await user2_client.threads.get(thread["thread_id"])
        print("❌ 用户 2 不应该看到用户 1 的线程！")
    except Exception as e:
        print("✅ 用户 2 被阻止访问用户 1 的线程：", e)


if __name__ == '__main__':

    # 用户电子邮件和密码
    email1 = ""
    email2 = ""
    password = "secure-password"  # 请更改此密码

    asyncio.run(main(email1, password, email2))
    print("所有操作完成！")
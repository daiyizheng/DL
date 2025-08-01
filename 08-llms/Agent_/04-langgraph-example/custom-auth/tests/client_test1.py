import asyncio
from langgraph_sdk import get_client

async def main():
    from langgraph_sdk import get_client

    # 为两个用户创建客户端
    alice = get_client(
        url="http://localhost:2024",
        headers={"Authorization": "Bearer user1-token"}
    )

    print("alice.graphs.list()", dir(alice))

    bob = get_client(
        url="http://localhost:2024",
        headers={"Authorization": "Bearer user2-token"}
    )

    # Alice 创建一个助手
    alice_assistant = await alice.assistants.create(graph_id="agent")
    print(f"✅ Alice 创建了助手：{alice_assistant['assistant_id']}")

    # Alice 创建一个线程并聊天
    alice_thread = await alice.threads.create()
    print(f"✅ Alice 创建了线程：{alice_thread['thread_id']}")

    res = await alice.runs.create(
        thread_id=alice_thread["thread_id"],
        assistant_id="agent",
        input={"messages": [{"role": "user", "content": "嗨，这是 Alice 的私有聊天"}]}
    )
    print("✅ Alice 的机器人响应：", res)

    # Bob 尝试访问 Alice 的线程
    try:
        await bob.threads.get(alice_thread["thread_id"])
        print("❌ Bob 不应该看到 Alice 的线程！")
    except Exception as e:
        print("✅ Bob 被正确拒绝访问：", e)

    # Bob 创建自己的线程
    bob_thread = await bob.threads.create()
    await bob.runs.create(
        thread_id=bob_thread["thread_id"],
        assistant_id="agent",
        input={"messages": [{"role": "user", "content": "嗨，这是 Bob 的私有聊天"}]}
    )
    print(f"✅ Bob 创建了自己的线程：{bob_thread['thread_id']}")

    # 列出线程 - 每个用户只看到自己的线程
    alice_threads = await alice.threads.search()
    bob_threads = await bob.threads.search()
    print(f"✅ Alice 看到 {len(alice_threads)} 个线程")
    print(f"✅ Bob 看到 {len(bob_threads)} 个线程")


if __name__ == '__main__':
    asyncio.run(main())
    print("所有操作完成！")
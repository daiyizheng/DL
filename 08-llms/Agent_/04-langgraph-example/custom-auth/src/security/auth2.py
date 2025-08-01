# 保留之前的处理程序...

from langgraph_sdk import Auth

# 保留上一教程中的测试用户
VALID_TOKENS = {
    "user1-token": {"id": "user1", "name": "Alice"},
    "user2-token": {"id": "user2", "name": "Bob"},
}

auth = Auth()

@auth.authenticate
async def get_current_user(authorization: str | None) -> Auth.types.MinimalUserDict:
    """上一教程中的认证处理程序。"""
    assert authorization
    scheme, token = authorization.split()
    assert scheme.lower() == "bearer"

    if token not in VALID_TOKENS:
        raise Auth.exceptions.HTTPException(status_code=401, detail="无效令牌")

    user_data = VALID_TOKENS[token]
    return {
        "identity": user_data["id"],
    }

@auth.on
async def add_owner(
    ctx: Auth.types.AuthContext,  # 包含当前用户的信息
    value: dict,  # 被创建或访问的资源
):
    """通过资源元数据使资源对创建者私有。"""
    # 示例：
    # ctx: AuthContext(
    #     permissions=[],
    #     user=ProxyUser(
    #         identity='user1',
    #         is_authenticated=True,
    #         display_name='user1'
    #     ),
    #     resource='threads',
    #     action='create_run'
    # )
    # value: 
    # {
    #     'thread_id': UUID('1e1b2733-303f-4dcd-9620-02d370287d72'),
    #     'assistant_id': UUID('fe096781-5601-53d2-b2f6-0d3403f7e9ca'),
    #     'run_id': UUID('1efbe268-1627-66d4-aa8d-b956b0f02a41'),
    #     'status': 'pending',
    #     'metadata': {},
    #     'prevent_insert_if_inflight': True,
    #     'multitask_strategy': 'reject',
    #     'if_not_exists': 'reject',
    #     'after_seconds': 0,
    #     'kwargs': {
    #         'input': {'messages': [{'role': 'user', 'content': 'Hello!'}]},
    #         'command': None,
    #         'config': {
    #             'configurable': {
    #                 'langgraph_auth_user': ... 你的用户对象 ...
    #                 'langgraph_auth_user_id': 'user1'
    #             }
    #         },
    #         'stream_mode': ['values'],
    #         'interrupt_before': None,
    #         'interrupt_after': None,
    #         'webhook': None,
    #         'feedback_keys': None,
    #         'temporary': False,
    #         'subgraphs': False
    #     }
    # }

    # 执行两件事：
    # 1. 将用户 ID 添加到资源的元数据中。每个 LangGraph 资源都有一个随资源持久化的 `metadata` 字典。
    # 此元数据在读取和更新操作中可用于过滤
    # 2. 返回一个过滤器，仅允许用户看到自己的资源
    filters = {"owner": ctx.user.identity}
    metadata = value.setdefault("metadata", {})
    metadata.update(filters)

    # 只允许用户看到自己的资源
    return filters


@auth.on.threads.create
async def on_thread_create(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.create.value,
):
    """在创建线程时添加所有者。

    此处理程序在创建新线程时运行，执行两件事：
    1. 在创建的线程上设置元数据以跟踪所有权
    2. 返回一个过滤器，确保只有创建者可以访问
    """
    # 示例 value：
    #  {'thread_id': UUID('99b045bc-b90b-41a8-b882-dabc541cf740'), 'metadata': {}, 'if_exists': 'raise'}

    # 在创建的线程上添加所有者元数据
    # 此元数据随线程存储并持久化
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity

    # 返回过滤器，仅限创建者访问
    return {"owner": ctx.user.identity}

@auth.on.threads.read
async def on_thread_read(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.read.value,
):
    """仅允许用户读取自己的线程。

    此处理程序在读取操作时运行。由于线程已存在，无需设置元数据，
    只需返回一个过滤器，确保用户只能看到自己的线程。
    """
    return {"owner": ctx.user.identity}

@auth.on.assistants
async def on_assistants(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.assistants.value,
):
    # 为了说明目的，我们将拒绝所有涉及助手资源的请求
    # 示例 value：
    # {
    #     'assistant_id': UUID('63ba56c3-b074-4212-96e2-cc333bbc4eb4'),
    #     'graph_id': 'agent',
    #     'config': {},
    #     'metadata': {},
    #     'name': 'Untitled'
    # }
    raise Auth.exceptions.HTTPException(
        status_code=403,
        detail="用户缺乏所需权限。",
    )

# 假设你在存储中按 (user_id, resource_type, resource_id) 组织信息
@auth.on.store()
async def authorize_store(ctx: Auth.types.AuthContext, value: dict):
    # 每个存储项的“namespace”字段是一个可以视为项目录的元组
    namespace: tuple = value["namespace"]
    assert namespace[0] == ctx.user.identity, "未授权"

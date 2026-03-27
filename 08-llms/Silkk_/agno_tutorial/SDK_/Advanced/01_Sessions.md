# 会话管理
## 会话

具有持久历史记录和状态的多轮对话线程。

当你调用该服务时Agent.run()，它会创建一个单一的、无状态的交互。代理会响应你的消息，仅此而已——不会记住刚刚发生的事情。

但大多数实际应用需要的是对话，而不仅仅是一次性的交流。这就是会话的作用所在。
​
### 什么是会话？
您可以将会话想象成一个对话线程。它是由用户与您的客服人员、团队或工作流之间来回交互（称为“运行”）组成的集合。每个会话都有一个唯一的标识符session_id，该标识符将该对话的所有运行、聊天记录、状态和指标关联起来。
以下是详细分析：


Session: `session_id`标识的多回合对话。包含该对话线程的所有运行、历史、状态和指标。

Run：会话中的一次互动。每次调用 `Agent.run()`、`Team.run()`或 `Workflow.run()` 时，都会创建一个新的`run_id`。把它想象成对话中的一对消息和回复。运行可以暂停以满足人工参与需求，并在这些需求解决后继续运行。


> 会话需要数据库来存储历史记录和状态。有关设置详情，请参阅“会话存储”部分。
> 工作流会话的工作方式有所不同：与存储对话消息的代理和团队会话不同，工作流会话跟踪完整的管道执行（运行），包括输入和输出。鉴于这些独特的特性，我们专门创建了“工作流会话”部分，其中涵盖了工作流特有的功能，例如基于运行的历史记录、会话状态和工作流代理。

### 单次运行示例

当你运行代理但未指定`session_id`时，Agno 会自动为你生成一个`run_id`和一个`session_id`：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

agent = Agent(model=OpenAIResponses(id="gpt-5.2"))

# Run the agent - Agno auto-generates session_id and run_id
response = agent.run("Tell me a 5 second short story about a robot")
print(response.content)
print(f"Run ID: {response.run_id}")        # Auto-generated UUID
print(f"Session ID: {response.session_id}") # Auto-generated UUID
```


这将创建一个新的会话，仅运行一次。但问题在于：如果没有配置数据库，则无法持久保存。会话session_id在仅存于本次运行中，但您无法稍后继续对话，因为没有任何内容被保存。要真正使用会话进行多轮对话，您需要配置数据库（即使是虚拟数据库InMemoryDb也可以）。

### 多用户对话
在生产环境中，多个用户经常同时与同一个客服人员或团队进行沟通。会话机制可以确保这些对话线程彼此隔离：
- user_id区分使用您产品的人。
- session_id区分该用户的对话主题（类似于“聊天标签”）。
- 只有通过add_history_to_context启用对话记录时，对话记录才会流入游戏。

要了解包括持久性、历史记录和每个用户的会话 ID 在内的完整演练，请按照“持久化会话”指南或“历史记录”指南进行操作。

## 持久会话

将会话数据存储在数据库中，以便进行多轮对话。

要启用跨多次运行的会话，您需要配置数据库。配置完成后，Agno 会自动存储对话历史记录、会话状态和运行元数据。已暂停的运行会保留其状态和要求，以便稍后继续运行。

> 数据库选择、连接字符串、凭据管理和操作指南均位于“数据库概览”页面。请在此处重用该设置——本页面仅添加特定于会话的注意事项。

### 支持的数据库
这里没有什么新内容——只需重用以下数据库驱动程序和指南/basics/storage：

- PostgreSQL（推荐）——生产级，支持自定义session_table名称以隔离工作负载。
- SQLite – 非常适合本地开发；只需像在其他地方一样替换文件路径即可。
- InMemoryDb——仅适用于测试或演示。进程退出时数据会消失。

如果需要调整索引、保留或连接池，请在共享存储层进行调整，以便每个功能（会话、内存、知识等）都能受益于相同的配置。

### 会话 ID
会话以session_id标识。使用同一个ID继续对话：

```python
# First run
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    session_id="conversation_123",
)
agent.run("My name is Alice")

# Later run with same session_id
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    session_id="conversation_123",  # Same ID
    add_history_to_context=True,     # Enable history
)
agent.run("What's my name?")  # Agent remembers "Alice"
```

需要自定义命名规则、缓存或便于用户界面显示的标签？请继续阅读会话管理指南。
​
### 存储的内容
配置数据库时，Agno 会自动存储：

- ✅消息- 用户输入和代理响应
- ✅运行元数据- 时间戳、令牌使用情况、模型信息
- ✅会话状态- 自定义键值数据
- ✅工具调用- 工具使用情况和结果（可选）
- ✅媒体- 图片、音频、文件（可选）

请参阅“存储控制”以自定义要保存的内容。

### 多用户会话
用于user_id追踪不同用户：

```python
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
)

# Specify the user_id and session_id on run to start or continue the conversation
agent.print_response("Hello!", session_id="session_456", user_id="alice@example.com")
```

### 会话存储方案
配置数据库时，Agno 会以结构化格式存储会话信息。以下是每个会话保存的内容：

| 场地 | 类型 | 描述 |
| :--- | :--- | :--- |
| session＿id | str | 此对话主题的唯一标识符 |
| session＿type | str | 会话类型（代理、团队或工作流程） |
| agent＿id | str | 代理 ID（如果这是一个代理会话） |
| team＿id | str | 团队 ID（如果这是团队会议） |
| workflow＿id | str | 工作流 ID（如果这是一个工作流会话） |
| user＿id | str | 此会话所属的用户 |
| session＿data | dict | 会话特定数据和状态 |
| agent＿data | dict | 代理配置和元数据 |
| team＿data | dict | 团队配置和元数据 |
| workflow＿data | dict | 工作流配置和元数据 |
| metadata | dict | 其他自定义元数据 |
| runs | list | 本次会话中的所有运行（交互） |
| summary | dict | 会话摘要（如果已启用） |
| created＿at | int | 会话创建时的 Unix 时间戳 |
| updated＿at | int | 上次更新的 Unix 时间戳 |

### 存储控制

控制哪些会话数据会持久化到数据库中。

随着会话数据的积累，您的数据库可能会快速增长。Agno 提供三个存储标志，让您可以精细控制哪些数据需要持久化：

- store_media 图片、视频、音频和文件上传
- store_tool_messages- 工具调用及其结果
- store_history_messages- 来自先前运行的历史消息


### 储控制的工作原理
您可以将这些标记视为数据库的过滤器，而不是代理的过滤器。在运行过程中，您的代理或团队可以看到所有内容：媒体、工具结果、历史记录。一切运行正常。过滤操作仅在运行结束后保存到数据库时才会发生。

因此，您可以关闭媒体或工具消息的存储功能，而不会造成任何故障。您的代理仍然会处理图像，工具仍然会运行，历史记录仍然会流向模型。您只是选择将哪些内容写入磁盘。即使您没有持久化所有数据，令牌指标仍然能够反映实际使用情况。

> 重要提示：store_tool_messages=False移除工具调用和结果对

禁用工具消息存储后，Agno 会移除工具结果，并从相应的助手消息（来自 LLM 的消息）中剥离工具调用。这是为了维护模型提供程序所期望的有效消息序列。

您的指标仍将显示实际使用的令牌，包括已移除的工具消息。

### 存储标志参考

| 旗帜 | 默认 | 它控制着什么 | 禁用时的影响 |
| :--- | :--- | :--- | :--- |
| store＿media | True | 用户上传的图片、视频、音频和文件 | 媒体文件未保存到数据库 |
| store＿tool＿messages | True | 工具调用及其结果（同时移除相应的助手消息） | 工具执行细节不存储，节省大量空间 |
| store＿history＿messages | False | 来自先前运行的历史消息 | 旧历史记录不会被保存，只保留当前运行结果。 |


### 禁用媒体存储
大型媒体上传（图片、PDF、音频）可能会占用大量会话数据。请关闭此功能，store_media=False并将原始文件存储在其他位置（例如 S3、GCS 等）。

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.db.sqlite import SqliteDb

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=SqliteDb(db_file="tmp/agent.db"),
    store_media=False,
)

from agno.team import Team
from agno.models.openai import OpenAIResponses
from agno.db.sqlite import SqliteDb

team = Team(
    model=OpenAIResponses(id="gpt-5.2"),
    db=SqliteDb(db_file="tmp/team.db"),
    store_media=False,
)
```


### 推荐工作流程
将文件上传到您首选的存储服务并保留 URL/ID

将该 URL 传递给代理/团队，以便其可以获取/处理文件

通过 store_media=False 跳过保留原始媒体

### 禁用工具存储
工具调用很容易占用大量存储空间（例如网页抓取的页面或大型 API 请求）。切换此选项store_tool_messages=False可从持久化运行记录中移除触发该工具调用的相应助手消息中的工具结果和工具调用。指标仍会显示实际的令牌使用情况。

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.db.sqlite import SqliteDb
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
    db=SqliteDb(db_file="tmp/agent.db"),
    store_tool_messages=False,
)

from agno.team import Team
from agno.models.openai import OpenAIResponses
from agno.db.sqlite import SqliteDb
from agno.tools.hackernews import HackerNewsTools

team = Team(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
    db=SqliteDb(db_file="tmp/team.db"),
    store_tool_messages=False,
)
```
#### 考虑因素

- 移除工具消息可以保持对提供商友好的消息顺序不变（没有多余的工具角色）。
- 稍后审核工具行为时，请在运行完成之前重新运行该工具或将其输出记录到其他位置。
- 当工具返回你不需要的二进制负载时，再配合store_media=False


### 历史记录存储
store_history_messages 默认为 False 以防止数据库膨胀。 历史记录在运行期间仍然到达 LLM（通过 add_history_to_context），但历史消息在持久化之前会被清除。 仅当前运行的消息会写入数据库。

如果需要持久保存完整的对话历史，请设置store_history_messages=True。

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.db.sqlite import SqliteDb


agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=SqliteDb(db_file="tmp/agent.db"),
    add_history_to_context=True,
    num_history_runs=3,
    store_history_messages=True,
)

from agno.team import Team
from agno.models.openai import OpenAIResponses
from agno.db.sqlite import SqliteDb

team = Team(
    model=OpenAIResponses(id="gpt-5.2"),
    db=SqliteDb(db_file="tmp/team.db"),
    add_history_to_context=True,
    num_history_runs=3,
    store_history_messages=True,
)
```


### 何时启用
- 您需要查看数据库中的历史对话记录。
- 您正在调试多运行交互
- 您的代理非常依赖会话的历史背景。


### 合并存储标志
您可以同时使用多个标志来优化存储策略：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.db.sqlite import SqliteDb

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=SqliteDb(db_file="tmp/agent.db"),
    store_media=False,           # Media stored externally
    store_tool_messages=False,   # Tool results not needed
    store_history_messages=False, # Only current run persisted
)


from agno.team import Team
from agno.models.openai import OpenAIResponses
from agno.db.sqlite import SqliteDb

team = Team(
    model=OpenAIResponses(id="gpt-5.2"),
    db=SqliteDb(db_file="tmp/team.db"),
    store_media=False,           # Media stored externally
    store_tool_messages=False,   # Tool results not needed
    store_history_messages=False, # Only current run persisted
)


```

当您需要减少存储空间占用时（例如，大型工具有效载荷可以随时重新运行、媒体文件存储在 S3 中、不需要的旧历史记录），请关闭此标志。当您需要完整的转录文本用于审计/合规性、分析或客服人员培训时，请保持此标志开启。
​


## 会话管理

管理会话标识符、名称和性能优化

Agno 的会话管理功能让您可以控制会话的识别、命名和缓存方式，从而实现最佳性能。运行可以暂停以满足人工干预的需求，并在这些需求得到满足后继续进行。
​

### 会话 ID
每个会话都有一个唯一的标识符（session_id），用于跟踪多次运行中的对话：
- 自动生成：如果未提供，Agno 会自动生成 UUID。
- 手动：您可以提供自己的会话 ID 以进行自定义跟踪
- 按用户：可与其他方式结合使用user_id，以跟踪多个用户的会话

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.db.sqlite import SqliteDb

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=SqliteDb(db_file="tmp/agent.db"),
)

# Use your own session ID
agent.run("Hello", session_id="user_123_session_456")


from agno.team import Team
from agno.models.openai import OpenAIResponses
from agno.db.sqlite import SqliteDb

team = Team(
    model=OpenAIResponses(id="gpt-5.2"),
    members=[...],
    db=SqliteDb(db_file="tmp/team.db"),
)

# Use your own session ID
team.run("Hello", session_id="user_123_session_789")
```

### 访问消息和聊天记录
您可以使用以下方法访问会话中的消息get_messages：
```python
session = agent.get_session(session_id="session_123")
messages = session.get_messages()


session = team.get_session(session_id="session_456")
messages = session.get_messages()

```
这样您就可以获得会话中所有运行的所有消息，包括工具调用和系统消息。

如果要生成仅包含用户和助手消息的更简洁列表，可以使用以下get_chat_history方法：

```python
messages = agent.get_chat_history(session_id="session_123")

messages = team.get_chat_history(session_id="session_456")
```

### 会话命名
会话名称是易于理解的标签，可以更轻松地识别和管理对话——非常适合收件箱式用户界面、支持队列或将对话链接回外部工单。

#### 手动命名
使用以下set_session_name()方式设置自定义名称：

```python
agent.set_session_name(session_id="session_001", session_name="Product Launch Planning")
name = agent.get_session_name(session_id="session_001")


team.set_session_name(session_id="session_001", session_name="Product Launch Planning")
name = team.get_session_name(session_id="session_001")
```

- 将会话 ID 视为唯一数据源；名称只是供人阅读的元数据。
- 当话题发生变化时，重命名对话——调用此方法的次数没有限制。
- 在向终端用户展示之前，在将其暴露给最终用户之前，请先用您自己的辅助函数set_session_name进行封装

#### 自动生成的名称
让人工智能根据对话内容生成有意义的名称

```python
session = agent.set_session_name(
    session_id="session_123",
    autogenerate=True,
)
# Access the generated name
name = agent.get_session_name(session_id="session_123")
print(name)  # e.g. "E-commerce API Planning"

session = team.set_session_name(
    session_id="session_456",
    autogenerate=True,
)
# Access the generated name
name = team.get_session_name(session_id="session_456")
print(name)  # e.g. "Product Launch Strategy"
```

调用此方法`set_session_name(autogenerate=True)`会要求模型读取会话中的前几条消息，并生成一个简短的（≤5 个词）标签。该方法返回更新后的会话对象。使用该方法`get_session_name()`可以检索生成的名称。

#### 最佳实践：
延迟生成，直到对话有了有意义的上下文（例如，在发送 2-3 条消息之后）。
- 提供备用方案：将调用封装在您自己的辅助函数中，如果生成失败，则回退到人工输入的姓名或工单 ID。
- 批量作业：遍历数据库中的会话 ID，并对set_session_name(..., autogenerate=True)每个会话 ID 调用一次 API。API 是同步的，因此请考虑模型延迟。
- 成本：每一代都会增加一次模型调用。session_summary_manager如果对成本敏感，请使用更便宜的模型或采用带外方式运行。


## 会话缓存
会话缓存将会话对象存储在内存中以提高性能。cache_session=True在首次数据库读取后将已填充的会话对象保留在内存中，避免后续运行进行额外的查询。

```python
from agno.team import Team
from agno.models.openai import OpenAIResponses

team = Team(
    model=OpenAIResponses(id="gpt-5.2"),
    session_id="team_session",
    cache_session=True,  # Enable in-memory caching
)

# First run loads from database and caches
team.run("First message")

# Subsequent runs use cached session (faster)
team.run("Second message")
```

## 历史管理
控制如何访问和使用对话历史记录。

已配置数据库的客服人员和团队会自动跟踪消息和运行历史记录。您可以通过多种方式访问​​和使用此历史记录，让您的客服人员和团队“记住”过去的对话。

### 常见模式
​
自动历史记录（最常用）

启用此功能`add_history_to_context=True`后，每次运行都会自动包含最近的邮件：

```python
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=SqliteDb(db_file="tmp/data.db"),
    add_history_to_context=True,
    num_history_runs=3,  # Last 3 conversation turns
)

team = Team(
    model=OpenAIResponses(id="gpt-5.2"),
    db=SqliteDb(db_file="tmp/data.db"),
    add_history_to_context=True,
    num_history_runs=3,  # Last 3 conversation turns
)
```
​
适用场景：聊天式产品、快速原型、任何需要根据前一轮对话获取上下文信息的场景。


### 按需访问历史记录
启用此功能`read_chat_history=True`后，模型将自行决定何时查找历史记录：

```python
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=SqliteDb(db_file="tmp/data.db"),
    read_chat_history=True,  # Model can call get_chat_history() tool
)
```
何时使用：分析、审计，或者当您希望模型有选择地访问历史记录而不是始终包含历史记录时。

#### 程序化访问
直接在代码中检索历史记录：

```python
# All messages excluding those marked as from_history
chat_history = agent.get_chat_history()

# User-assistant message pairs from each run
messages = agent.get_session_messages()

# Last run output with metrics and tool calls
last_run = agent.get_last_run_output()
```

何时使用：构建自己的用户界面、进行分析、调试或需要原始转录文本时。


### 选择图案
- 简短聊天：保留默认设置（关闭聊天记录）或add_history_to_context启用num_history_runs=3
- 长生命周期线程：结合有限的历史记录（num_history_runs=2）和会话摘要，以保持令牌易于管理。
- 工具密集型代理：用于max_tool_calls_from_history限制上下文中的工具调用噪声
- 审计/调试流程：启用read_chat_history=True后，模型仅在需要时才查找信息。
- 跨会话回忆：search_session_history=True与num_history_sessions=2（保持较低水平以避免上下文限制）一起使用
- 程序化工作流程：直接在代码中调用get_session_messages()/get_chat_history()


## 会议总结

自动将冗长的对话浓缩成简洁的摘要

随着对话时长增加，将完整的聊天记录传递给您的 LLM 系统会变得既费时又费力。会话摘要功能通过自动将对话浓缩成简洁的摘要来解决这个问题，这些摘要能够抓住关键要点。

把它想象成在漫长的会议中做笔记——你不需要记录下所有发言内容，只需要记录重要的部分。

### 问题：代币成本不断上涨
如果没有摘要，每条消息都会添加到您的上下文窗口中：

```python
Run 1: 100 tokens
Run 2: 250 tokens (100 history + 150 new)
Run 3: 450 tokens (250 history + 200 new)
Run 4: 750 tokens (450 history + 300 new)
...exponential growth

```

这很快就会变得成本高昂，而且会受到上下文的限制。


### 解决方案：自动摘要
会议总结概括了您的会议记录：

```python
Run 1: 100 tokens
Run 2: 250 tokens
[Summary created: 50 tokens]
Run 3: 250 tokens (50 summary + 200 new)
Run 4: 350 tokens (50 summary + 300 new)
...linear growth
```

### 好处：
- ✅ 大幅降低代币成本
- ✅ 避免上下文窗口限制
- ✅ 保持对话连贯性
- ✅ 自动创建和更新

### 工作原理
会议总结遵循简单的三步模式：

1 启用摘要生成

可设置 `enable＿session＿summaries＝True` 为您的代理或团队。运行结束后，当有值得总结的消息时，系统会自动创建和更新摘要，并将其存储在您的数据库中。

2 在上下文中使用摘要

设置 `add_session_summary_to_context＝True` 为在消息中包含摘要（如果您启用了会话摘要生成，则默认启用此功能）。这样，系统不会发送数十条历史消息，而是仅发送精简的摘要，从而在保留上下文的同时显著减少令牌数量。

3 自定义（可选）

用于 `SessionSummaryManager` 控制摘要生成——使用更经济的模型、自定义提示或更改摘要格式。这样，您可以使用轻量级模型生成摘要，从而优化成本，同时保持主代理的强大功能。

### 启用会话摘要
启用此功能`enable_session_summaries=True`后，Agno 将为每个会话维护一个滚动摘要。摘要与存储的历史记录并列显示，以后可以重复使用以保存令牌。

```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.models.openai import OpenAIResponses

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=PostgresDb(db_url="postgresql+psycopg://ai:ai@localhost:5532/ai"),
    enable_session_summaries=True,
)

agent.print_response("Hi my name is John and I live in New York", session_id="conversation_123")

# Retrieve the summary
summary = agent.get_session_summary(session_id="conversation_123")
if summary:
    print(summary.summary, summary.topics)


from agno.team import Team
from agno.db.postgres import PostgresDb
from agno.models.openai import OpenAIResponses

team = Team(
    model=OpenAIResponses(id="gpt-5.2"),
    db=PostgresDb(db_url="postgresql+psycopg://ai:ai@localhost:5532/ai"),
    enable_session_summaries=True,
)

team.print_response("Hi my name is John and I live in New York", session_id="conversation_123")

# Retrieve the summary
summary = team.get_session_summary(session_id="conversation_123")
if summary:
    print(summary.summary, summary.topics)
```
#### 定制生成
- 提供一个选项SessionSummaryManager以指定更便宜的型号或自定义提示
- get_session_summary通过实例化一个轻量级代理，在所有会话中调用函数，以带外方式运行摘要生成。

### 在上下文中使用摘要
如果你启用了会话摘要生成，add_session_summary_to_context=True默认是启用的。如果你不想生成摘要，但仍想在上下文中使用，可以设置 add_session_summary_to_context=True。或者，如果你不想在上下文中使用摘要，可以设置 add_session_summary_to_context=False。


```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.models.openai import OpenAIResponses

db = PostgresDb(db_url="postgresql+psycopg://ai:ai@localhost:5532/ai")

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    add_session_summary_to_context=True,
)

agent.print_response("Hi my name is John and I live in New York", session_id="conversation_123")


from agno.team import Team
from agno.db.postgres import PostgresDb
from agno.models.openai import OpenAIResponses

db = PostgresDb(db_url="postgresql+psycopg://ai:ai@localhost:5532/ai")

team = Team(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    add_session_summary_to_context=True,
)

team.print_response("Hi my name is John and I live in New York", session_id="conversation_123")

```

Agno 会在每次运行前自动从存储中加载最新的摘要。您仍然可以混合使用最近的历史记录：

```python
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    add_session_summary_to_context=True,
    add_history_to_context=True,
    num_history_runs=2,  # Summary for long-term memory, last 2 runs for detail
)

team = Team(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    add_session_summary_to_context=True,
    add_history_to_context=True,
    num_history_runs=2,  # Summary for long-term memory, last 2 runs for detail
)

```

## 工作流程会议

通过会话历史记录跟踪多步骤工作流程的执行情况。

工作流会话会跟踪工作流的执行历史记录。与存储对话消息的代理或团队会话不同，工作流会话存储完整的工作流运行记录，每次运行都代表从输入到最终输出的所有工作流步骤的完整执行。

换个角度想：
- 代理/团队会话= 对话历史记录（来回消息）
- 工作流会话= 执行历史记录（包含结果的完整管道运行）


### 何时使用工作流会话
当您需要时，请使用工作流会话：
- 跟踪多次运行的工作流执行历史记录
- 在工作流程的各个步骤之间共享状态（例如在管道阶段之间传递数据）
- 通过访问过去的输入和输出，使工作流能够从之前的运行中学习。
- 保存工作流程结果以进行分析和调试
- 在多个工作流执行过程中保持上下文一致性

> 大多数情况下，建议在工作流程中添加会话持久性。

### 工作流会话的工作原理
当您使用数据库创建工作流时，Agno 会自动为您管理会话：


```python
from agno.workflow import Workflow
from agno.db.sqlite import SqliteDb

workflow = Workflow(
    name="Research Pipeline",
    db=SqliteDb(db_file="workflows.db"),
    steps=[...],
)

# Each run creates or updates the workflow session
result = workflow.run(input="AI trends", session_id="session_123")
```

每次运行工作流时，Agno：
- run_id为本次执行创建一个唯一标识符
- 存储输入、输出和所有步骤结果
- 使用新运行更新会话
- 使历史记录可供将来运行使用
​

### 工作流程会话结构
工作流会话存储：

```python
@dataclass
class WorkflowSession:
    session_id: str           # Unique session identifier
    user_id: str | None       # User who owns this session
    workflow_id: str | None   # Which workflow this belongs to
    workflow_name: str | None # Name of the workflow
    
    # List of all workflow runs (executions)
    runs: List[WorkflowRunOutput] | None
    
    # Session-specific data
    session_data: Dict | None    # Includes session_name, session_state
    workflow_data: Dict | None   # Workflow configuration
    metadata: Dict | None        # Custom metadata
    
    created_at: int | None    # Unix timestamp
    updated_at: int | None    # Unix timestamp
```

### 存储的内容
每次工作流运行都会存储：
- 输入：传递给的数据workflow.run()
- 输出：工作流程的最终结果。
- 步骤结果：流程中每个步骤的输出。
- 会话数据：执行时间、状态、指标
- 会话状态：步骤间共享的数据（如果使用）
​

### 与代理/团队会议的主要区别
如果您熟悉代理会话或团队会话，以下是主要区别：

| 特征 | 经纪人／团队会议 | 工作流程会议 |
| :--- | :--- | :--- |
| 储存了什么 | 信息和对话轮次 | 包含步骤结果的完整工作流程运行 |
| 历史类型 | 基于消息的（聊天记录） | 基于运行（执行历史） |
| 摘要 | 得到支持 enable＿session＿summaries | 不支持（存储完整运行记录） |
| 历史格式 | LLM 上下文中的消息 | 先前运行结果已添加到步骤输入中 |


### 数据库选项
工作流会话需要数据库来保存执行历史记录。Agno 支持多种数据库类型：

```python
from agno.db.sqlite import SqliteDb

# Quick start - SQLite
workflow = Workflow(
    name="Research Pipeline",
    db=SqliteDb(db_file="workflows.db"),
    steps=[...],
)
```

### 工作流程历史记录
工作流历史记录允许工作流步骤访问先前运行的结果。启用此功能后，Agno 会格式化先前运行的结果，并将其添加到每个步骤的输入中，以便后续执行可以基于该上下文进行构建。

#### 启用步骤历史记录

```python
from agno.workflow import Workflow
from agno.db.sqlite import SqliteDb

workflow = Workflow(
    name="Content Pipeline",
    db=SqliteDb(db_file="workflows.db"),
    steps=[...],
    add_workflow_history_to_steps=True,  # Include previous runs
    num_history_runs=5,                  # Limit how many runs to load
)
```

#### 历史格式
Agno 会将过去的运行结果封装在一个结构化的 XML 块中，然后再将其插入到每个步骤的输入中：

```python
<workflow_history_context>
[Workflow Run-1]
User input: Create a blog post about AI
Workflow output: [Full output from run]

[Workflow Run-2]
User input: Write about machine learning
Workflow output: [Full output from run]
</workflow_history_context>
```

有关高级控制、逐步覆盖和程序化访问模式，请参阅工作流历史记录实施指南。

### 会话命名
为工作流程会话命名，以便于识别：
​
#### 手动命名

```python
from agno.workflow import Workflow
from agno.db.sqlite import SqliteDb

workflow = Workflow(
    name="Research Pipeline",
    db=SqliteDb(db_file="workflows.db"),
    steps=[...],
)

workflow.run(input="Analyze AI trends", session_id="session_123")
workflow.set_session_name(session_id="session_123", session_name="AI Trends Analysis Q4 2024")

# Retrieve the name
name = workflow.get_session_name(session_id="session_123")
print(name)  # "AI Trends Analysis Q4 2024"
```

#### 自动生成
工作流会话可以自动生成基于时间戳的名称：

```python
workflow = Workflow(
    name="Research Pipeline",
    description="Automated research and analysis pipeline",
    db=SqliteDb(db_file="workflows.db"),
    steps=[...],
)

workflow.run(input="Research topic", session_id="session_123")
workflow.set_session_name(session_id="session_123", autogenerate=True)

name = workflow.get_session_name(session_id="session_123")
print(name)  # "Automated research and analysis pipel - 2024-11-19 14:30"
```

## 指标

### 代理指标

代理运行和会话指标，用于衡量令牌使用情况和性能。

在 Agno 中运行代理时，您收到的响应（RunOutput）包含有关运行的详细指标。这些指标可帮助您了解资源使用情况（例如令牌使用情况和时间）、性能以及模型和工具调用的其他方面。

指标分为多个层级：

- 按消息：每条消息（助手、工具等）都有自己的指标。
- 每次运行：每个运行RunOutput都有自己的指标。
- 按会话：AgentSession 包含聚合的 session_metrics，它们是会话的所有 RunOutput.metrics 的总和。

### 用法示例
假设你有一个代理程序，它会执行一些任务，并且你想在任务运行后分析相关指标。以下是如何访问和打印这些指标的方法：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools
from agno.db.sqlite import SqliteDb
from rich.pretty import pprint

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
    db=SqliteDb(db_file="tmp/agents.db"),
    markdown=True,
)

run_response = agent.run(
    "What are the top stories on HackerNews?"
)

# Print metrics per message
if run_response.messages:
    for message in run_response.messages:
        if message.role == "assistant":
            if message.content:
                print(f"Message: {message.content}")
            elif message.tool_calls:
                print(f"Tool calls: {message.tool_calls}")
            print("---" * 5, "Metrics", "---" * 5)
            pprint(message.metrics.to_dict())
            print("---" * 20)

# Print the aggregated metrics for the whole run
print("---" * 5, "Run Metrics", "---" * 5)
pprint(run_response.metrics.to_dict())
# Print the aggregated metrics for the whole session
print("---" * 5, "Session Metrics", "---" * 5)
pprint(agent.get_session_metrics().to_dict())
```

您将看到包含以下信息的输出结果：
- input_tokens：发送给模型的令牌数量。
- output_tokens：从模型中收到的令牌数量。
- total_tokensinput_tokens：和的总和output_tokens。
- audio_input_tokens：发送给模型的用于音频输入的令牌数量。
- audio_output_tokens：从模型接收到的音频输出令牌数。
- audio_total_tokensaudio_input_tokens：和的总和audio_output_tokens。
- cache_read_tokens从缓存中读取的令牌数量。
- cache_write_tokens：写入缓存的令牌数量。
- reasoning_tokens用于推理的标记数量。
- duration：运行持续时间（秒）。
- time_to_first_token：生成第一个令牌所花费的时间。
- provider_metrics任何提供商特定的指标。

### 团队指标


团队运行和会话指标，用于衡量令牌使用情况和性能。

在 Agno 中运行团队时，您收到的响应（TeamRunOutput）包含有关运行的详细指标。这些指标可帮助您了解资源使用情况（例如令牌使用情况和时间）、性能以及模型和工具调用的其他方面，包括团队领导和团队成员的相关信息。

指标分为多个层级：
- 按消息：每条消息（助手、工具等）都有自己的指标。
- 每个成员的运行情况：每个团队成员的运行情况都有自己的指标。您可以通过TeamRunOutput设置来使成员运行情况可用store_member_responses=True，
- 团队级别：TeamRunOutput汇总所有团队领导和团队成员消息中的指标。
- 会话级别：会话中所有运行的汇总指标，包括团队领导和所有团队成员的指标。


### 用法示例
假设你有一个团队负责执行一些任务，并且你想在任务完成后分析相关指标。以下是如何访问和打印这些指标的方法：


```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.yfinance import YFinanceTools
from agno.utils.pprint import pprint_run_response
from rich.pretty import pprint

# Create team members
stock_agent = Agent(
    name="Stock Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    role="Get stock prices and financial data.",
    tools=[YFinanceTools()],
)

# Create the team
team = Team(
    name="Finance Team",
    model=OpenAIResponses(id="gpt-5.2"),
    members=[stock_agent],
    markdown=True,
    store_member_responses=True,
)

# Run the team
run_response = team.run(
    "What is the stock price of NVDA?"
)
pprint_run_response(run_response, markdown=True)

# Print team leader message metrics
print("---" * 5, "Team Leader Message Metrics", "---" * 5)
if run_response.messages:
    for message in run_response.messages:
        if message.role == "assistant":
            if message.content:
                print(f"Message: {message.content}")
            elif message.tool_calls:
                print(f"Tool calls: {message.tool_calls}")
            print("---" * 5, "Metrics", "---" * 5)
            pprint(message.metrics)
            print("---" * 20)

# Print aggregated team leader metrics
print("---" * 5, "Aggregated Metrics of Team", "---" * 5)
pprint(run_response.metrics)

# Print team leader session metrics
print("---" * 5, "Session Metrics", "---" * 5)
pprint(team.get_session_metrics().to_dict())

# Print team member message metrics
print("---" * 5, "Team Member Message Metrics", "---" * 5)
if run_response.member_responses:
    for member_response in run_response.member_responses:
        if member_response.messages:
            for message in member_response.messages:
                if message.role == "assistant":
                    if message.content:
                        print(f"Member Message: {message.content}")
                    elif message.tool_calls:
                        print(f"Member Tool calls: {message.tool_calls}")
                    print("---" * 5, "Member Metrics", "---" * 5)
                    pprint(message.metrics)
                    print("---" * 20)
```


您将看到包含以下信息的输出结果：
- input_tokens：发送给模型的令牌数量。
- output_tokens：从模型中收到的令牌数量。
- total_tokensinput_tokens：和的总和output_tokens。
- audio_input_tokens：发送给模型的用于音频输入的令牌数量。
- audio_output_tokens：从模型接收到的音频输出令牌数。
- audio_total_tokensaudio_input_tokens：和的总和audio_output_tokens。
- cache_read_tokens从缓存中读取的令牌数量。
- cache_write_tokens：写入缓存的令牌数量。
- reasoning_tokens用于推理的标记数量。
- duration：运行持续时间（秒）。
- time_to_first_token：生成第一个令牌所花费的时间。
- provider_metrics任何提供商特定的指标。


### 团队指标分析

本示例演示如何访问和分析全面的团队指标，包括消息级指标、会话指标和成员特定绩效数据。

#### 代码

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.yfinance import YFinanceTools
from agno.utils.pprint import pprint_run_response
from rich.pretty import pprint

db = SqliteDb(db_file="tmp/team_metrics.db")

stock_agent = Agent(
    name="Stock Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    role="Get stock prices and financial data.",
    tools=[YFinanceTools()],
)

team = Team(
    name="Finance Team",
    model=OpenAIResponses(id="gpt-5.2"),
    members=[stock_agent],
    db=db,
    session_id="team_metrics_demo",
    markdown=True,
    show_members_responses=True,
    store_member_responses=True,
)

run_output = team.run("What is the stock price of NVDA?")
pprint_run_response(run_output, markdown=True)

# Analyze team leader message metrics
print("=" * 50)
print("TEAM LEADER MESSAGE METRICS")
print("=" * 50)

if run_output.messages:
    for message in run_output.messages:
        if message.role == "assistant":
            if message.content:
                print(f"Message: {message.content[:100]}...")
            elif message.tool_calls:
                print(f"Tool calls: {message.tool_calls}")

            print("-" * 30, "Metrics", "-" * 30)
            pprint(message.metrics)
            print("-" * 70)

# Analyze aggregated team metrics
print("=" * 50)
print("AGGREGATED TEAM METRICS")
print("=" * 50)
pprint(run_output.metrics)

# Analyze session-level metrics
print("=" * 50)
print("SESSION METRICS")
print("=" * 50)
pprint(team.get_session_metrics(session_id="team_metrics_demo"))

# Analyze individual member metrics
print("=" * 50)
print("TEAM MEMBER MESSAGE METRICS")
print("=" * 50)

if run_output.member_responses:
    for member_response in run_output.member_responses:
        if member_response.messages:
            for message in member_response.messages:
                if message.role == "assistant":
                    if message.content:
                        print(f"Member Message: {message.content[:100]}...")
                    elif message.tool_calls:
                        print(f"Member Tool calls: {message.tool_calls}")

                    print("-" * 20, "Member Metrics", "-" * 20)
                    pprint(message.metrics)
                    print("-" * 60)
```
​
## 工作流程指标
工作流运行和会话指标，用于衡量令牌使用情况和性能。

在 Agno 中运行工作流时，您得到的响应（WorkflowRunOutput）包含有关工作流执行的详细指标。
这些指标可帮助您了解工作流程中所有代理、团队和自定义函数的令牌使用情况、执行时间、性能和步骤级别的详细信息。

指标分为多个层级：
- 每个工作流程：每个工作流程WorkflowRunOutput都包含一个指标对象，其中包含工作流程持续时间。
- 按步骤：每个步骤都有自己的指标，包括持续时间、令牌使用情况和模型信息。
- 按会话：会话指标汇总会话中所有运行的所有步骤级指标。


### 用法示例

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.workflow import Step, Workflow
from rich.pretty import pprint

# Define agents
hackernews_agent = Agent(
    name="HackerNews Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
    role="Extract key insights from HackerNews posts",
)

finance_agent = Agent(
    name="Finance Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[YFinanceTools()],
    role="Get stock prices and financial data",
)

# Define research team
research_team = Team(
    name="Research Team",
    members=[hackernews_agent, finance_agent],
    instructions="Research tech topics from HackerNews and financial data",
)

content_planner = Agent(
    name="Content Planner",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="Plan a content schedule based on research",
)

# Create workflow
workflow = Workflow(
    name="Content Creation Workflow",
    db=SqliteDb(db_file="tmp/workflow.db"),
    steps=[
        Step(name="Research Step", team=research_team),
        Step(name="Content Planning Step", agent=content_planner),
    ],
)

# Run workflow
response = workflow.run(input="AI trends in 2024")

# Print workflow-level metrics
print("Workflow Metrics")
if response.metrics:
    pprint(response.metrics.to_dict())

# Print workflow duration
if response.metrics and response.metrics.duration:
    print(f"\nTotal execution time: {response.metrics.duration:.2f} seconds")

# Print step-level metrics
print("Step Metrics")
if response.metrics:
    for step_name, step_metrics in response.metrics.steps.items():
        print(f"\nStep: {step_name}")
        print(f"Executor: {step_metrics.executor_name} ({step_metrics.executor_type})")
        if step_metrics.metrics:
            print(f"Duration: {step_metrics.metrics.duration:.2f}s")
            print(f"Tokens: {step_metrics.metrics.total_tokens}")

# Print session metrics
print("Session Metrics")
pprint(workflow.get_session_metrics().to_dict())
```

您将看到包含以下信息的输出结果：

工作流级别指标：
- duration：工作流总执行时间（秒）（从开始到结束，包括编排开销）
- steps：将步骤名称映射到其各个步骤指标的字典

步骤级指标：
- step_name步骤名称
- executor_type执行者类型（“代理人”、“团队”或“职能部门”）
- executor_name遗嘱执行人姓名
- metrics执行指标，包括令牌数、持续时间和模型信息（参见指标架构）

会话指标：
- 汇总会话中所有运行的步骤级指标（令牌数、持续时间）。
- 仅包含代理/团队执行时间，不包含工作流编排开销
# 
## 什么是工具？

为代理提供会话、上下文、内存和知识的持久存储空间。

数据库是智能体工程的基础组成部分。将数据库添加到智能体中，即可获得用于存储会话、上下文、内存、学习数据和评估数据集的持久存储空间。

- 聊天记录。包含多轮对话的上下文信息。
- 会话持久化。跨请求存储会话信息和对话历史记录。
- 状态管理。跨运行存储内部代理状态。对代理规划至关重要。
- 上下文控制。概括、压缩、丰富和精简上下文，以获得更好的回复。
- 记忆和知识。存储用户级事实、可搜索知识、决策轨迹和已学习的见解。
- 跟踪和评估。存储详细的跟踪信息，用于调试、监控和构建评估数据集。
- 数据所有权。无第三方依赖。查询您自己的数据库。构建评估数据集，提取小样本，标记低质量响应以供审核。
- 优秀的软件就是这样构建的。智能代理也不例外。


## 快速入门
```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb

agent = Agent(
    db=SqliteDb(db_file="agent.db"),
    add_history_to_context=True,
    num_history_runs=3,
)

# First message
agent.print_response("I'm working on a Python API project", session_id="dev_session")

# Later — agent remembers the context
agent.print_response("What testing framework should I use?", session_id="dev_session")
```

现在，代理会保存会话，并将最近 3 次运行包含在每次请求中。

### 与团队和工作流程协作
存储功能在代理、团队和工作流中完全一致：

```python
from agno.team import Team
from agno.workflow import Workflow
from agno.db.postgres import PostgresDb

db = PostgresDb(db_url="postgresql://user:pass@localhost:5432/mydb")

team = Team(db=db, ...)
workflow = Workflow(db=db, ...)
```

### 支持的数据库
Agno 支持 13 种以上的数据库用于会话存储。开发环境使用 SQLite，生产环境使用 PostgreSQL。查看所有支持的数据库。


### 异步支持
对于异步应用程序，请使用异步数据库类：
```python
from agno.agent import Agent
from agno.db.postgres import AsyncPostgresDb

agent = Agent(
    db=AsyncPostgresDb(db_url="postgresql+psycopg_async://..."),
)
```

### 聊天记录
在多轮对话中，应将之前的消息放在上下文中考虑。

聊天记录支持多轮对话。如果没有聊天记录，每次对话都是孤立的——客服人员无法了解之前的对话内容。借助数据库add_history_to_context=True，之前的消息会自动包含在每次请求中。

#### 启用聊天记录
设置add_history_to_context=True为每次运行都包含之前的消息：

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb

agent = Agent(
    db=SqliteDb(db_file="agent.db"),
    add_history_to_context=True,
    num_history_runs=3,  # Include last 3 turns
)

agent.print_response("My name is Sarah", session_id="chat_123")
agent.print_response("What's my name?", session_id="chat_123")  # Agent knows: "Sarah"
```

#### 控制历史大小
历史记录越多，包含的令牌就越多。使用以下参数控制包含哪些令牌：
| 范围 | 描述 |
| :--- | :--- |
| num＿history＿runs | 要包含的先前运行次数（默认值： 3 ） |
| num＿history＿messages | 所有运行中包含的最大消息数 |
| max＿tool＿calls＿from＿history | 限制历史记录中的工具调用消息 |

```python
agent = Agent(
    db=SqliteDb(db_file="agent.db"),
    add_history_to_context=True,
    num_history_runs=5,
    num_history_messages=20,  # Cap at 20 messages total
)
```



### 按需访问历史记录
不要总是提供历史记录，而是让代理人决定何时查找：

```python
agent = Agent(
    db=SqliteDb(db_file="agent.db"),
    read_chat_history=True,  # Agent gets a get_chat_history() tool
)

```

​### 跨会期历史记录
跨多个会话搜索贯穿整个对话的上下文：

```python
agent = Agent(
    db=SqliteDb(db_file="agent.db"),
    search_session_history=True,
    num_history_sessions=2,  # Search last 2 sessions
)
```
可用于构建自定义用户界面、调试或导出转录文本。

### 团队历史
团队支持成员之间共享更多历史记录：
```python
team = Team(
    db=SqliteDb(db_file="team.db"),
    add_history_to_context=True,
    num_history_runs=3,
    add_team_history_to_members=True,  # Share history across team members
)
```
有了它add_team_history_to_members=True，成员代理可以看到整个团队的对话，而不仅仅是他们自己的互动。


### 工作流程历史记录
工作流用于`add_workflow_history_to_steps`将先前运行的结果传递给后续步骤：

```python
from agno.workflow import Workflow

workflow = Workflow(
    db=SqliteDb(db_file="workflow.db"),
    add_workflow_history_to_steps=True,
    num_history_runs=5,
    steps=[...],
)
```

### 选择图案

| 设想 | 配置 |
| :--- | :--- |
| 聊天式产品 | add＿history＿to＿context＝True ，num＿history＿runs＝3 |
| 长时间的对话 | 有限的历史记录＋会议摘要 |
| 工具密集型代理 | 添加 max＿tool＿calls＿from＿history 以减少噪音 |
| 跨会次回忆 | search＿session＿history＝True ，num＿history＿sessions＝2 |
| 选择性查找 | read＿chat＿history＝True（代理人决定何时查找） |
| 自定义用户界面 | get＿chat＿history（）以编程方式使用 |


## 会话存储
从数据库中存储和检索代理会话。

当您向代理添加数据库时，会话会自动存储。会话会将相关的运行分组到一个对话线程中——每条消息、回复和元数据都会持久化存储在一个会话中session_id。本页介绍如何访问和配置该存储。

### 配置会话表
默认情况下，会话数据存储在agno_sessions表中。如果该表不存在，则会自动创建。

用于session_table将会话存储在自定义表中：
```python

from agno.db.postgres import PostgresDb

db = PostgresDb(
    db_url="postgresql://user:password@localhost:5432/mydb",
    session_table="my_agent_sessions",
)

agent = Agent(db=db)
```

### 存储的内容
每个会话记录包含：

| 场地 | 类型 | 描述 |
| :--- | :--- | :--- |
| session＿id | str | 唯一会话标识符 |
| session＿type | str | 会话类型（代理、团队或工作流程） |
| agent＿id | str | 代理 ID（如果是代理会话） |
| team＿id | str | 团队 ID（如果是团队会话） |
| workflow＿id | str | 工作流 ID（如果是工作流会话） |
| user＿id | str | 此会话所属的用户 |
| session＿data | dict | 会话特定数据和状态 |
| agent＿data | dict | 代理配置和元数据 |
| team＿data | dict | 团队配置和元数据 |
| workflow＿data | dict | 工作流配置和元数据 |
| metadata | dict | 其他自定义元数据 |
| runs | list | 本次会话中的所有运行（交互） |
| summary | dict | 会话摘要（如果已启用） |
| created＿at | int | 会话创建时的 Unix 时间曜 |
| updated＿at | int | 上次更新的 Unix 时间戳 |

### 检索会话
用于get_session()检索已存储的会话：

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb

agent = Agent(db=SqliteDb(db_file="agent.db"))

agent.print_response("What is the capital of France?", session_id="session_123")

# Retrieve the session
session = agent.get_session(session_id="session_123")

# Access session data
print(session.session_id)
print(session.runs)  # List of runs with messages and responses
```

### 检索会话
用于get_session()检索已存储的会话：
```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb

agent = Agent(db=SqliteDb(db_file="agent.db"))

agent.print_response("What is the capital of France?", session_id="session_123")

# Retrieve the session
session = agent.get_session(session_id="session_123")

# Access session data
print(session.session_id)
print(session.runs)  # List of runs with messages and responses

```

### 与团队和工作流程协作
会话存储对于团队和工作流的工作方式完全相同：

```python
from agno.team import Team
from agno.workflow import Workflow
from agno.db.sqlite import SqliteDb

db = SqliteDb(db_file="agno.db")

team = Team(db=db, ...)
workflow = Workflow(db=db, ...)

# Retrieve sessions the same way
team_session = team.get_session(session_id="team_session_123")
workflow_session = workflow.get_session(session_id="workflow_session_456")
```

## 支持的数据库
### 数据库索引

Agno支持的所有数据库索引。

Agno 支持以下按类别划分的数据库提供商：

### 关系型数据库
![](https://i-blog.csdnimg.cn/direct/e3a8e0834a704757ac9d80b03b64ae01.png)

### NoSQL数据库
![](https://i-blog.csdnimg.cn/direct/1e663542f4594f9e92bcf576d07228e9.png)


​### 数据库服务
![](https://i-blog.csdnimg.cn/direct/7b6e2d8338fb4bcaac72e4fab7912e2e.png)


### 存储和文件系统
![](https://i-blog.csdnimg.cn/direct/fe2e29b99ec34bcdac7a2af1fdba4416.png)


### PostgreSQL

使用 PostgreSQL 进行代理会话存储和持久化。

Agno 支持使用PostgreSQL作为数据库PostgresDb。

```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai" # Replace with your own connection string

# Setup your Database
db = PostgresDb(db_url=db_url)

# Setup your Agent with the Database
agent = Agent(db=db)
```


#### 运行 Postgres（使用 PgVector）
安装Docker Desktop并使用以下命令在5532端口运行PgVector：

```bash
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  agnohq/pgvector:16
```
#### 参数
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | Optional［str］ | － | 数据库实例的ID。默认为UUID。 |
| db_url | Optional［str］ | － | 要连接的数据库URL。 |
| db_engine | Optional［Engine］ | － | 要使用的 SQLAlchemy 数据库引擎。 |
| db_schema | Optional［str］ | － | 要使用的数据库模式。 |
| session_table | Optional［str］ | － | 用于存储代理、团队和工作流会话的表的名称。 |
| memory_table | Optional［str］ | － | 用于存储内存的表的名称。 |
| metrics_table | Optional［str］ | － | 用于存储指标的表的名称。 |
| eval_table | Optional［str］ | － | 用于存储评估运行数据的表的名称。 |
| knowledge_table | Optional［str］ | － | 用于存储知识内容的表的名称。 |
| traces_table | Optional［str］ | － | 用于存储跟踪信息的表的名称。 |
| spans_table | Optional［str］ | － | 用于存储跨度的表的名称。 |



#### Postgres for Agent
Agno 支持使用 PostgreSQL 作为使用该类的代理的存储后端PostgresDb。

#### 用法

安装Docker Desktop并使用以下命令在5532端口运行PgVector：

```bash
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  agnohq/pgvector:16
```

```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.tools.hackernews import HackerNewsTools

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

db = PostgresDb(db_url=db_url)

agent = Agent(
    db=db,
    tools=[HackerNewsTools()],
    add_history_to_context=True,
)
agent.print_response("How many people live in Canada?")
agent.print_response("What is their national anthem called?")
```

#### 参数
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | Optional［str］ | － | 数据库实例的ID。默认为UUID。 |
| db＿url | Optional［str］ | － | 要连接的数据库URL。 |
| db＿engine | Optional［Engine］ | － | 要使用的 SQLAlchemy 数据库引擎。 |
| db＿schema | Optional［str］ | － | 要使用的数据库模式。 |
| session＿table | Optional［str］ | － | 用于存储代理、团队和工作流会话的表的名称。 |
| memory＿table | Optional［str］ | － | 用于存储内存的表的名称。 |
| metrics＿table | Optional［str］ | － | 用于存储指标的表的名称。 |
| eval＿table | Optional［str］ | － | 用于存储评估运行数据的表的名称。 |
| knowledge＿table | Optional［str］ | － | 用于存储知识内容的表的名称。 |
| traces＿table | Optional［str］ | － | 用于存储跟踪信息的表的名称。 |
| spans＿table | Optional［str］ | － | 用于存储跨度的表的名称。 |


#### Postgres for Team
Agno 支持使用该类将 PostgreSQL 作为 Teams 的存储后端PostgresDb。
​
#### 用法

安装Docker Desktop并使用以下命令在5532端口运行PgVector：

```bash
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  agnohq/pgvector:16
```

```python
"""
Run: `uv pip install openai newspaper4k lxml_html_clean agno` to install the dependencies
"""

from typing import List

from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.hackernews import HackerNewsTools
from pydantic import BaseModel

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
db = PostgresDb(db_url=db_url)

class Article(BaseModel):
    title: str
    summary: str
    reference_links: List[str]

hn_researcher = Agent(
    name="HackerNews Researcher",
    model=OpenAIResponses(id="gpt-5.2"),
    role="Gets top stories from hackernews.",
    tools=[HackerNewsTools()],
)

web_searcher = Agent(
    name="Web Searcher",
    model=OpenAIResponses(id="gpt-5.2"),
    role="Searches the web for information on a topic",
    tools=[HackerNewsTools()],
    add_datetime_to_context=True,
)

hn_team = Team(
    name="HackerNews Team",
    model=OpenAIResponses(id="gpt-5.2"),
    members=[hn_researcher, web_searcher],
    db=db,
    instructions=[
        "First, search hackernews for what the user is asking about.",
        "Then, ask the web searcher to search for each story to get more information.",
        "Finally, provide a thoughtful and engaging summary.",
    ],
    output_schema=Article,
    markdown=True,
    show_members_responses=True,
)

hn_team.print_response("Write an article about the top 2 stories on hackernews")
```

#### 参数
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | Optional［str］ | － | 数据库实例的ID。默认为UUID。 |
| db＿url | Optional［str］ | － | 要连接的数据库URL。 |
| db＿engine | Optional［Engine］ | － | 要使用的 SQLAlchemy 数据库引擎。 |
| db＿schema | Optional［str］ | － | 要使用的数据库模式。 |
| session＿table | Optional［str］ | － | 用于存储代理、团队和工作流会话的表的名称。 |
| memory＿table | Optional［str］ | － | 用于存储内存的表的名称。 |
| metrics＿table | Optional［str］ | － | 用于存储指标的表的名称。 |
| eval＿table | Optional［str］ | － | 用于存储评估运行数据的表的名称。 |
| knowledge＿table | Optional［str］ | － | 用于存储知识内容的表的名称。 |
| traces＿table | Optional［str］ | － | 用于存储跟踪信息的表的名称。 |
| spans＿table | Optional［str］ | － | 用于存储跨度的表的名称。 |


#### Postgres for Workflows
Agno 支持使用 PostgreSQL 作为工作流的存储后端PostgresDb。

#### 用法
​
运行 PgVector
安装Docker Desktop并使用以下命令在5532端口运行PgVector

```bash
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  agnohq/pgvector:16

```


```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.hackernews import HackerNewsTools
from agno.workflow.step import Step
from agno.workflow.workflow import Workflow

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Define agents
hackernews_agent = Agent(
    name="Hackernews Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
    role="Extract key insights and content from Hackernews posts",
)
web_agent = Agent(
    name="Web Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
    role="Search the web for the latest news and trends",
)

# Define research team for complex analysis
research_team = Team(
    name="Research Team",
    members=[hackernews_agent, web_agent],
    instructions="Research tech topics from Hackernews and the web",
)

content_planner = Agent(
    name="Content Planner",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=[
        "Plan a content schedule over 4 weeks for the provided topic and research content",
        "Ensure that I have posts for 3 posts per week",
    ],
)

# Define steps
research_step = Step(
    name="Research Step",
    team=research_team,
)

content_planning_step = Step(
    name="Content Planning Step",
    agent=content_planner,
)

# Create and use workflow
if __name__ == "__main__":
    content_creation_workflow = Workflow(
        name="Content Creation Workflow",
        description="Automated content creation from blog posts to social media",
        db=PostgresDb(
            session_table="workflow_session",
            db_url=db_url,
        ),
        steps=[research_step, content_planning_step],
    )
    content_creation_workflow.print_response(
        input="AI trends in 2024",
        markdown=True,
    )

```

### 异步 PostgreSQL
使用 PostgreSQL 异步存储代理会话。

Agno 支持使用异步方式使用PostgreSQL，可通过该类AsyncPostgresDb实现。
​
```python

from agno.agent import Agent
from agno.db.postgres import AsyncPostgresDb

# Replace with your own connection string, and notice the `async_` prefix
db_url = "postgresql+psycopg_async://ai:ai@localhost:5532/ai"

# Setup your Database
db = AsyncPostgresDb(db_url=db_url)

# Setup your Agent with the Database
agent = Agent(db=db)
```


#### 运行 Postgres（使用 PgVector）
安装Docker Desktop并使用以下命令在5532端口运行PgVector：
```bash
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  agnohq/pgvector:16
```



#### 参数

| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | Optional［str］ | － | 数据库实例的ID。默认为UUID。 |
| db＿url | Optional［str］ | － | 要连接的数据库URL。 |
| db＿engine | Optional［AsyncEngine］ | － | 要使用的 SQLAlchemy 异步数据库引擎。 |
| db＿schema | Optional［str］ | － | 要使用的数据库模式。 |
| session＿table | Optional［str］ | － | 用于存储代理、团队和工作流会话的表的名称。 |
| memory＿table | Optional［str］ | － | 用于存储内存的表的名称。 |
| metrics＿table | Optional［str］ | － | 用于存储指标的表的名称。 |
| eval＿table | Optional［str］ | － | 用于存储评估运行数据的表的名称。 |
| knowledge＿table | Optional［str］ | － | 用于存储知识内容的表的名称。 |
| traces＿table | Optional［str］ | － | 用于存储跟踪信息的表的名称。 |
| spans＿table | Optional［str］ | － | 用于存储跨度的表的名称。 |


#### 代理的异步 Postgres

Agno 支持使用异步方式使用PostgreSQL，可通过该类AsyncPostgresDb实现。


#### 用法
​
运行 PgVector

安装Docker Desktop并使用以下命令在5532端口运行PgVector：
```bash
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  agnohq/pgvector:16

```

```python
import asyncio

from agno.agent import Agent
from agno.db.postgres import AsyncPostgresDb
from agno.tools.hackernews import HackerNewsTools

db_url = "postgresql+psycopg_async://ai:ai@localhost:5532/ai"
db = AsyncPostgresDb(db_url=db_url)

agent = Agent(
    db=db,
    tools=[HackerNewsTools()],
    add_history_to_context=True,
    add_datetime_to_context=True,
)

asyncio.run(agent.aprint_response("How many people live in Canada?"))
asyncio.run(agent.aprint_response("What is their national anthem called?"))
```


#### 参数
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | Optional［str］ | － | 数据库实例的ID。默认为UUID。 |
| db＿url | Optional［str］ | － | 要连接的数据库URL。 |
| db＿engine | Optional［AsyncEngine］ | － | 要使用的 SQLAlchemy 异步数据库引擎。 |
| db＿schema | Optional［str］ | － | 要使用的数据库模式。 |
| session＿table | Optional［str］ | － | 用于存储代理、团队和工作流会话的表的名称。 |
| memory＿table | Optional［str］ | － | 用于存储内存的表的名称。 |
| metrics＿table | Optional［str］ | － | 用于存储指标的表的名称。 |
| eval＿table | Optional［str］ | － | 用于存储评估运行数据的表的名称。 |
| knowledge＿table | Optional［str］ | － | 用于存储知识内容的表的名称。 |
| traces＿table | Optional［str］ | － | 用于存储跟踪信息的表的名称。 |
| spans＿table | Optional［str］ | － | 用于存储跨度的表的名称。 |


### 面向团队的异步 Postgres

Agno 支持使用异步方式使用PostgreSQL，可通过该类AsyncPostgresDb实现。
​
#### 用法
​
运行 PgVector

安装Docker Desktop并使用以下命令在5532端口运行PgVector：

```bash
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  agnohq/pgvector:16

```

async_postgres_for_team.py


```python
import asyncio
from typing import List

from agno.agent import Agent
from agno.db.postgres import AsyncPostgresDb
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.hackernews import HackerNewsTools
from pydantic import BaseModel

db_url = "postgresql+psycopg_async://ai:ai@localhost:5532/ai"
db = AsyncPostgresDb(db_url=db_url)

class Article(BaseModel):
    title: str
    summary: str
    reference_links: List[str]

hn_researcher = Agent(
    name="HackerNews Researcher",
    model=OpenAIResponses(id="gpt-5.2"),
    role="Gets top stories from hackernews.",
    tools=[HackerNewsTools()],
)

web_searcher = Agent(
    name="Web Searcher",
    model=OpenAIResponses(id="gpt-5.2"),
    role="Searches the web for information on a topic",
    tools=[HackerNewsTools()],
    add_datetime_to_context=True,
)

hn_team = Team(
    name="HackerNews Team",
    model=OpenAIResponses(id="gpt-5.2"),
    members=[hn_researcher, web_searcher],
    db=db,
    instructions=[
        "First, search hackernews for what the user is asking about.",
        "Then, ask the web searcher to search for each story to get more information.",
        "Finally, provide a thoughtful and engaging summary.",
    ],
    output_schema=Article,
    markdown=True,
    show_members_responses=True,
)

asyncio.run(
    hn_team.aprint_response("Write an article about the top 2 stories on hackernews")
)

```


#### 参数
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | Optional［str］ | － | 数据库实例的ID。默认为UUID。 |
| db＿url | Optional［str］ | － | 要连接的数据库URL。 |
| db＿engine | Optional［AsyncEngine］ | － | 要使用的 SQLAlchemy 异步数据库引擎。 |
| db＿schema | Optional［str］ | － | 要使用的数据库模式。 |
| session＿table | Optional［str］ | － | 用于存储代理、团队和工作流会话的表的名称。 |
| memory＿table | Optional［str］ | － | 用于存储内存的表的名称。 |
| metrics＿table | Optional［str］ | － | 用于存储指标的表的名称。 |
| eval＿table | Optional［str］ | － | 用于存储评估运行数据的表的名称。 |
| knowledge＿table | Optional［str］ | － | 用于存储知识内容的表的名称。 |
| traces＿table | Optional［str］ | － | 用于存储跟踪信息的表的名称。 |
| spans＿table | Optional［str］ | － | 用于存储跨度的表的名称。 |


#### 异步 Postgres 用于工作流

Agno 支持使用异步方式使用PostgreSQL，可通过该类AsyncPostgresDb实现。

####  用法
​
运行 PgVector

安装Docker Desktop并使用以下命令在5532端口运行PgVector：

```bash

docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  agnohq/pgvector:16
```

async_postgres_for_workflow.py


```python
import asyncio

from agno.agent import Agent
from agno.db.postgres import AsyncPostgresDb
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.hackernews import HackerNewsTools
from agno.workflow.step import Step
from agno.workflow.workflow import Workflow

db_url = "postgresql+psycopg_async://ai:ai@localhost:5532/ai"
db = AsyncPostgresDb(db_url=db_url)

# Define agents
hackernews_agent = Agent(
    name="Hackernews Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
    role="Extract key insights and content from Hackernews posts",
)
web_agent = Agent(
    name="Web Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
    role="Search the web for the latest news and trends",
)

# Define research team for complex analysis
research_team = Team(
    name="Research Team",
    members=[hackernews_agent, web_agent],
    instructions="Research tech topics from Hackernews and the web",
)

content_planner = Agent(
    name="Content Planner",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=[
        "Plan a content schedule over 4 weeks for the provided topic and research content",
        "Ensure that I have posts for 3 posts per week",
    ],
)

# Define steps
research_step = Step(
    name="Research Step",
    team=research_team,
)

content_planning_step = Step(
    name="Content Planning Step",
    agent=content_planner,
)

# Create and use workflow
if __name__ == "__main__":
    content_creation_workflow = Workflow(
        name="Content Creation Workflow",
        description="Automated content creation from blog posts to social media",
        db=db,
        steps=[research_step, content_planning_step],
    )
    asyncio.run(
        content_creation_workflow.aprint_response(
            input="AI trends in 2024",
            markdown=True,
        )
    )
```



#### 参数
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | Optional［str］ | － | 数据库实例的ID。默认为UUID。 |
| db＿url | Optional［str］ | － | 要连接的数据库URL。 |
| db＿engine | Optional［AsyncEngine］ | － | 要使用的 SQLAlchemy 异步数据库引擎。 |
| db＿schema | Optional［str］ | － | 要使用的数据库模式。 |
| session＿table | Optional［str］ | － | 用于存储代理、团队和工作流会话的表的名称。 |
| memory＿table | Optional［str］ | － | 用于存储内存的表的名称。 |
| metrics＿table | Optional［str］ | － | 用于存储指标的表的名称。 |
| eval＿table | Optional［str］ | － | 用于存储评估运行数据的表的名称。 |
| knowledge＿table | Optional［str］ | － | 用于存储知识内容的表的名称。 |
| traces＿table | Optional［str］ | － | 用于存储跟踪信息的表的名称。 |
| spans＿table | Optional［str］ | － | 用于存储跨度的表的名称。 |



## MySQL
使用 MySQL 进行代理会话存储和持久化。

Agno 类支持使用MySQL作为数据库MySQLDb。
​

### 用法
```python
from agno.agent import Agent
from agno.db.mysql import MySQLDb

# Setup your Database
db = MySQLDb(db_url="mysql+pymysql://ai:ai@localhost:3306/ai")

# Setup your Agent with the Database
agent = Agent(db=db)
```

### 运行 MySQL
安装Docker Desktop并使用以下命令在3306端口运行MySQL：

```bash
docker run -d \
  --name mysql \
  -e MYSQL_ROOT_PASSWORD=ai \
  -e MYSQL_DATABASE=ai \
  -e MYSQL_USER=ai \
  -e MYSQL_PASSWORD=ai \
  -p 3306:3306 \
  -d mysql:8
```

#### 参数
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | Optional［str］ | － | 数据库实例的ID。默认为UUID。 |
| db＿engine | Optional［Engine］ | － | 要使用的 SQLAlchemy 数据库引擎。 |
| db＿schema | Optional［str］ | － | 要使用的数据库模式。 |
| db＿url | Optional［str］ | － | 要连接的数据库URL。 |
| session＿table | Optional［str］ | － | 用于存储代理、团队和工作流会话的表的名称。 |
| memory＿table | Optional［str］ | － | 用于存储内存的表的名称。 |
| metrics＿table | Optional［str］ | － | 用于存储指标的表的名称。 |
| eval＿table | Optional［str］ | － | 用于存储评估运行数据的表的名称。 |
| knowledge＿table | Optional［str］ | － | 用于存储知识内容的表的名称。 |
| traces＿table | Optional［str］ | － | 用于存储跟踪信息的表的名称。 |
| spans＿table | Optional［str］ | － | 用于存储跨度的表的含称。 |

## 异步 MySQL

使用 MySQL 异步存储代理会话信息。

Agno 支持使用异步MySQL，通过该类AsyncMySQLDb实现。
​
### 用法

```python
from agno.agent import Agent
from agno.db.postgres import AsyncPostgresDb

# Replace with your own connection string, and notice the `async_` prefix
db_url = "postgresql+psycopg_async://ai:ai@localhost:5532/ai"

# Setup your Database
db = AsyncPostgresDb(db_url=db_url)

# Setup your Agent with the Database
agent = Agent(db=db)
```



### 运行 MySQL
安装Docker Desktop并使用以下命令在3306端口运行MySQL：

```bash
docker run -d \
  --name mysql \
  -e MYSQL_ROOT_PASSWORD=ai \
  -e MYSQL_DATABASE=ai \
  -e MYSQL_USER=ai \
  -e MYSQL_PASSWORD=ai \
  -p 3306:3306 \
  -d mysql:8
```

### 参数
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | Optional［str］ | － | 数据库实例的ID。默认为UUID。 |
| db＿url | Optional［str］ | － | 要连接的数据库URL。 |
| db＿engine | Optional［AsyncEngine］ | － | 要使用的 SQLAlchemy 异步数据库引擎。 |
| db＿schema | Optional［str］ | － | 要使用的数据库模式。 |
| session＿table | Optional［str］ | － | 用于存储代理、团队和工作流会话的表的名称。 |
| memory＿table | Optional［str］ | － | 用于存储内存的表的名称。 |
| metrics＿table | Optional［str］ | － | 用于存储指标的表的名称。 |
| eval＿table | Optional［str］ | － | 用于存储评估运行数据的表的名称。 |
| knowledge＿table | Optional［str］ | － | 用于存储知识内容的表的名称。 |
| traces＿table | Optional［str］ | － | 用于存储跟踪信息的表的名称。 |
| spans＿table | Optional［str］ | － | 用于存储跨度的表的名称。 |


## SQLite
使用 SQLite 存储本地代理会话信息。

Agno 支持使用Sqlite作为数据库SqliteDb。


### 用法

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb

# Setup the SQLite database
db = SqliteDb(db_file="tmp/data.db")

# Setup a basic agent with the SQLite database
agent = Agent(db=db)
```

### 参数
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | Optional［str］ | － | 数据库实例的ID。默认为UUID。 |
| db＿engine | Optional［Engine］ | － | 要使用的 SQLAlchemy 数据库引擎。 |
| db＿url | Optional［str］ | － | 要连接的数据库URL。 |
| db＿file | Optional［str］ | － | 要连接的数据库文件。 |
| session＿table | Optional［str］ | － | 用于存储代理、团队和工作流会话的表的名称。 |
| memory＿table | Optional［str］ | － | 用于存储用户记忆的表的名称。 |
| metrics＿table | Optional［str］ | － | 用于存储指标的表的名称。 |
| eval＿table | Optional［str］ | － | 用于存储评估运行数据的表的名称。 |
| knowledge＿table | Optional［str］ | － | 用于存储知识文档数据的表的名称。 |
| traces＿table | Optional［str］ | － | 用于存储跟踪信息的表的名称。 |
| spans＿table | Optional［str］ | － | 用于存储跨度的表的名称。 |


## 异步 SQLite

使用 SQLite 异步存储代理会话信息。

Agno 支持使用异步SQLite，通过该类AsyncSqliteDb实现。
​
### 用法
```python
from agno.agent import Agent
from agno.db.sqlite import AsyncSqliteDb

# Setup the SQLite database
db = AsyncSqliteDb(db_file="tmp/data.db")

# Setup a basic agent with the SQLite database
agent = Agent(db=db)
```

### 参数
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| db＿engine | Optional［Engine］ | － | 要使用的 SQLAlchemy 数据库引擎。 |
| db＿url | Optional［str］ | － | 要连接的数据库URL。 |
| db＿file | Optional［str］ | － | 要连接的数据库文件。 |
| session＿table | Optional［str］ | － | 用于存储代理、团队和工作流会话的表的名称。 |
| memory＿table | Optional［str］ | － | 用于存储用户记忆的表的名称。 |
| metrics＿table | Optional［str］ | － | 用于存储指标的表的名称。 |
| eval＿table | Optional［str］ | － | 用于存储评估运行数据的表的名称。 |
| knowledge＿table | Optional［str］ | － | 用于存储知识文档数据的表的名称。 |
| traces＿table | Optional［str］ | － | 用于存储跟踪信息的表的名称。 |
| spans＿table | Optional［str］ | － | 用于存储跨度的表的名称。 |

## 内存存储

使用内存存储进行测试和开发。

Agno 支持将内存存储与该类一起使用InMemoryDb。这样，您就可以使用所有依赖于数据库的功能，而无需自行设置数据库。

### 用法
```python
from agno.agent import Agent
from agno.db.in_memory import InMemoryDb

# Setup in-memory database
db = InMemoryDb()

# Create agent with database
agent = Agent(db=db)
```

## DynamoDB

使用 DynamoDB 进行代理会话存储和持久化。

Agno 支持使用DynamoDB作为数据库DynamoDb。

### 用法
要连接到 DynamoDB，您需要有效的 AWS 凭证。您可以将其设置为环境变量：
- AWS_REGION要连接的 AWS 区域。
- AWS_ACCESS_KEY_ID：您的 AWS 访问密钥 ID。
- AWS_SECRET_ACCESS_KEY您的 AWS 秘密访问密钥。


```python
from agno.db.dynamo import DynamoDb

# Setup your Database
db = DynamoDb()

# Setup your Agent with the Database
agent = Agent(db=db)
```

### 参数

| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | Optional［str］ | － | 数据库实例的ID。默认为UUID。 |
| db＿client | None | － | 要使用的 DynamoDB 客户端。 |
| region＿name | optional［str］ | － | AWS 区域名称。 |
| aws＿access＿key＿id | optional［str］ | － | AWS 访问密钥 ID。 |
| aws＿secret＿access＿key | optional［str］ | － | AWS 秘密访问密钥。 |
| session＿table | optional［str］ | － | 会话表的名称。 |
| memory＿table | optional［str］ | － | 内存表的名称。 |
| metrics＿table | optional［str］ | － | 指标表的名称。 |
| eval＿table | optional［str］ | － | 评估表的名称。 |
| knowledge＿table | optional［str］ | － | 知识表的名称。 |
| traces＿table | optional［str］ | － | 跟踪表的名称。 |
| spans＿table | optional［str］ | － | spans 表的名称。 |


## MongoDB数据库

使用 MongoDB 进行代理会话存储和持久化。

Agno 支持使用MongoDB作为数据库MongoDb。

### 用法

```python

from agno.agent import Agent
from agno.db.mongo import MongoDb

# MongoDB connection settings
db_url = "mongodb://localhost:27017"

db = MongoDb(db_url=db_url)

# Setup your Agent with the Database
agent = Agent(db=db)
```

### 运行 MongoDB
安装Docker Desktop并使用以下命令在27017端口运行MongoDB：

```bash
docker run -d \
  --name local-mongo \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=mongoadmin \
  -e MONGO_INITDB_ROOT_PASSWORD=secret \
  mongo
```

### 参数
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | Optional［str］ | － | 数据库实例的ID。默认为UUID。 |
| db＿client | Optional［MongoClient］ | － | 要使用的 MongoDB 客户端。 |
| db＿name | Optional［str］ | － | 要使用的数据库名称。 |
| db＿url | Optional［str］ | － | 要连接的数据库URL。 |
| session＿collection | Optional［str］ | － | 用于存储会话的集合名称。 |
| memory＿collection | Optional［str］ | － | 用于存储回忆的收藏集名称。 |
| metrics＿collection | Optional［str］ | － | 用于存储指标的集合名称。 |
| eval＿collection | Optional［str］ | － | 用于存储评估运行结果的集合名称。 |
| knowledge＿collection | Optional［str］ | － | 用于存储知识文档的集合名称。 |
| traces＿collection | Optional［str］ | － | 用于存储跟踪信息的集合名称。 |
| spans＿collection | Optional［str］ | － | 用于存储跨度的集合的名称。 |


## 异步 MongoDB

使用 MongoDB 异步存储代理会话。

Agno 支持使用MongoDB异步方式，通过该类AsyncMongoDb实现。

### 用法

```python
from agno.agent import Agent
from agno.db.mongo import AsyncMongoDb

# MongoDB connection settings
db_url = "mongodb://localhost:27017"

db = AsyncMongoDb(db_url=db_url)

# Setup your Agent with the Database
agent = Agent(db=db)
```

### 运行 MongoDB
安装Docker Desktop并使用以下命令在27017端口运行MongoDB：

```bash
docker run -d \
  --name local-mongo \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=mongoadmin \
  -e MONGO_INITDB_ROOT_PASSWORD=secret \
  mongo
```

### 参数

| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | Optional［str］ | － | 数据库实例的ID。默认为UUID。 |
| db＿client | Optional［MongoClient］ | － | 要使用的 MongoDB 客户端。 |
| db＿name | Optional［str］ | － | 要使用的数据库名称。 |
| db＿url | Optional［str］ | － | 要连接的数据库URL。 |
| session＿collection | Optional［str］ | － | 用于存储会话的集合名称。 |
| memory＿collection | Optional［str］ | － | 用于存储回忆的收藏集名称。 |
| metrics＿collection | Optional［str］ | － | 用于存储指标的集合名称。 |
| eval＿collection | Optional［str］ | － | 用于存储评估运行结果的集合名称。 |
| knowledge＿collection | Optional［str］ | － | 用于存储知识文档的集合名称。 |
| traces＿collection | Optional［str］ | － | 用于存储跟踪信息的集合名称。 |
| spans＿collection | Optional［str］ | － | 用于存储跨度的集合的名称。 |


## JSON 文件作为数据库

使用本地 JSON 文件进行简单的代理会话存储。

Agno 支持使用本地 JSON 文件作为类中的“数据库” JsonDb。这是一种无需设置数据库即可存储代理会话数据的简便方法。

### 用法
```python
from agno.agent import Agent
from agno.db.json import JsonDb

# Setup the JSON database
db = JsonDb(db_path="tmp/json_db")

# Setup your Agent with the Database
agent = Agent(db=db)
```

### 参数
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | Optional［str］ | － | 数据库实例的ID。默认为UUID。 |
| db＿path | Optional［str］ | － | JSON 文件存储目录的路径。 |
| session＿table | Optional［str］ | － | 用于存储会话的 JSON 文件名（不带 ．json 扩展名）。 |
| memory＿table | Optional［str］ | － | 用于存储内存的JSON文件名。 |
| metrics＿table | Optional［str］ | － | 用于存储指标的 JSON 文件名。 |
| eval＿table | Optional［str］ | － | 用于存储评估运行结果的 JSON 文件名。 |
| knowledge＿table | Optional［str］ | － | 用于存储知识内容的JSON文件名。 |
| traces＿table | Optional［str］ | － | 用于存储跟踪信息的 JSON 文件名。 |
| spans＿table | Optional［str］ | － | 用于存储跨度的 JSON 文件名。 |


## SingleStore
使用 SingleStore 进行代理会话存储和持久化。

Agno 支持使用Singlestore作为数据库SingleStoreDb。

您可以按照 Singlestore 的文档开始使用。
​
### 用法

```python
from os import getenv

from agno.agent import Agent
from agno.db.singlestore import SingleStoreDb

# Configure SingleStore DB connection
USERNAME = getenv("SINGLESTORE_USERNAME")
PASSWORD = getenv("SINGLESTORE_PASSWORD")
HOST = getenv("SINGLESTORE_HOST")
PORT = getenv("SINGLESTORE_PORT")
DATABASE = getenv("SINGLESTORE_DATABASE")
db_url = (
    f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}?charset=utf8mb4"
)

# Setup your Database
db = SingleStoreDb(db_url=db_url)

# Create an agent with SingleStore db
agent = Agent(db=db)
```

### 参数
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | Optional［str］ | － | 数据库实例的ID。默认为UUID。 |
| db＿engine | Optional［Engine］ | － | 要使用的 SQLAlchemy 数据库引擎。 |
| db＿schema | Optional［str］ | － | 要使用的数据库模式。 |
| db＿url | Optional［str］ | － | 要连接的数据库URL。 |
| session＿table | Optional［str］ | － | 用于存储代理、团队和工作流会话的表的名称。 |
| memory＿table | Optional［str］ | － | 用于存储内存的表的名称。 |
| metrics＿table | Optional［str］ | － | 用于存储指标的表的名称。 |
| eval＿table | Optional［str］ | － | 用于存储评估运行数据的表的名称。 |
| knowledge＿table | Optional［str］ | － | 用于存储知识内容的表的名称。 |
| traces＿table | Optional［str］ | － | 用于存储跟踪信息的表的名称。 |
| spans＿table | Optional［str］ | － | 用于存储跨度的表的名称。 |


## SurreabDB

使用 SurrealDB 存储代理会话信息。

Agno 支持使用SurreabDB作为数据库SurrealDb。

您可以按照 SurreabDB 的文档开始使用。

使用以下命令在本地运行 SurreabDB：

```bash
docker run --rm --pull always -p 8000:8000 surrealdb/surrealdb:latest start --user root --pass root
```

### 用法
```python
from agno.agent import Agent
from agno.db.surrealdb import SurrealDb

# SurrealDB connection parameters
SURREALDB_URL = "ws://localhost:8000"
SURREALDB_USER = "root"
SURREALDB_PASSWORD = "root"
SURREALDB_NAMESPACE = "agno"
SURREALDB_DATABASE = "surrealdb_for_agent"

creds = {"username": SURREALDB_USER, "password": SURREALDB_PASSWORD}
db = SurrealDb(None, SURREALDB_URL, creds, SURREALDB_NAMESPACE, SURREALDB_DATABASE)

agent = Agent(db=db)
```

### 参数
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | Optional［str］ | － | 数据库实例的ID。默认为UUID。 |
| client | Optional［Union［BlockingWsSurrealConnection， BlockingHttpSurrealConnection］］ | － | 阻塞连接，无论是 HTTP 连接还是 WebSocket 连接。 |
| db＿url | str | － | SurrealDB 连接 URL。 |
| db＿creds | dict［str，str］ | － | 数据库凭据字典（用户名，密码）。 |
| db＿ns | str | － | 要使用的 SurrealDB 命名空间。 |
| db＿db | str | － | 要使用的 SurrealDB 数据库名称。 |
| session＿table | Optional［str］ | － | 用于存储代理、团队和工作流会话的表的名称。 |
| memory＿table | Optional［str］ | － | 用于存储用户记忆的表的名称。 |
| metrics＿table | Optional［str］ | － | 用于存储指标的表的名称。 |
| eval＿table | Optional［str］ | － | 用于存储评估运行数据的表的名称。 |
| knowledge＿table | Optional［str］ | － | 用于存储知识文档数据的表的名称。 |
| culture＿table | Optional［str］ | － | 用于存储文化知识数据的表的名称。 |
| traces＿table | Optional［str］ | － | 用于存储跟踪信息的表的名称。 |
| spans＿table | Optional［str］ | － | 用于存储跨度的表的名称。 |


## Redis
使用 Redis 进行代理会话存储和持久化。

Agno 类支持使用Redis作为数据库RedisDb。
​

### 用法
​
运行 Redis

安装Docker Desktop并使用以下命令在6379端口运行Redis：
```bash
docker run -d \
  --name my-redis \
  -p 6379:6379 \
  redis
```

### redis_for_agent.py
```python
from agno.agent import Agent
from agno.db.redis import RedisDb

# Initialize Redis db (use the right db_url for your setup)
db = RedisDb(db_url="redis://localhost:6379")

# Create agent with Redis db
agent = Agent(db=db)
```

### 参数
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | Optional［str］ | － | 数据库实例的ID。默认为UUID。 |
| redis＿client | Optional［Redis］ | － | 要使用的 Redis 客户端实例。如果未提供，则会创建一个新的客户端。 |
| db＿url | Optional［str］ | － | Redis 连接 URL（例如，＂redis：／／localhost：6379／0＂或 ＂redis：／／user：pass＠host：port／db＂） |
| db＿prefix | str | ＂agno＂ | 所有 Redis 键的前缀。 |
| expire | Optional［int］ | － | Redis 键的 TTL（生存时间），单位为秒。 |
| session＿table | Optional［str］ | － | 用于存储会话的表的名称。 |
| memory＿table | Optional［str］ | － | 用于存储内存的表的名称。 |
| metrics＿table | Optional［str］ | － | 用于存储指标的表的名称。 |
| eval＿table | Optional［str］ | － | 用于存储评估运行结果的表的名称。 |
| knowłedge＿table | Optional［str］ | － | 用于存储知识文档的表的名称。 |
| traces＿table | Optional［str］ | － | 用于存储跟踪信息的表的名称。 |
| spans＿table | Optional［str］ | － | 用于存储跨度的表的名称。 |


## Google Cloud Storage for Agent
Agno 支持使用 Google Cloud Storage (GCS) 作为代理的存储后端GcsJsonDb。此存储后端将会话数据以 JSON blob 的形式存储在 GCS 存储桶中。
​
### 用法
配置代理程序使用 GCS 存储，以启用基于云的会话持久性。

```python
import uuid
import google.auth
from agno.agent import Agent
from agno.db.base import SessionType
from agno.db.gcs_json import GcsJsonDb
from agno.tools.hackernews import HackerNewsTools

# Obtain the default credentials and project id from your gcloud CLI session.
credentials, project_id = google.auth.default()

# Generate a unique bucket name using a base name and a UUID4 suffix.
base_bucket_name = "example-gcs-bucket"
unique_bucket_name = f"{base_bucket_name}-{uuid.uuid4().hex[:12]}"
print(f"Using bucket: {unique_bucket_name}")

# Initialize GCSJsonDb with explicit credentials, unique bucket name, and project.
db = GcsJsonDb(
    bucket_name=unique_bucket_name,
    prefix="agent/",
    project=project_id,
    credentials=credentials,
)

# Initialize the Agno agent with the new storage backend and HackerNews tools.
agent1 = Agent(
    db=db,
    tools=[HackerNewsTools()],
    add_history_to_context=True,
    debug_mode=False,
)

# Execute sample queries.
agent1.print_response("How many people live in Canada?")
agent1.print_response("What is their national anthem called?")

# Create a new agent and make sure it pursues the conversation
agent2 = Agent(
    db=db,
    session_id=agent1.session_id,
    tools=[HackerNewsTools()],
    add_history_to_context=True,
    debug_mode=False,
)

agent2.print_response("What's the name of the country we discussed?")
agent2.print_response("What is that country's national sport?")

```

### gcs_for_agent.py


```python
import uuid
import google.auth
from agno.agent import Agent
from agno.db.base import SessionType
from agno.db.gcs_json import GcsJsonDb
from agno.tools.hackernews import HackerNewsTools

# Obtain the default credentials and project id from your gcloud CLI session.
credentials, project_id = google.auth.default()

# Generate a unique bucket name using a base name and a UUID4 suffix.
base_bucket_name = "example-gcs-bucket"
unique_bucket_name = f"{base_bucket_name}-{uuid.uuid4().hex[:12]}"
print(f"Using bucket: {unique_bucket_name}")

# Initialize GCSJsonDb with explicit credentials, unique bucket name, and project.
db = GcsJsonDb(
    bucket_name=unique_bucket_name,
    prefix="agent/",
    project=project_id,
    credentials=credentials,
)

# Initialize the Agno agent with the new storage backend and HackerNews tools.
agent1 = Agent(
    db=db,
    tools=[HackerNewsTools()],
    add_history_to_context=True,
    debug_mode=False,
)

# Execute sample queries.
agent1.print_response("How many people live in Canada?")
agent1.print_response("What is their national anthem called?")

# Create a new agent and make sure it pursues the conversation
agent2 = Agent(
    db=db,
    session_id=agent1.session_id,
    tools=[HackerNewsTools()],
    add_history_to_context=True,
    debug_mode=False,
)

agent2.print_response("What's the name of the country we discussed?")
agent2.print_response("What is that country's national sport?")
```

### 先决条件
1. Google Cloud SDK 设置

安装Google Cloud SDK

运行gcloud init以配置您的帐户和项目


2. GCS权限

请确保您的帐户拥有足够的权限（例如，存储管理员权限）来创建和管理 GCS 存储桶：

```bash

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="user:YOUR_EMAIL@example.com" \
    --role="roles/storage.admin"
```

3. 验证

使用 gcloud CLI 会话中的默认凭据：
```bash
gcloud auth application-default login
```

或者，如果使用服务帐户，请将GOOGLE_APPLICATION_CREDENTIALS环境变量设置为服务帐户 JSON 文件的路径。
4.  Python依赖项

安装所需的 Python 软件包：
```bash
pip install google-auth google-cloud-storage openai ddgs
```


使用 Docker 进行设置

对于不使用真实 GCS 的本地测试，可以使用fake-gcs-server。

创建docker-compose.yml文件：

```yaml
version: '3.8'
services:
  fake-gcs-server:
    image: fsouza/fake-gcs-server:latest
    ports:
      - "4443:4443"
    command: ["-scheme", "http", "-port", "4443", "-public-host", "localhost"]
    volumes:
      - ./fake-gcs-data:/data


```


启动模拟 GCS 服务器：

```bash

docker-compose up -d
```


2. 使用 Docker 的 Fake GCS

设置环境变量，将 API 调用定向到模拟器：
```bash

export STORAGE_EMULATOR_HOST="http://localhost:4443"
python gcs_for_agent.py
```
使用 Fake GCS 时，不会强制进行身份验证，客户端会自动检测模拟器端点。


### 参数
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | Optional［str］ | － | 数据库实例的ID。默认为UUID。 |
| bucket＿name | str | － | 用于存储 JSON 文件的 GCS 存储桶名称。 |
| prefix | Optional［str］ | － | 用于组织存储桶中文件的路径前缀。默认为＂agno／＂。 |
| session＿table | Optional［str］ | － | 用于存储会话的 JSON 文件名（不带 ．json 扩展名）。 |
| memory＿table | Optional［str］ | － | 用于存储用户记忆的JSON文件名。 |
| metrics＿table | Optional［str］ | － | 用于存储指标的 JSON 文件名。 |
| eval＿table | Optional［str］ | － | 用于存储评估运行结果的 JSON 文件名。 |
| knowledge＿table | Optional［str］ | － | 用于存储知识内容的JSON文件名。 |
| traces＿table | Optional［str］ | － | 用于存储跟踪信息的 JSON 文件名。 |
| spans＿table | Optional［str］ | － | 用于存储跨度的 JSON 文件名。 |
| project | Optional［str］ | － | GCP 项目 ID。如果为 None，则使用默认项目。 |
| credentials | Optional［Any］ | － | GCP 凭据。如果未指定，则使用默认凭据。 |


## Firestore
使用 Firestore 进行代理会话存储和持久化。

Agno 类支持使用Firestore作为数据库FirestoreDb。

您可以按照 Firestore 的入门指南开始使用。


### 用法
您需要project_id为该类提供一个参数FirestoreDb。Firestore 将使用您的 Google Cloud 凭据自动连接。


```python

from agno.agent import Agent
from agno.db.firestore import FirestoreDb

PROJECT_ID = "agno-os-test"  # Use your project ID here

# Setup the Firestore database
db = FirestoreDb(project_id=PROJECT_ID)

# Setup your Agent with the Database
agent = Agent(db=db)
```


### 先决条件
请确保您的 gcloud 项目已启用 Firestore。请参阅Firestore 文档。

安装依赖项：uv pip install openai google-cloud-firestore agno

请确保您的 gcloud 项目已设置完毕，并且您拥有访问 Firestore 所需的权限。

### 参数
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | Optional［str］ | － | 数据库实例的ID。默认为UUID。 |
| db＿client | Optional［Client］ | － | 要使用的 Firestore 客户端。 |
| project＿id | Optional［str］ | － | Firestore 的 GCP 项目 ID。 |
| session＿collection | Optional［str］ | － | 用于存储会话的集合名称。 |
| memory＿collection | Optional［str］ | － | 用于存储回忆的收藏集名称。 |
| metrics＿collection | Optional［str］ | － | 用于存储指标的集合名称。 |
| eval＿collection | Optional［str］ | － | 用于存储评估运行结果的集合名称。 |
| knowledge＿collection | Optional［str］ | － | 用于存储知识文档的集合名称。 |
| traces＿collection | Optional［str］ | － | 用于存储跟踪信息的集合名称。 |
| spans＿collection | Optional［str］ | － | 用于存储跨度的集合的名称。 |


## Supabase
使用 Supabase PostgreSQL 进行代理会话存储。

Agno 支持将Supabase与PostgresDb该类一起使用。

您可以按照 Supabase 的入门指南开始使用。

您可以在课程章节中阅读更多关于该PostgresDb课程的信息。


### 用法

```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb
from os import getenv

# Get your Supabase project and password
SUPABASE_PROJECT = getenv("SUPABASE_PROJECT")
SUPABASE_PASSWORD = getenv("SUPABASE_PASSWORD")

SUPABASE_DB_URL = (
    f"postgresql://postgres:{SUPABASE_PASSWORD}@db.{SUPABASE_PROJECT}:5432/postgres"
)

# Setup the Supabase database
db = PostgresDb(db_url=SUPABASE_DB_URL)

# Setup your Agent with the Database
agent = Agent(db=db)
```

### 参数
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | Optional［str］ | － | 数据库实例的ID。默认为UUID。 |
| db＿url | Optional［str］ | － | 要连接的数据库URL。 |
| db＿engine | Optional［Engine］ | － | 要使用的 SQLAlchemy 数据库引擎。 |
| db＿schema | Optional［str］ | － | 要使用的数据库模式。 |
| session＿table | Optional［str］ | － | 用于存储代理、团队和工作流会话的表的名称。 |
| memory＿table | Optional［str］ | － | 用于存储内存的表的名称。 |
| metrics＿table | Optional［str］ | － | 用于存储指标的表的名称。 |
| eval＿table | Optional［str］ | － | 用于存储评估运行数据的表的名称。 |
| knowledge＿table | Optional［str］ | － | 用于存储知识内容的表的名称。 |
| traces＿table | Optional［str］ | － | 用于存储跟踪信息的表的名称。 |
| spans＿table | Optional［str］ | － | 用于存储跨度的表的名称。 |


## Neon
使用 Neon 无服务器 PostgreSQL 进行代理会话存储。

Agno 支持将Neon与PostgresDb该类一起使用。

您可以按照 Neon 的入门指南开始使用。

您也可以在课程章节中阅读更多关于该PostgresDb课程的信息。


### 用法

```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb
from os import getenv

# Get your Neon database URL
NEON_DB_URL = getenv("NEON_DB_URL")

# Setup the Neon database
db = PostgresDb(db_url=NEON_DB_URL)

# Setup your Agent with the Database
agent = Agent(db=db)
```

### 参数

| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | Optional［str］ | － | 数据库实例的ID。默认为UUID。 |
| db＿url | Optional［str］ | － | 要连接的数据库URL。 |
| db＿engine | Optional［Engine］ | － | 要使用的 SQLAlchemy 数据库引擎。 |
| db＿schema | Optional［str］ | － | 要使用的数据库模式。 |
| session＿table | Optional［str］ | － | 用于存储代理、团队和工作流会话的表的名称。 |
| memory＿table | Optional［str］ | － | 用于存储内存的表的名称。 |
| metrics＿table | Optional［str］ | － | 用于存储指标的表的名称。 |
| eval＿table | Optional［str］ | － | 用于存储评估运行数据的表的名称。 |
| knowledge＿table | Optional［str］ | － | 用于存储知识内容的表的名称。 |
| traces＿table | Optional［str］ | － | 用于存储跟踪信息的表的名称。 |
| spans＿table | Optional［str］ | － | 用于存储跨度的表的名称。 |


## 选择自定义表名

Agno 允许您在使用数据库时自定义表名，从而在组织数据存储方面提供灵活性。
​
### 用法
初始化数据库连接时，请指定自定义表名。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb

# Setup the SQLite database with custom table names
db = SqliteDb(
    db_file="tmp/data.db",
    # Selecting which tables to use
    session_table="agent_sessions",
    memory_table="agent_memories",
    metrics_table="agent_metrics",
)

# Setup a basic agent with the SQLite database
agent = Agent(
    db=db,
    update_memory_on_run=True,
    add_history_to_context=True,
    add_datetime_to_context=True,
)

# The Agent sessions and runs will now be stored in SQLite with custom table names
agent.print_response("How many people live in Canada?")
agent.print_response("And in Mexico?")
agent.print_response("List my messages one by one")
```

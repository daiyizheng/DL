# 学习

## 机器学习

能够通过每次交互学习和改进的智能体。

Agno 通过将智能体与学习存储库相结合，将智能体转化为学习机器。学习存储库是持久化的后端，可以随时间推移捕获用户画像、记忆和知识。
```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=SqliteDb(db_file="tmp/agents.db"),
    learning=True,
)
```

一行代码。您的代理程序现在可以记住用户，并随着时间的推移而不断改进。
​
### 学习商店
每家商店都蕴含着不同类型的知识：

| 店铺 | 它捕捉到的内容 | 范围 |
| :--- | :--- | :--- |
| 用户个人资料 | 结构化事实（姓名、角色、偏好） | 每用户 |
| 用户内存 | 从对话中得出的非结构化观察 | 每用户 |
| 会话上下文 | 本届会议的目标、计划和进展 | 每节课 |
| 实体记忆 | 关于外部事物（公司、项目、人员）的事实 | 可配置 |
| 已习得的知识 | 跨用户共享的见解 | 可配置 |
| 决策日志 | 审计和学习的决策及其理由 | 每代理人 |


### 命名空间
部分商店支持通过以下方式配置共享范围namespace：
| 命名空间 | 谁可以访问 |
| :--- | :--- |
| ＂user＂ | 仅限当前用户 |
| ＂global＂ | 所有人（默认） |
| "Custom" | 显式分组（例如＂engineering＂，，＂sales＿west＂） |


### 维护
馆长负责维护记忆的健康：

```python
lm = agent.get_learning_machine()

# Remove memories older than 90 days
lm.curator.prune(user_id="alice", max_age_days=90)

# Remove duplicates
lm.curator.deduplicate(user_id="alice")
```

### 快速入门

启用智能体的学习功能。

#### 促进学习
最简单的方法：设置learning=True。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=SqliteDb(db_file="tmp/agents.db"),
    learning=True,
)
```

这使得在“始终”模式下能够提取用户配置文件和用户记忆。代理会自动捕获信息并在以后的会话中调用这些信息。

### 测试一下

```python
# Session 1: Share information
agent.print_response(
    "Hi! I'm Sarah, I work at Acme Corp as a data scientist.",
    user_id="sarah@acme.com",
    session_id="session_1",
)

# Session 2: Agent remembers
agent.print_response(
    "What do you know about me?",
    user_id="sarah@acme.com",
    session_id="session_2",
)
```

### 选择学习内容
为了更好地控制，请单独配置各个门店：

```python
from agno.learn import LearningMachine

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=LearningMachine(
        user_profile=True,      # Structured facts (name, role, preferences)
        user_memory=True,       # Unstructured observations
        session_context=True,   # Session summary and goals
        entity_memory=False,    # Facts about external entities
        learned_knowledge=False # Insights across users (requires Knowledge)
    ),
)

```

有关各学习商店的详细信息，请参阅“学习商店”页面。


### 选择学习方式
每个门店可以使用不同的学习模式：


```python

from agno.learn import (
    LearningMachine,
    LearningMode,
    UserProfileConfig,
    UserMemoryConfig,
)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=LearningMachine(
        user_profile=UserProfileConfig(mode=LearningMode.ALWAYS),
        user_memory=UserMemoryConfig(mode=LearningMode.AGENTIC),
    ),
)
```
有关各学习商店的详细信息，请参阅“学习商店”页面。

| 模式 | 工作原理 |
| :--- | :--- |
| 总是 | 提取过程在每次响应后自动运行 |
| 代理 | 代理人收到工具后决定保存哪些内容 |
| 提出 | 代理提出学习方案，您批准后再保存。 |


### 生产数据库
生产环境请使用 PostgreSQL：

```python
from agno.db.postgres import PostgresDb

db = PostgresDb(db_url="postgresql+psycopg://ai:ai@localhost:5532/ai")

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=True,
)
```

## 学习商店

每家商店都蕴含着不同类型的知识。

机器学习机器协调多个存储库，每个存储库都针对特定类型的知识进行了优化。


| 店铺 | 它捕捉到的内容 | 范围 | 用例 |
| :--- | :--- | :--- | :--- |
| 用户个人资料 | 结构化字段（姓名、角色、偏好） | 每用户 | 个性化 |
| 用户内存 | 非结构化的观察和事实 | 每用户 | 上下文保留 |
| 会话上下文 | 目标、计划和进展 | 每节课 | 长时间运行的任务 |
| 实体记忆 | 关于外部实体的事实 | 可配置 | 知识图谱 |
| 已习得的知识 | 跨用户共享的见解 | 可配置 | 团队整体改进 |
| 决策日志 | 有理有据的决策 | 每代理人 | 审计与学习 |


### 指南


![](https://i-blog.csdnimg.cn/direct/e7844e7c754f419e9ea475430e7d1d39.png)


## 用户个人资料
关于用户的结构化事实。

用户个人资料存储会捕获有关用户的结构化字段：姓名、首选名称以及您定义的自定义字段。

| 方面 | 价值 |
| :--- | :--- |
| 范围 | 每用户 |
| 持久性 | 永久有效（随着新信息的获取而更新） |
| 默认模式 | 总是 |
| 支持的模式 | 永远，代理 |


### 基本用法
```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.learn import LearningMachine
from agno.models.openai import OpenAIResponses

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=PostgresDb(db_url="postgresql+psycopg://ai:ai@localhost:5532/ai"),
    learning=LearningMachine(user_profile=True),
)

# Session 1: Share information
agent.print_response(
    "Hi! I'm Alice Chen, but please call me Ali.",
    user_id="alice@example.com",
    session_id="session_1",
)

# Session 2: Profile is recalled automatically
agent.print_response(
    "What's my name?",
    user_id="alice@example.com",
    session_id="session_2",
)
```

### 始终模式
每次响应后都会自动进行数据提取。代理看不到任何工具。

```python
from agno.learn import LearningMachine, LearningMode, UserProfileConfig

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=LearningMachine(
        user_profile=UserProfileConfig(mode=LearningMode.ALWAYS),
    ),
)
```
权衡：每次交互需要额外调用LLM。

### 代理模式
代理人收到update_profile工具后，决定何时更新。

```python
from agno.learn import LearningMachine, LearningMode, UserProfileConfig

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=LearningMachine(
        user_profile=UserProfileConfig(mode=LearningMode.AGENTIC),
    ),
)

agent.print_response(
    "Please remember that my name is Bob Smith.",
    user_id="bob@example.com",
)
```

权衡：代理可能会错过隐含的个人资料信息。

### 默认字段
| 场地 | 描述 |
| :--- | :--- |
| name | 姓名 |
| preferred＿name | 他们希望别人怎么称呼他们 |


### 自定义模式
扩展域的基本架构：

```python
from dataclasses import dataclass, field
from typing import Optional
from agno.learn.schemas import UserProfile

@dataclass
class CustomerProfile(UserProfile):
    company: Optional[str] = field(
        default=None,
        metadata={"description": "Company or organization"}
    )
    plan_tier: Optional[str] = field(
        default=None,
        metadata={"description": "Subscription tier: free | pro | enterprise"}
    )
    role: Optional[str] = field(
        default=None,
        metadata={"description": "Job title or role"}
    )
    timezone: Optional[str] = field(
        default=None,
        metadata={"description": "User's timezone"}
    )

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=LearningMachine(
        user_profile=UserProfileConfig(schema=CustomerProfile),
    ),
)
```

### 访问个人资料数据

```python
lm = agent.get_learning_machine()

# Get profile
profile = lm.user_profile_store.get(user_id="alice@example.com")
print(profile.name)
print(profile.preferred_name)

# Debug output
lm.user_profile_store.print(user_id="alice@example.com")
```


### 上下文注入
配置文件会自动注入到系统提示符中：

```xml
<user_profile>
Name: Alice Chen
Preferred Name: Ali
Company: Acme Corp
Role: Data Scientist
</user_profile>
```

无需手动构建上下文。

### 用户配置文件与用户内存

| 用户个人资料 | 用户内存 |
| :--- | :--- |
| 结构化字段 | 非结构化文本 |
| 固定模式 | 灵活的观察 |
| 已就地更新 | 随着时间的推移而增加 |
| 准确回忆 | 语义搜索 |

用户个人资料包含：姓名、公司、职位、偏好设置等信息，并可定义相应值。
使用用户记忆来记录诸如“喜欢详细的解释”或“从事机器学习项目”之类的观察结果。


## 用户内存

关于用户的非结构化观察。

用户记忆存储捕获有关用户的非结构化观察结果：偏好、行为和上下文，这些内容不适合放入结构化的个人资料字段中。

| 方面 | 价值 |
| :--- | :--- |
| 范围 | 每用户 |
| 持久性 | 长期（可选择进行内容管理） |
| 默认模式 | 总是 |
| 支持的模式 | 永远，代理 |


### 基本用法
```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.learn import LearningMachine
from agno.models.openai import OpenAIResponses

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=PostgresDb(db_url="postgresql+psycopg://ai:ai@localhost:5532/ai"),
    learning=LearningMachine(user_memory=True),
)

# Session 1: Share preferences
agent.print_response(
    "I prefer code examples over explanations. Also, I'm working on a machine learning project.",
    user_id="alice@example.com",
    session_id="session_1",
)

# Session 2: Memory is recalled
agent.print_response(
    "Explain async/await in Python",
    user_id="alice@example.com",
    session_id="session_2",
)
```

该代理知道要包含代码示例，并且可能与机器学习上下文相关。

### 始终模式
每次响应后，记忆都会自动提取。

```python
from agno.learn import LearningMachine, LearningMode, UserMemoryConfig

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=LearningMachine(
        user_memory=UserMemoryConfig(mode=LearningMode.ALWAYS),
    ),
)
```
权衡：每次交互需要额外调用LLM。

### 代理模式
代理程序会接收用于显式管理内存的工具。

```python

from agno.learn import LearningMachine, LearningMode, UserMemoryConfig

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=LearningMachine(
        user_memory=UserMemoryConfig(mode=LearningMode.AGENTIC),
    ),
)

agent.print_response(
    "Remember that I always want to see error handling in code examples.",
    user_id="alice@example.com",
)
```

### 被捕获的内容
| 有利于用户记忆 | 更适合用户个人资料 |
| :--- | :--- |
| ＂更喜欢详细的解释＂ | 姓名：＂陈爱丽丝＂ |
| ＂正在进行机器学习项目＂ | 公司：＂Acme Corp＂ |
| ＂异步代码编写困难＂ | 职位：＂数据科学家＂ |
| ＂使用 VS Code＂ | 时区：＂PST＂ |

### 内存数据模型
| 场地 | 描述 |
| :--- | :--- |
| user＿id | 此内存属于用户 |
| memories | 内存条目列表（ id ，，content 可选元数据） |
| agent＿id | 审计跟踪的代理上下文 |
| team＿id | 审计跟踪的团队背景 |
| created＿at | 创建时 |
| updated＿at | 最后更新 |


### 访问内存

```python
lm = agent.get_learning_machine()

# Get all memories
memories = lm.user_memory_store.get(user_id="alice@example.com")
if memories:
    for memory in memories.memories:
        print(f"- {memory.get('content')}")

# Debug output
lm.user_memory_store.print(user_id="alice@example.com")
```

### 上下文注入
相关记忆被注入到系统提示符中：

```xml
<user_memory>
- Prefers code examples over explanations
- Working on a machine learning project
- Uses Python 3.11
- Prefers concise responses
</user_memory>
```

### 策展
随着时间的推移，记忆会不断积累。使用管理员来维护这些记忆：

```python
lm = agent.get_learning_machine()

# Remove memories older than 90 days
lm.curator.prune(user_id="alice@example.com", max_age_days=90)

# Remove duplicates
lm.curator.deduplicate(user_id="alice@example.com")
```


### 与用户个人资料结合
同时使用这两个平台来全面了解用户：

```python

from agno.learn import LearningMachine

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=LearningMachine(
        user_profile=True,  # Structured: name, company
        user_memory=True,   # Unstructured: preferences, context
    ),
)
```

### 会话上下文

积极参与活动的目标、计划和进展。

会话上下文存储记录了对话的当前状态：讨论了什么、目标是什么以及取得了哪些进展。与其他会累积数据的存储不同，会话上下文是一个快照，每次更新都会被替换。

| 方面 | 价值 |
| :--- | :--- |
| 范围 | 每节课 |
| 持久性 | 会话生命周期（更新时替换） |
| 默认模式 | 总是 |
| 支持的模式 | 总是 |

### 基本用法

```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.learn import LearningMachine
from agno.models.openai import OpenAIResponses

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=PostgresDb(db_url="postgresql+psycopg://ai:ai@localhost:5532/ai"),
    learning=LearningMachine(session_context=True),
)

# Session tracks what's being discussed
agent.print_response(
    "I'm designing a REST API for a todo app. Should I use PUT or PATCH for updates?",
    user_id="alice@example.com",
    session_id="api_design",
)

# Later in the session, context is maintained
agent.print_response(
    "What about the delete endpoint?",
    user_id="alice@example.com",
    session_id="api_design",
)
```

代理了解 REST API 设计的最新背景信息。


### 摘要模式
默认行为。无需详细计划即可捕捉对话的精髓。

```python
from agno.learn import LearningMachine, SessionContextConfig

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=LearningMachine(
        session_context=SessionContextConfig(),
    ),
)
```
记录内容包括：正在进行的工作、已做出的关键决策、当前状态、未解决的问题。

### 规划模式
启用规划功能，以便跟踪目标、规划步骤和进度。

```python
from agno.learn import LearningMachine, SessionContextConfig

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=LearningMachine(
        session_context=SessionContextConfig(enable_planning=True),
    ),
)

agent.print_response(
    "Help me deploy a Python app to production. Give me the steps.",
    user_id="alice@example.com",
    session_id="deploy_app",
)

# Later, progress is tracked
agent.print_response(
    "Done with step 1. What's next?",
    user_id="alice@example.com",
    session_id="deploy_app",
)

```
### 数据模型
| 场地 | 描述 |
| :--- | :--- |
| session＿id | 唯一会话标识符 |
| user＿id | 此会话属于该用户 |
| summary | 讨论过的内容 |
| goal | 用户试图完成什么（规划模式） |
| plan | 实现目标的步骤（计划模式） |
| progress | 已完成步骤（计划模式） |
| created＿at | 创建时 |
| updated＿at | 最后更新 |

### 访问会话上下文

```python
lm = agent.get_learning_machine()

context = lm.session_context_store.get(session_id="api_design")
if context:
    print(f"Summary: {context.summary}")
    if context.goal:
        print(f"Goal: {context.goal}")

# Debug output
lm.session_context_store.print(session_id="api_design")
```
### 上下文注入
会话上下文被注入到系统提示符中：

```xml
<session_context>
Summary: Helping user design a REST API for a todo app. Discussed resource naming conventions. Currently exploring HTTP methods for CRUD operations.

Goal: Design complete REST API for todo application

Plan:
  1. Define resource endpoints
  2. Choose HTTP methods for each operation
  3. Design request/response schemas
  4. Add authentication

Completed:
  ✓ Define resource endpoints
</session_context>
```

### 何时使用
在以下情况下，会话上下文至关重要：
- 消息历史记录会被截断：长时间的对话会丢失早期上下文。
- 会话恢复：用户休息后返回。
- 复杂的多步骤任务：跟踪整个工作流程的进度
- 交接：其他代理人或人员需要了解状态

### 与其他店铺合并
会话上下文与用户级存储配合使用效果很好：
```python
from agno.learn import LearningMachine, SessionContextConfig

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=LearningMachine(
        user_profile=True,                                  # Who the user is
        session_context=SessionContextConfig(enable_planning=True),  # Current state
    ),
)
```
长期用户知识加上短期会话状态。


## 实体记忆

关于公司、项目和人员的事实。

实体记忆库用于存储关于外部实体（包括公司、人员、项目和系统）的结构化知识。您可以将其视为您代理人的专业人脉库，它会随着时间的推移不断积累知识。

| 方面 | 价值 |
| :--- | :--- |
| 范围 | 可配置（全局、用户或自定义命名空间） |
| 持久性 | 长期 |
| 默认模式 | 总是 |
| 支持的模式 | 永远，代理 |


### 基本用法

```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.learn import LearningMachine
from agno.models.openai import OpenAIResponses

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=PostgresDb(db_url="postgresql+psycopg://ai:ai@localhost:5532/ai"),
    learning=LearningMachine(entity_memory=True),
)

# Entities are extracted automatically
agent.print_response(
    "Just met with Acme Corp. They're a fintech startup in SF, "
    "50 employees. CTO is Jane Smith. They use Python and Postgres.",
    user_id="sales@example.com",
    session_id="session_1",
)

# Later, entity knowledge is recalled
agent.print_response(
    "What do we know about Acme Corp?",
    user_id="sales@example.com",
    session_id="session_2",
)
```

### 始终模式
实体是从对话中自动提取的。

```python
from agno.learn import LearningMachine, LearningMode, EntityMemoryConfig

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=LearningMachine(
        entity_memory=EntityMemoryConfig(mode=LearningMode.ALWAYS),
    ),
)
```

### 代理模式
代理程序会接收用于显式管理实体的工具。

```python
from agno.learn import LearningMachine, LearningMode, EntityMemoryConfig

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=LearningMachine(
        entity_memory=EntityMemoryConfig(mode=LearningMode.AGENTIC),
    ),
)

agent.print_response(
    "Create an entry for Acme Corp - they're a fintech startup with 50 employees.",
    user_id="sales@example.com",
)
```

可用工具： search_entities, create_entity, update_entity, add_fact, update_fact, delete_fact, add_event, add_relationship

### 数据模型


| 场地 | 描述 |
| :--- | :--- |
| entity＿id | 唯一标识符（例如，＂acme＿corp＂） |
| entity＿type | 类别：＂公司＂、＂个人＂、＂项目＂ |
| name | 显示名称 |
| description | 简述 |
| properties | 键值元数据 |
| facts | 永恒的真理 |
| events | 时限性事件 |
| relationships | 与其他实体的联系 |

### 访问实体内存

```python
lm = agent.get_learning_machine()

# Search for entities
entities = lm.entity_memory_store.search(
    query="acme",
    entity_type="company",
    limit=10
)

for entity in entities:
    print(f"{entity.name}: {entity.facts}")

# Debug output
lm.entity_memory_store.print(entity_id="acme_corp", entity_type="company")
```

### 上下文注入
相关实体被注入到系统提示符中：

```xml
<entity_memory>
**Acme Corp** (company)
Properties: industry: fintech, size: 50 employees

Facts:
  - Uses PostgreSQL and Redis for their data layer
  - Headquarters in San Francisco

Events:
  - Launched v2.0 with new ML features (2025-01-15)
  - Closed $50M Series B led by Sequoia (2024-Q3)

Relationships:
  - CEO: jane_smith
  - competitor_of: beta_inc
</entity_memory>
```

### 命名空间
控制谁可以访问实体数据：
```python
from agno.learn import EntityMemoryConfig

# Global: shared with everyone (default)
entity_memory=EntityMemoryConfig(namespace="global")

# User: private per user
entity_memory=EntityMemoryConfig(namespace="user")

# Custom: explicit grouping
entity_memory=EntityMemoryConfig(namespace="sales_team")
```

### 事实与事件
| 运用事实 | 使用事件 |
| :--- | :--- |
| 技术栈 | 产品发布 |
| 总部所在地 | 融资轮 |
| 员工人数 | 故障或事故 |
| 行业／领域 | 宣布建立合作伙伴关系 |
| 定价模式 | 重要会议 |

### 关系类型
实体链接的常见模式：

- 人们：CEO，CTO，engineer_at，founder，reports_to
- 公司：competitor_of，，，partner_of​acquired_by​subsidiary_of
- 项目：uses，，，depends_on​integrates_with​owned_by


## 已习得的知识
可跨用户共享的洞察。

学习知识库捕获可重用的洞察、模式和最佳实践，这些洞察、模式和实践适用于所有用户和会话。借助语义搜索，智能体可以自动查找并应用相关知识。

| 方面 | 价值 |
| :--- | :--- |
| 范围 | 可配置（全局、用户或自定义命名空间） |
| 持久性 | 长期 |
| 默认模式 | 代理 |
| 支持的模式 | 总是，主动，提议 |
| 需要 | 包含向量数据库的知识库 |

### 先决条件
学习知识需要一个用于语义搜索的知识库：

```python
from agno.knowledge import Knowledge
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.vectordb.pgvector import PgVector, SearchType

knowledge = Knowledge(
    vector_db=PgVector(
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        table_name="learned_knowledge",
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
)
```

### 基本用法

```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.learn import LearningMachine
from agno.models.openai import OpenAIResponses

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=PostgresDb(db_url="postgresql+psycopg://ai:ai@localhost:5532/ai"),
    learning=LearningMachine(
        knowledge=knowledge,
        learned_knowledge=True,
    ),
)

# User 1 saves an insight
agent.print_response(
    "Save this: When comparing cloud providers, always check egress costs first - "
    "they can be 10x different between providers.",
    user_id="alice@example.com",
)

# User 2 benefits from the insight
agent.print_response(
    "I'm choosing between AWS and GCP for our data platform. What should I consider?",
    user_id="bob@example.com",
)
```

### 代理模式
代理程序会接收用于显式管理知识的工具。

```python
from agno.learn import LearningMachine, LearningMode, LearnedKnowledgeConfig

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=LearningMachine(
        knowledge=knowledge,
        learned_knowledge=LearnedKnowledgeConfig(mode=LearningMode.AGENTIC),
    ),
)
```
可用工具search_learnings：save_learning
代理会在回答问题和保存之前进行搜索（以避免重复）。
### 提议模式
代理会提出学习结果供用户确认后再保存。
```python
from agno.learn import LearningMachine, LearningMode, LearnedKnowledgeConfig

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=LearningMachine(
        knowledge=knowledge,
        learned_knowledge=LearnedKnowledgeConfig(mode=LearningMode.PROPOSE),
    ),
)

agent.print_response(
    "That's a great insight about Docker networking. We should remember that.",
    user_id="alice@example.com",
)
# Agent proposes the learning, user confirms before it's saved
```

### 始终模式
每次响应后，系统都会自动提取学习数据。

```python
from agno.learn import LearningMachine, LearningMode, LearnedKnowledgeConfig

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=LearningMachine(
        knowledge=knowledge,
        learned_knowledge=LearnedKnowledgeConfig(mode=LearningMode.ALWAYS),
    ),
)

```

权衡：每次互动需要额外调用 LLM，可能会损失一些价值不高的洞察。
​
### 数据模型

| 场地 | 描述 |
| :--- | :--- |
| title | 简短、可搜索的标题 |
| learning | 真正的洞察力 |
| context | 这适用于何时／何地 |
| tags | 组织类别 |
| namespace | 共享范围 |
| user＿id | 所有者（如果命名空间＝＂user＂） |
| created＿at | 被捕获时 |


### 应该节省什么

| 值得保存 | 不要保存 |
| :--- | :--- |
| 非显而易见的发现 | 原始事实或数据 |
| 可重用模式 | 用户特定偏好 |
| 领域特定见解 | 常识 |
| 问题解决方法 | 对话摘要 |
| 最佳实践 | 临时信息 |

很好的例子：
“在比较云服务提供商时，务必先查看出口流量成本——它们之间的差异非常大（AWS：
0.09/`GB`地质公园，GCP：0.12/`GB`，Cloudflare R2：免费）。”

糟糕的例子：
“AWS 存在出口流量费用。”


### 获取已学知识
```python
lm = agent.get_learning_machine()

# Search for relevant learnings
results = lm.learned_knowledge_store.search(query="cloud costs", limit=5)
for result in results:
    print(f"{result.title}: {result.learning}")

# Debug output
lm.learned_knowledge_store.print(query="cloud costs")
```

### 上下文注入
通过语义搜索注入相关知识：


```xml
<relevant_learnings>
**Cloud egress cost variations**
Context: When selecting cloud providers for data-intensive workloads
Insight: Always check egress costs first - they can be 10x different between providers.

**API rate limiting strategies**
Context: When designing APIs with high traffic
Insight: Use token bucket algorithm for rate limiting - it handles bursts better than fixed windows.
</relevant_learnings>
```

### 命名空间
控制知识共享：

```python
from agno.learn import LearnedKnowledgeConfig

# Global: shared with all users (default)
learned_knowledge=LearnedKnowledgeConfig(namespace="global")

# User: private per user
learned_knowledge=LearnedKnowledgeConfig(namespace="user")

# Custom: team or domain-specific
learned_knowledge=LearnedKnowledgeConfig(namespace="engineering")
```


### 与其他店铺合并
```python
from agno.learn import LearningMachine

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=LearningMachine(
        knowledge=knowledge,
        user_profile=True,       # Who the user is
        user_memory=True,        # User's preferences
        learned_knowledge=True,  # Collective insights
    ),
)
```
基于集体智慧的个性化回复。

### 决策日志
决策需有理由，以便进行审计和学习。

决策日志存储记录智能体做出的决策，包括决策理由、上下文和结果。它可用于审核智能体行为、调试意外结果以及构建反馈回路。

| 方面 | 价值 |
| :--- | :--- |
| 范围 | 每代理人 |
| 持久性 | 长期 |
| 默认模式 | 总是（DecisionLogConfig（）），主动的（ decision＿log＝True ） |
| 支持的模式 | 永远，代理 |

### 基本用法

```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.learn import LearningMachine, DecisionLogConfig
from agno.models.openai import OpenAIChat

agent = Agent(
    id="my-agent",
    model=OpenAIChat(id="gpt-4o"),
    db=PostgresDb(db_url="postgresql+psycopg://ai:ai@localhost:5532/ai"),
    learning=LearningMachine(
        decision_log=DecisionLogConfig(),  # defaults to LearningMode.ALWAYS
    ),
    instructions=[
        "When you make a significant choice, use log_decision to record it.",
        "Include your reasoning and alternatives you considered.",
    ],
)

agent.print_response(
    "I need help choosing between Python and JavaScript for web scraping.",
    session_id="session_1",
)

# View logged decisions
lm = agent.get_learning_machine()
lm.decision_log_store.print(agent_id="my-agent", limit=5)
```

### 代理模式
代理程序会收到用于明确记录决策的工具。

```python
from agno.learn import LearningMachine, LearningMode, DecisionLogConfig

agent = Agent(
    id="my-agent",
    model=OpenAIChat(id="gpt-4o"),
    db=db,
    learning=LearningMachine(
        decision_log=DecisionLogConfig(mode=LearningMode.AGENTIC),
    ),
)
```

可用工具：log_decision，，record_outcomesearch_decisions
代理决定何时将某个决定记录下来。

### 始终模式
工具调用会自动记录为决策。

```python
from agno.learn import LearningMachine, LearningMode, DecisionLogConfig

agent = Agent(
    id="my-agent",
    model=OpenAIChat(id="gpt-4o"),
    db=db,
    learning=LearningMachine(
        decision_log=DecisionLogConfig(mode=LearningMode.ALWAYS),
    ),
    tools=[DuckDuckGoTools()],
)

agent.print_response("What are the latest developments in AI agents?")
# Tool calls are automatically recorded as decisions
```
缺点：会记录每次工具调用，可能会产生噪声。
​
### 数据模型

| 场地 | 描述 |
| :--- | :--- |
| id | 唯一标识符（例如，＂dec＿abc123＂） |
| decision | 最终决定了 |
| reasoning | 做出这项决定的原因 |
| decision＿type | 类别：工具选择、响应样式、澄清 |
| context | 需要做出决定的情况 |
| alternatives | 考虑过的其他方案 |
| confidence | 置信度（0．0 到 1．0） |
| outcome | 结果如何呢？ |
| outcome＿quality | 是好、坏还是中性？ |
| created＿at | 做出决定时 |


### 记录结果
根据实际情况更新决策，以建立反馈机制：

```python
lm = agent.get_learning_machine()

# Via store directly
lm.decision_log_store.update_outcome(
    decision_id="dec_abc123",
    outcome="User was satisfied with Python recommendation",
    outcome_quality="good",
)
```

或者，客服人员可以record_outcome在对话过程中使用该工具。

### 获取决策

```python
lm = agent.get_learning_machine()

# Search decisions
decisions = lm.decision_log_store.search(
    agent_id="my-agent",
    decision_type="tool_selection",
    days=7,
    limit=10,
)

for d in decisions:
    print(f"{d.decision}: {d.reasoning}")

# Debug output
lm.decision_log_store.print(agent_id="my-agent", limit=5)
```

### 上下文注入
最近的决策会输入到系统提示中：

```xml
<decision_log>
Recent decisions:

- **Recommended Python over JavaScript**
  Reasoning: Web scraping libraries are more mature in Python
  Outcome: User was satisfied

- **Used web search for current info**
  Reasoning: Question about recent developments requires fresh data
</decision_log>
```

### 决策类型
决策组织常用分类：

| 类型 | 何时使用 |
| :--- | :--- |
| tool＿selection | 选择要调用的工具 |
| response＿style | 决定如何格式化或措辞回复 |
| clarification | 选择询问更多信息 |
| escalation | 决定听从人类的意见 |
| approach | 在解决方案策略之间进行选择 |

### 用例
- 审计：审查代理人做出的决策及其原因。
- 调试：通过检查原因来理解意外行为
- 学习：分析结果模式以改进代理指令
- 反馈循环：记录结果以识别成功模式

## 学习模式

控制智能体何时以及如何学习。

学习模式控制着学习机器何时以及如何捕获信息。每个商店可以使用不同的模式。

| 模式 | 工作原理 | 权衡 |
| :--- | :--- | :--- |
| 总是 | 提取过程在每次响应后自动运行 | 每次互动额外调用LLM |
| 代理 | 代理人收到工具后决定保存哪些内容 | 可能遗漏隐含信息 |
| 提出 | 代理提出学习方案，用户确认后保存。 | 需要用户交互 |

### 始终模式
提取过程在后台自动进行，无需任何代理工具。

```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.learn import LearningMachine, LearningMode, UserProfileConfig
from agno.models.openai import OpenAIResponses

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=PostgresDb(db_url="postgresql+psycopg://ai:ai@localhost:5532/ai"),
    learning=LearningMachine(
        user_profile=UserProfileConfig(mode=LearningMode.ALWAYS),
    ),
)

# Profile info extracted automatically - no tool calls visible
agent.print_response(
    "I'm Alice Chen, but please call me Ali.",
    user_id="alice@example.com",
)
```
最适用于：用户配置文件、用户内存、会话上下文、实体内存

### 代理模式
代理人接收工具并决定何时保存。

```python
from agno.learn import LearningMachine, LearningMode, UserProfileConfig

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=LearningMachine(
        user_profile=UserProfileConfig(mode=LearningMode.AGENTIC),
    ),
)

# Agent decides to call update_profile tool
agent.print_response(
    "Remember that I prefer dark mode interfaces.",
    user_id="alice@example.com",
)
```

最适合：学习知识、决策日志

### 按商店分类的工具

| 店铺 | 工具 |
| :--- | :--- |
| 用户个人资料 | update＿profile |
| 用户内存 | update＿user＿memory |
| 实体记忆 | search＿entities ，，，，，，，，create＿entity update＿entity add＿fact update＿fact delete＿fact add＿event add＿relationship |
| 已习得的知识 | search＿learnings，save＿learning |
| 决策日志 | log＿decision ，，record＿outcome search＿decisions |

### 提议模式
代理会提出学习建议。用户必须确认后才能保存。
```python
from agno.learn import LearningMachine, LearningMode, LearnedKnowledgeConfig

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=LearningMachine(
        knowledge=knowledge,
        learned_knowledge=LearnedKnowledgeConfig(mode=LearningMode.PROPOSE),
    ),
)

# Agent proposes, user confirms
agent.print_response(
    "That's a great insight about API rate limits - we should remember that.",
    user_id="alice@example.com",
)
```

注意：建议模式目前仅适用于已学习的知识。
最适合：高风险知识、受监管环境、质量控制

### 组合模式
对不同店铺使用不同的模式：

```python
from agno.learn import (
    LearningMachine,
    LearningMode,
    UserProfileConfig,
    UserMemoryConfig,
    LearnedKnowledgeConfig,
)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=LearningMachine(
        user_profile=UserProfileConfig(mode=LearningMode.ALWAYS),     # Automatic
        user_memory=UserMemoryConfig(mode=LearningMode.ALWAYS),       # Automatic
        learned_knowledge=LearnedKnowledgeConfig(mode=LearningMode.AGENTIC),  # Agent-driven
    ),
)
```

### 商店默认值

| 店铺 | 默认模式 | 原因 |
| :--- | :--- | :--- |
| 用户个人资料 | 总是 | 姓名和偏好应以一致的方式记录。 |
| 用户内存 | 总是 | 观测数据被动地积累起来 |
| 会话上下文 | 总是 | 会话状态需要持续跟踪 |
| 实体记忆 | 总是 | 持续提取从日常对话中捕获实体事实／事件 |
| 已习得的知识 | 代理 | 代理决定哪些见解值得保存 |
| 决策日志 | 总是（DecisionLogConfig（）），主动的 （ decision＿log＝True ） | 支持自动日志记录和显式日志记录工作流程 |


### 选择模式
| 设想 | 模式 |
| :--- | :--- |
| 记录用户名和偏好 | 总是 |
| 自动构建用户内存 | 总是 |
| 跟踪会话进度 | 总是 |
| 智能体驱动的知识采集 | 代理 |
| 构建实体知识图谱 | 总是 |
| 审计代理人的决定 | 代理 |
| 高价值集体知识 | 提出 |
| 合规敏感型学习 | 提出 |

## 自定义模式


为您的域名扩展商店的自定义字段。

学习型学习库默认使用预定义的模式。您可以添加自定义字段来扩展这些模式，从而捕获特定领域的信息。
​
### 扩展用户个人资料
默认UserProfile包含name和preferred_name。请为您的域名添加字段：

```python
from dataclasses import dataclass, field
from typing import Optional
from agno.learn.schemas import UserProfile

@dataclass
class CustomerProfile(UserProfile):
    company: Optional[str] = field(
        default=None,
        metadata={"description": "Company or organization"}
    )
    plan_tier: Optional[str] = field(
        default=None,
        metadata={"description": "Subscription tier: free | pro | enterprise"}
    )
    role: Optional[str] = field(
        default=None,
        metadata={"description": "Job title or role"}
    )
    timezone: Optional[str] = field(
        default=None,
        metadata={"description": "User's timezone"}
    )
```

### 使用自定义架构：

```python
from agno.learn import LearningMachine, UserProfileConfig

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    learning=LearningMachine(
        user_profile=UserProfileConfig(schema=CustomerProfile),
    ),
)
```

### 实地指南
​
#### 使用元数据描述
这metadata={"description": ...}告诉LLM要提取什么：
```python
# Good: Clear description guides extraction
role: Optional[str] = field(
    default=None,
    metadata={"description": "Job title like 'Data Scientist' or 'Engineering Manager'"}
)

# Less effective: No description
role: Optional[str] = None
```

#### 使用可选字段
所有自定义字段都应该Optional有默认值：

```python
# Good
company: Optional[str] = field(default=None, metadata={...})

# Bad: Required field will fail if not extracted
company: str
```

#### 文档约束值
对于已知选项的字段，请在描述中列出这些选项：

```python
plan_tier: Optional[str] = field(
    default=None,
    metadata={"description": "Subscription tier: free | pro | enterprise"}
)
```

### 领域示例
​
#### SaaS 支持
```python
@dataclass
class SupportProfile(UserProfile):
    company: Optional[str] = field(
        default=None,
        metadata={"description": "Company name"}
    )
    plan: Optional[str] = field(
        default=None,
        metadata={"description": "Plan: starter | professional | enterprise"}
    )
    account_id: Optional[str] = field(
        default=None,
        metadata={"description": "Account or customer ID"}
    )
    primary_use_case: Optional[str] = field(
        default=None,
        metadata={"description": "Main use case or workflow"}
    )
```

#### 开发者工具

```python
@dataclass
class DeveloperProfile(UserProfile):
    primary_language: Optional[str] = field(
        default=None,
        metadata={"description": "Primary language: python | javascript | go | rust"}
    )
    framework: Optional[str] = field(
        default=None,
        metadata={"description": "Primary framework: react | django | fastapi"}
    )
    experience_years: Optional[int] = field(
        default=None,
        metadata={"description": "Years of programming experience"}
    )
    editor: Optional[str] = field(
        default=None,
        metadata={"description": "Editor: vscode | neovim | intellij"}
    )
```

### 扩展其他模式
​
#### 实体记忆

```python
from agno.learn.schemas import EntityMemory

@dataclass
class CompanyEntity(EntityMemory):
    industry: Optional[str] = field(
        default=None,
        metadata={"description": "Industry: fintech | healthcare | saas"}
    )
    funding_stage: Optional[str] = field(
        default=None,
        metadata={"description": "Stage: seed | series_a | series_b | public"}
    )
    employee_count: Optional[int] = field(
        default=None,
        metadata={"description": "Number of employees"}
    )
```

#### 已习得的知识

```python
from agno.learn.schemas import LearnedKnowledge

@dataclass
class TechnicalInsight(LearnedKnowledge):
    applicable_languages: Optional[List[str]] = field(
        default=None,
        metadata={"description": "Languages this applies to"}
    )
    performance_impact: Optional[str] = field(
        default=None,
        metadata={"description": "Performance impact: high | medium | low"}
    )
    complexity: Optional[str] = field(
        default=None,
        metadata={"description": "Complexity: simple | moderate | complex"}
    )
```

### 完整示例

```python
from dataclasses import dataclass, field
from typing import Optional
from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.learn import LearningMachine, UserProfileConfig
from agno.learn.schemas import UserProfile
from agno.models.openai import OpenAIResponses

@dataclass
class EnterpriseProfile(UserProfile):
    company: Optional[str] = field(
        default=None,
        metadata={"description": "Company name"}
    )
    department: Optional[str] = field(
        default=None,
        metadata={"description": "Department: engineering | sales | marketing"}
    )
    role: Optional[str] = field(
        default=None,
        metadata={"description": "Job title"}
    )
    region: Optional[str] = field(
        default=None,
        metadata={"description": "Region: NA | EMEA | APAC"}
    )

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=PostgresDb(db_url="postgresql+psycopg://ai:ai@localhost:5532/ai"),
    learning=LearningMachine(
        user_profile=UserProfileConfig(schema=EnterpriseProfile),
    ),
)

# Custom fields extracted automatically
agent.print_response(
    "Hi, I'm Sarah Chen, VP of Engineering at Acme Corp. We're the EMEA team.",
    user_id="sarah@acme.com",
)
```


# 记忆
## 什么是记忆？


赋予您的客服人员记住用户偏好、上下文和过往互动的能力，从而提供真正个性化的体验。

想象一下，一位客服人员能记住你上周的产品偏好，或者一位私人助理知道你喜欢早上开会，但前提是你已经喝过咖啡。这就是 Agno 记忆功能的强大之处。
​
## 内存的工作原理
当对话中出现相关信息，例如用户的姓名、偏好或习惯时，具备记忆功能的智能体会自动将其存储在数据库中。之后，当这些信息再次变得相关时，智能体会自动检索并将其自然地运用到对话中。这样，智能体就能在与用户的互动中有效地学习每个用户的信息。

## 内存入门
设置内存非常简单：只需连接数据库并启用内存功能即可。以下是一个基本设置示例：

```python

from agno.agent import Agent
from agno.db.sqlite import SqliteDb

# Setup your database
db = SqliteDb(db_file="agno.db")

# Setup your Agent with Memory
agent = Agent(
    db=db,
    update_memory_on_run=True, # This enables Memory for the Agent
)
```

有了它`update_memory_on_run=True`，您的智能助手会在每次对话后自动创建和更新记忆。它会提取相关信息并存储起来，并在需要时调用，无需人工干预。

最适合：客户支持、个人助理、需要保持一致记忆行为的对话应用程序。

### 代理记忆（enable_agentic_memory=True）
代理通过内置工具完全控制内存管理。它会根据对话上下文决定何时创建、更新或删除内存。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb

# Setup your database
db = SqliteDb(db_file="agno.db")

# Setup your Agent with Agentic Memory
agent = Agent(
    db=db,
    enable_agentic_memory=True, # This enables Agentic Memory for the Agent
)
```

借助智能体记忆，智能体配备了在认为必要时管理记忆的工具。这赋予了智能体更大的灵活性，但也要求智能体能够智能地决定记住哪些信息。

最适合：复杂的工作流程、多轮交互，其中代理需要根据上下文决定哪些内容值得记住。

> 重要提示：请勿同时启用这两个选项update_memory_on_run，enable_agentic_memory因为它们互斥。虽然同时启用这两个选项不会造成任何问题，但enable_agentic_memory始终会优先启用其中一个，并且另一个update_memory_on_run会被忽略。


### 存储：记忆的居所
内存存储在您连接到代理的数据库中。Agno 支持所有主流数据库系统：Postgres、SQLite、MongoDB 等。请查看存储文档以获取支持的数据库完整列表和设置说明。

默认情况下，记忆存储在agno_memories表（或文档数据库中的集合）中。如果代理首次尝试存储记忆时该表不存在，Agno 会自动创建它，无需手动设置架构。

**自定义表名**

您可以为存储内存指定自定义表名：

```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb

# Setup your database
db = PostgresDb(
    db_url="postgresql://user:password@localhost:5432/my_database",
    memory_table="my_memory_table", # Specify the table to store memories
)

# Setup your Agent with the database
agent = Agent(db=db, update_memory_on_run=True)

# Run the Agent. This will store a session in our "my_memory_table"
agent.print_response("Hi! My name is John Doe and I like to play basketball on the weekends.")

agent.print_response("What are my hobbies?")

```

#### 手动记忆提取
虽然对话过程中会自动调用记忆，但您也可以使用该get_user_memories方法手动检索它们。这对于调试、显示用户个人资料或构建自定义记忆界面非常有用：

```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb

# Setup your database
db = PostgresDb(
    db_url="postgresql://user:password@localhost:5432/my_database",
    memory_table="my_memory_table", # Specify the table to store memories
)

# Setup your Agent with the database
agent = Agent(db=db)

# Run the Agent. This will store a memory in our "my_memory_table"
agent.print_response("I love sushi!", user_id="123")

# Retrieve the memories about the user
memories = agent.get_user_memories(user_id="123")
print(memories)
```

### 内存数据模型
数据库中存储的每个内存条目都包含以下字段：

| 场地 | 类型 | 描述 |
| :--- | :--- | :--- |
| memory＿id | str | 内存的唯一标识符。 |
| memory | str | 内存内容，以字符串形式存储。 |
| topics | list | 记忆的主题。 |
| input | str | 生成该内存的输入。 |
| user＿id | str | 内存的用户ID。 |
| agent＿id | str | 内存的代理 ID。 |
| team＿id | str | 内存的团队 ID。 |
| updated＿at | int | 内存上次更新的时间戳。 |


### 与记忆打交道

自定义记忆的创建方式、控制上下文包含、在代理之间共享记忆，以及使用记忆工具进行高级工作流程。

基本的内存配置可以满足大多数使用场景，但有时您需要更精细的控制。本指南涵盖了自定义内存行为、控制存储内容以及构建具有共享内存的复杂多智能体系统的高级模式。
​
### 自定义内存管理器
LLM控制MemoryManager记忆的创建和更新，以及记忆的生成方式。您可以自定义设置以使用特定模型、添加隐私规则或更改记忆的提取方式：
```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.memory import MemoryManager
from agno.models.openai import OpenAIResponses

# Setup your database
db = SqliteDb(db_file="agno.db")

# Setup your Memory Manager, to adjust how memories are created
memory_manager = MemoryManager(
    db=db,
    # Select the model used for memory creation and updates. If unset, the default model of the Agent is used.
    model=OpenAIResponses(id="gpt-5.2"),
    # You can also provide additional instructions
    additional_instructions="Don't store the user's real name",
)

# Now provide the adjusted Memory Manager to your Agent
agent = Agent(
    db=db,
    memory_manager=memory_manager,
    update_memory_on_run=True,
)

agent.print_response("My name is John Doe and I like to play basketball on the weekends.")

agent.print_response("What's do I do in weekends?")

```

在这个例子中，内存管理器会存储用户的兴趣爱好信息，但不会包含用户的真实姓名。这对于医疗保健、法律或其他对隐私要求较高的应用场景非常有用。

### 记忆与背景

启用此功能后，系统会在每次请求时自动将当前用户的记忆添加到代理的上下文中。但在某些情况下，例如当您基于记忆构建分析功能或希望代理使用工具显式搜索记忆时，您可能希望存储记忆而不自动将其包含在上下文中。
用于`add_memories_to_context=False`在后台收集记忆，同时保持代理的上下文简洁：

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb

# Setup your database
db = SqliteDb(db_file="agno.db")

# Setup your Agent with Memory
agent = Agent(
    db=db,
    update_memory_on_run=True, # This enables Memory for the Agent
    add_memories_to_context=False, # This disables adding memories to the context
)

```


### 内存优化
随着用户不断积累记忆，并且每次请求都会将这些记忆添加到上下文中，令牌成本可能会显著增加。内存优化通过将多个记忆合并为更少、更高效的记忆来降低这些成本，同时保留所有关键信息。

何时进行优化：
- 拥有 50 个以上记忆的用户
- 在高成本操作之前
- 长期运行应用的定期维护

```python

from agno.memory.strategies.types import MemoryOptimizationStrategyType

# Optimize memories for a user
optimized = agent.memory_manager.optimize_memories(
    user_id="user_123",
    strategy=MemoryOptimizationStrategyType.SUMMARIZE,
    apply=True,  # Set to False to preview without saving
)
```
请参阅内存优化指南，了解详细使用方法和最佳实践。


### 使用内存工具
与其使用自动内存管理，不如为智能体提供显式的工具来创建、检索、更新和删除内存。这种方法赋予智能体更大的控制权和推理能力，使其能够决定何时存储信息，何时搜索现有信息。

何时使用内存工具：
- 你希望代理人能够判断某件事是否值得记住。
- 你需要对内存操作进行细粒度控制（分别进行创建、更新和删除操作）。
- 你正在构建一个系统，在这个系统中，智能体需要显式地搜索内存，而不是让内存自动加载。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.tools.memory import MemoryTools

# Create a database connection
db = SqliteDb(
    db_file="tmp/memory.db"
)

memory_tools = MemoryTools(
    db=db,
)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[memory_tools],
    markdown=True,
)

if __name__ == "__main__":
    agent.print_response(
        "My name is John Doe and I like to hike in the mountains on weekends. "
        "I like to travel to new places and experience different cultures. "
        "I am planning to travel to Africa in December. ",
        user_id="john_doe@example.com",
        stream=True
    )

    # This won't use the session history, but instead will use the memory tools to get the memories
    agent.print_response("What have you remembered about me?", stream=True, user_id="john_doe@example.com")

```

请参阅内存工具文档了解更多详情。


### 代理之间共享内存
在多代理系统中，您通常希望代理之间共享用户知识。例如，客服代理可能需要了解用户的偏好，销售代理也应该知道这些信息。在 Agno 中，这很容易实现：只需将多个代理连接到同一个数据库即可。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb

# Setup your database
db = SqliteDb(db_file="agno.db")

# Setup your Agents with the same database and Memory enabled
agent_1 = Agent(db=db, update_memory_on_run=True)
agent_2 = Agent(db=db, update_memory_on_run=True)

# The first Agent will create a Memory about the user name here:
agent_1.print_response("Hi! My name is John Doe")

# The second Agent will be able to retrieve the Memory about the user name here:
agent_2.print_response("What is my name?")

```

所有连接到同一数据库的代理会自动为每个用户共享记忆。只要使用相同的user_id，这在不同代理类型、团队和工作流程中都适用。


## 内存优化
​
### 代码

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.memory import MemoryManager, SummarizeStrategy
from agno.memory.strategies.types import MemoryOptimizationStrategyType
from agno.models.openai import OpenAIResponses

db_file = "tmp/memory_summarize_strategy.db"
db = SqliteDb(db_file=db_file)

user_id = "user2"

# Create agent with memory enabled
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    update_memory_on_run=True,
)

# Create some memories for a user
print("Creating memories...")
agent.print_response(
    "I have a wonderful pet dog named Max who is 3 years old. He's a golden retriever and he's such a friendly and energetic dog. "
    "We got him as a puppy when he was just 8 weeks old. He loves playing fetch in the park and going on long walks. "
    "Max is really smart too - he knows about 15 different commands and tricks. Taking care of him has been one of the most "
    "rewarding experiences of my life. He's basically part of the family now.",
    user_id=user_id,
)
agent.print_response(
    "I currently live in San Francisco, which is an amazing city despite all its challenges. I've been here for about 5 years now. "
    "I work in the tech industry as a product manager at a mid-sized software company. The tech scene here is incredible - "
    "there are so many smart people working on interesting problems. The cost of living is definitely high, but the opportunities "
    "and the community make it worthwhile. I live in the Mission district which has great food and a vibrant culture.",
    user_id=user_id,
)
agent.print_response(
    "On weekends, I really enjoy hiking in the beautiful areas around the Bay Area. There are so many amazing trails - "
    "from Mount Tamalpais to Big Basin Redwoods. I usually go hiking with a group of friends and we try to explore new trails every month. "
    "I also love trying new restaurants. San Francisco has such an incredible food scene with cuisines from all over the world. "
    "I'm always on the lookout for hidden gems and new places to try. My favorite types of cuisine are Japanese, Thai, and Mexican.",
    user_id=user_id,
)
agent.print_response(
    "I've been learning to play the piano for about a year and a half now. It's something I always wanted to do but never had time for. "
    "I finally decided to commit to it and I practice almost every day, usually for 30-45 minutes. "
    "I'm working through classical pieces right now - I can play some simple Bach and Mozart compositions. "
    "My goal is to eventually be able to play some jazz piano as well. Having a creative hobby like this has been great for my mental health "
    "and it's nice to have something completely different from my day job.",
    user_id=user_id,
)

# Check current memories
print("\nBefore optimization:")
memories_before = agent.get_user_memories(user_id=user_id)
print(f"  Memory count: {len(memories_before)}")

# Count tokens before optimization
strategy = SummarizeStrategy()
tokens_before = strategy.count_tokens(memories_before)
print(f"  Token count: {tokens_before} tokens")

print("\nIndividual memories:")
for i, memory in enumerate(memories_before, 1):
    print(f"  {i}. {memory.memory}")

# Create memory manager and optimize memories
memory_manager = MemoryManager(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
)

print("\nOptimizing memories with 'summarize' strategy...")
memory_manager.optimize_memories(
    user_id=user_id,
    strategy=MemoryOptimizationStrategyType.SUMMARIZE,  # Combine all memories into one
    apply=True,  # Apply changes to database
)

# Check optimized memories
print("\nAfter optimization:")
memories_after = agent.get_user_memories(user_id=user_id)
print(f"  Memory count: {len(memories_after)}")

# Count tokens after optimization
tokens_after = strategy.count_tokens(memories_after)
print(f"  Token count: {tokens_after} tokens")

# Calculate reduction
if tokens_before > 0:
    reduction_pct = ((tokens_before - tokens_after) / tokens_before) * 100
    tokens_saved = tokens_before - tokens_after
    print(f"  Reduction: {reduction_pct:.1f}% ({tokens_saved} tokens saved)")

if memories_after:
    print("\nSummarized memory:")
    print(f"  {memories_after[0].memory}")
else:
    print("\n No memories found after optimization")

```
## 存储使用情况
### MongoDB 内存

```python
from agno.agent import Agent
from agno.db.mongo import MongoDb

# Setup MongoDb
db_url = "mongodb://localhost:27017"

db = MongoDb(db_url=db_url)

agent = Agent(
    db=db,
    update_memory_on_run=True,
)

agent.print_response("My name is John Doe and I like to play basketball on the weekends.")
agent.print_response("What's do I do in weekends?")
```

### PostgreSQL 的内存管理
```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb

# Setup Postgres
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
db = PostgresDb(db_url=db_url)

agent = Agent(
    db=db,
    update_memory_on_run=True,
)

agent.print_response("My name is John Doe and I like to play basketball on the weekends.")
agent.print_response("What's do I do in weekends?")

```

### Memory with SQLite

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb

# Setup the SQLite database
db = SqliteDb(db_file="tmp/data.db")

# Setup a basic agent with the SQLite database
agent = Agent(
    db=db,
    update_memory_on_run=True,
)

agent.print_response("My name is John Doe and I like to play basketball on the weekends.")
agent.print_response("What's do I do in weekends?")
```

### Redis 内存

```python

from agno.agent import Agent
from agno.db.redis import RedisDb

# Setup Redis
# Initialize Redis db (use the right db_url for your setup)
db = RedisDb(db_url="redis://localhost:6379")

# Create agent with Redis db
agent = Agent(
    db=db,
    update_memory_on_run=True,
)

agent.print_response("My name is John Doe and I like to play basketball on the weekends.")
agent.print_response("What's do I do in weekends?")
```


### 内存管理器使用情况
#### 独立内存
```python
from agno.db.postgres import PostgresDb
from agno.memory import MemoryManager, UserMemory
from rich.pretty import pprint

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

memory = MemoryManager(db=PostgresDb(db_url=db_url))

# Add a memory for the default user
memory.add_user_memory(
    memory=UserMemory(memory="The user's name is John Doe", topics=["name"]),
)
print("Memories:")
pprint(memory.get_user_memories())

# Add memories for Jane Doe
jane_doe_id = "jane_doe@example.com"
print(f"\nUser: {jane_doe_id}")
memory_id_1 = memory.add_user_memory(
    memory=UserMemory(memory="The user's name is Jane Doe", topics=["name"]),
    user_id=jane_doe_id,
)
memory_id_2 = memory.add_user_memory(
    memory=UserMemory(memory="She likes to play tennis", topics=["hobbies"]),
    user_id=jane_doe_id,
)
memories = memory.get_user_memories(user_id=jane_doe_id)
print("Memories:")
pprint(memories)

# Delete a memory
print("\nDeleting memory")
assert memory_id_2 is not None
memory.delete_user_memory(user_id=jane_doe_id, memory_id=memory_id_2)
print("Memory deleted\n")
memories = memory.get_user_memories(user_id=jane_doe_id)
print("Memories:")
pprint(memories)

# Replace a memory
print("\nReplacing memory")
assert memory_id_1 is not None
memory.replace_user_memory(
    memory_id=memory_id_1,
    memory=UserMemory(memory="The user's name is Jane Mary Doe", topics=["name"]),
    user_id=jane_doe_id,
)
print("Memory replaced")
memories = memory.get_user_memories(user_id=jane_doe_id)
print("Memories:")
pprint(memories)
```

### 记忆创造

通过向代理提供文本或消息列表来创建用户记忆。
​
```python

from agno.db.postgres import PostgresDb
from agno.memory import MemoryManager, UserMemory
from agno.models.message import Message
from agno.models.openai import OpenAIResponses
from rich.pretty import pprint

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

memory_db = PostgresDb(db_url=db_url)

memory = MemoryManager(model=OpenAIResponses(id="gpt-5.2"), db=memory_db)

john_doe_id = "john_doe@example.com"
memory.add_user_memory(
    memory=UserMemory(
        memory="""
I enjoy hiking in the mountains on weekends,
reading science fiction novels before bed,
cooking new recipes from different cultures,
playing chess with friends,
and attending live music concerts whenever possible.
Photography has become a recent passion of mine, especially capturing landscapes and street scenes.
I also like to meditate in the mornings and practice yoga to stay centered.
"""
    ),
    user_id=john_doe_id,
)


memories = memory.get_user_memories(user_id=john_doe_id)
print("John Doe's memories:")
pprint(memories)

jane_doe_id = "jane_doe@example.com"
# Send a history of messages and add memories
memory.create_user_memories(
    messages=[
        Message(role="user", content="My name is Jane Doe"),
        Message(role="assistant", content="That is great!"),
        Message(role="user", content="I like to play chess"),
        Message(role="assistant", content="That is great!"),
    ],
    user_id=jane_doe_id,
)

memories = memory.get_user_memories(user_id=jane_doe_id)
print("Jane Doe's memories:")
pprint(memories)
```


### 自定义内存指令

通过向代理提供文本或消息列表来创建用户记忆

```python
from agno.db.postgres import PostgresDb
from agno.memory import MemoryManager
from agno.models.anthropic.claude import Claude
from agno.models.message import Message
from agno.models.openai import OpenAIResponses
from rich.pretty import pprint

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

memory_db = PostgresDb(db_url=db_url)

memory = MemoryManager(
    model=OpenAIResponses(id="gpt-5.2"),
    memory_capture_instructions="""\
                    Memories should only include details about the user's academic interests.
                    Only include which subjects they are interested in.
                    Ignore names, hobbies, and personal interests.
                    """,
    db=memory_db,
)

john_doe_id = "john_doe@example.com"

memory.create_user_memories(
    input="""\
My name is John Doe.

I enjoy hiking in the mountains on weekends,
reading science fiction novels before bed,
cooking new recipes from different cultures,
playing chess with friends.

I am interested to learn about the history of the universe and other astronomical topics.
""",
    user_id=john_doe_id,
)


memories = memory.get_user_memories(user_id=john_doe_id)
print("John Doe's memories:")
pprint(memories)


# Use default memory manager
memory = MemoryManager(model=Claude(id="claude-3-5-sonnet-latest"), db=memory_db)
jane_doe_id = "jane_doe@example.com"

# Send a history of messages and add memories
memory.create_user_memories(
    messages=[
        Message(role="user", content="Hi, how are you?"),
        Message(role="assistant", content="I'm good, thank you!"),
        Message(role="user", content="What are you capable of?"),
        Message(
            role="assistant",
            content="I can help you with your homework and answer questions about the universe.",
        ),
        Message(role="user", content="My name is Jane Doe"),
        Message(role="user", content="I like to play chess"),
        Message(
            role="user",
            content="Actually, forget that I like to play chess. I more enjoy playing table top games like dungeons and dragons",
        ),
        Message(
            role="user",
            content="I'm also interested in learning about the history of the universe and other astronomical topics.",
        ),
        Message(role="assistant", content="That is great!"),
        Message(
            role="user",
            content="I am really interested in physics. Tell me about quantum mechanics?",
        ),
    ],
    user_id=jane_doe_id,
)

memories = memory.get_user_memories(user_id=jane_doe_id)
print("Jane Doe's memories:")
pprint(memories)

```


### 内存搜索
如何使用不同的检索方法搜索用户记忆。
- last_n：检索最后 n 个记忆
- first_n：检索前 n 个记忆
- agentic使用智能搜索检索记忆

```python
from agno.db.postgres import PostgresDb
from agno.memory import MemoryManager, UserMemory
from agno.models.openai import OpenAIResponses
from rich.pretty import pprint

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

memory_db = PostgresDb(db_url=db_url)

memory = MemoryManager(model=OpenAIResponses(id="gpt-5.2"), db=memory_db)

john_doe_id = "john_doe@example.com"
memory.add_user_memory(
    memory=UserMemory(memory="The user enjoys hiking in the mountains on weekends"),
    user_id=john_doe_id,
)
memory.add_user_memory(
    memory=UserMemory(
        memory="The user enjoys reading science fiction novels before bed"
    ),
    user_id=john_doe_id,
)
print("John Doe's memories:")
pprint(memory.get_user_memories(user_id=john_doe_id))

memories = memory.search_user_memories(
    user_id=john_doe_id, limit=1, retrieval_method="last_n"
)
print("\nJohn Doe's last_n memories:")
pprint(memories)

memories = memory.search_user_memories(
    user_id=john_doe_id, limit=1, retrieval_method="first_n"
)
print("\nJohn Doe's first_n memories:")
pprint(memories)

memories = memory.search_user_memories(
    user_id=john_doe_id,
    query="What does the user like to do on weekends?",
    retrieval_method="agentic",
)
print("\nJohn Doe's memories similar to the query (agentic):")
pprint(memories)

```


### 内存优化

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.memory import MemoryManager, SummarizeStrategy
from agno.memory.strategies.types import MemoryOptimizationStrategyType
from agno.models.openai import OpenAIResponses

db_file = "tmp/memory_summarize_strategy.db"
db = SqliteDb(db_file=db_file)

user_id = "user2"

# Create agent with memory enabled
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    update_memory_on_run=True,
)

# Create some memories for a user
print("Creating memories...")
agent.print_response(
    "I have a wonderful pet dog named Max who is 3 years old. He's a golden retriever and he's such a friendly and energetic dog. "
    "We got him as a puppy when he was just 8 weeks old. He loves playing fetch in the park and going on long walks. "
    "Max is really smart too - he knows about 15 different commands and tricks. Taking care of him has been one of the most "
    "rewarding experiences of my life. He's basically part of the family now.",
    user_id=user_id,
)
agent.print_response(
    "I currently live in San Francisco, which is an amazing city despite all its challenges. I've been here for about 5 years now. "
    "I work in the tech industry as a product manager at a mid-sized software company. The tech scene here is incredible - "
    "there are so many smart people working on interesting problems. The cost of living is definitely high, but the opportunities "
    "and the community make it worthwhile. I live in the Mission district which has great food and a vibrant culture.",
    user_id=user_id,
)
agent.print_response(
    "On weekends, I really enjoy hiking in the beautiful areas around the Bay Area. There are so many amazing trails - "
    "from Mount Tamalpais to Big Basin Redwoods. I usually go hiking with a group of friends and we try to explore new trails every month. "
    "I also love trying new restaurants. San Francisco has such an incredible food scene with cuisines from all over the world. "
    "I'm always on the lookout for hidden gems and new places to try. My favorite types of cuisine are Japanese, Thai, and Mexican.",
    user_id=user_id,
)
agent.print_response(
    "I've been learning to play the piano for about a year and a half now. It's something I always wanted to do but never had time for. "
    "I finally decided to commit to it and I practice almost every day, usually for 30-45 minutes. "
    "I'm working through classical pieces right now - I can play some simple Bach and Mozart compositions. "
    "My goal is to eventually be able to play some jazz piano as well. Having a creative hobby like this has been great for my mental health "
    "and it's nice to have something completely different from my day job.",
    user_id=user_id,
)

# Check current memories
print("\nBefore optimization:")
memories_before = agent.get_user_memories(user_id=user_id)
print(f"  Memory count: {len(memories_before)}")

# Count tokens before optimization
strategy = SummarizeStrategy()
tokens_before = strategy.count_tokens(memories_before)
print(f"  Token count: {tokens_before} tokens")

print("\nIndividual memories:")
for i, memory in enumerate(memories_before, 1):
    print(f"  {i}. {memory.memory}")

# Create memory manager and optimize memories
memory_manager = MemoryManager(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
)

print("\nOptimizing memories with 'summarize' strategy...")
memory_manager.optimize_memories(
    user_id=user_id,
    strategy=MemoryOptimizationStrategyType.SUMMARIZE,  # Combine all memories into one
    apply=True,  # Apply changes to database
)

# Check optimized memories
print("\nAfter optimization:")
memories_after = agent.get_user_memories(user_id=user_id)
print(f"  Memory count: {len(memories_after)}")

# Count tokens after optimization
tokens_after = strategy.count_tokens(memories_after)
print(f"  Token count: {tokens_after} tokens")

# Calculate reduction
if tokens_before > 0:
    reduction_pct = ((tokens_before - tokens_after) / tokens_before) * 100
    tokens_saved = tokens_before - tokens_after
    print(f"  Reduction: {reduction_pct:.1f}% ({tokens_saved} tokens saved)")

if memories_after:
    print("\nSummarized memory:")
    print(f"  {memories_after[0].memory}")
else:
    print("\n No memories found after optimization")
```

## 具有记忆能力的代理
### 代理记忆

记忆功能使代理能够回忆起有关用户的信息。

记忆是智能体上下文的一部分，它可以帮助智能体提供最佳、最个性化的响应。

### 用户内存
以下是一个在代理中使用内存的简单示例。

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.db.postgres import PostgresDb
from rich.pretty import pprint

user_id = "ava"

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

db = PostgresDb(
  db_url=db_url,
  memory_table="user_memories",  # Optionally specify a table name for the memories
)


# Initialize Agent
memory_agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    # Give the Agent the ability to update memories
    enable_agentic_memory=True,
    # OR - Run the MemoryManager automatically after each response
    update_memory_on_run=True,
    markdown=True,
)

db.clear_memories()

memory_agent.print_response(
    "My name is Ava and I like to ski.",
    user_id=user_id,
    stream=True,
)
print("Memories about Ava:")
pprint(memory_agent.get_user_memories(user_id=user_id))

memory_agent.print_response(
    "I live in san francisco, where should i move within a 4 hour drive?",
    user_id=user_id,
    stream=True,
)
print("Memories about Ava:")
pprint(memory_agent.get_user_memories(user_id=user_id))
```

### 具有记忆的代理

赋予代理跨会话的持久记忆。

本示例向您展示如何将持久内存与代理一起使用。

每次运行后，都会创建/更新用户内存。

要启用此功能，请`update_memory_on_run=True`在代理配置中进行设置。
​
```python
from uuid import uuid4

from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.models.openai import OpenAIResponses
from rich.pretty import pprint

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

db = PostgresDb(db_url=db_url)

db.clear_memories()

session_id = str(uuid4())
john_doe_id = "john_doe@example.com"

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    update_memory_on_run=True,
)

agent.print_response(
    "My name is John Doe and I like to hike in the mountains on weekends.",
    stream=True,
    user_id=john_doe_id,
    session_id=session_id,
)

agent.print_response(
    "What are my hobbies?", stream=True, user_id=john_doe_id, session_id=session_id
)

memories = agent.get_user_memories(user_id=john_doe_id)
print("John Doe's memories:")
pprint(memories)

agent.print_response(
    "Ok i dont like hiking anymore, i like to play soccer instead.",
    stream=True,
    user_id=john_doe_id,
    session_id=session_id,
)

# You can also get the user memories from the agent
memories = agent.get_user_memories(user_id=john_doe_id)
print("John Doe's memories:")
pprint(memories)
```


### 代理记忆
本示例向您展示如何将持久内存与代理一起使用。

每次运行期间，代理程序可以创建/更新/删除用户记忆。

要启用此功能，请enable_agentic_memory=True在代理配置中进行设置。

```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.models.openai import OpenAIResponses
from rich.pretty import pprint

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

db = PostgresDb(db_url=db_url)


john_doe_id = "john_doe@example.com"

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    enable_agentic_memory=True,
)

agent.print_response(
    "My name is John Doe and I like to hike in the mountains on weekends.",
    stream=True,
    user_id=john_doe_id,
)

agent.print_response("What are my hobbies?", stream=True, user_id=john_doe_id)

memories = agent.get_user_memories(user_id=john_doe_id)
print("Memories about John Doe:")
pprint(memories)


agent.print_response(
    "Remove all existing memories of me.",
    stream=True,
    user_id=john_doe_id,
)

memories = agent.get_user_memories(user_id=john_doe_id)
print("Memories about John Doe:")
pprint(memories)

agent.print_response(
    "My name is John Doe and I like to paint.", stream=True, user_id=john_doe_id
)

memories = agent.get_user_memories(user_id=john_doe_id)
print("Memories about John Doe:")
pprint(memories)


agent.print_response(
    "I don't paint anymore, i draw instead.", stream=True, user_id=john_doe_id
)

memories = agent.get_user_memories(user_id=john_doe_id)

print("Memories about John Doe:")
pprint(memories)
```

### 代理之间共享内存

此示例演示了如何在代理之间共享内存。

这意味着一个特工创建的记忆，其他特工也可以访问。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools
from rich.pretty import pprint

db = SqliteDb(db_file="agents.db")

john_doe_id = "john_doe@example.com"

chat_agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    description="You are a helpful assistant that can chat with users",
    db=db,
    update_memory_on_run=True,
)

chat_agent.print_response(
    "My name is John Doe and I like to hike in the mountains on weekends.",
    stream=True,
    user_id=john_doe_id,
)

chat_agent.print_response("What are my hobbies?", stream=True, user_id=john_doe_id)


research_agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    description="You are a research assistant that can help users with their research questions",
    tools=[HackerNewsTools()],
    db=db,
    update_memory_on_run=True,
)

research_agent.print_response(
    "I love reading about AI. What are the top stories on Hacker News about AI?",
    stream=True,
    user_id=john_doe_id,
)

memories = research_agent.get_user_memories(user_id=john_doe_id)
print("Memories about John Doe:")
pprint(memories)
```


### 自定义内存管理器

本示例展示了如何配置内存管理器。

我们还为内存管理器设置了自定义系统提示。您可以覆盖整个系统提示，也可以添加附加指令，这些指令会添加到系统提示的末尾。

```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.memory import MemoryManager
from agno.models.openai import OpenAIResponses
from rich.pretty import pprint

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

db = PostgresDb(db_url=db_url)

# You can also override the entire `system_message` for the memory manager
memory_manager = MemoryManager(
    model=OpenAIResponses(id="gpt-5.2"),
    additional_instructions="""
    IMPORTANT: Don't store any memories about the user's name. Just say "The User" instead of referencing the user's name.
    """,
    db=db,
)

john_doe_id = "john_doe@example.com"

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    memory_manager=memory_manager,
    update_memory_on_run=True,
    user_id=john_doe_id,
)

agent.print_response(
    "My name is John Doe and I like to swim and play soccer.", stream=True
)

agent.print_response("I dont like to swim", stream=True)


memories = agent.get_user_memories(user_id=john_doe_id)

print("John Doe's memories:")
pprint(memories)
```

### 多用户、多会话聊天

本示例演示如何运行多用户、多会话聊天。

```python
"""
In this example, we have 3 users and 4 sessions.

User 1 has 2 sessions.
User 2 has 1 session.
User 3 has 1 session.
"""

import asyncio

from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.models.openai import OpenAIResponses

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

db = PostgresDb(db_url=db_url)

user_1_id = "user_1@example.com"
user_2_id = "user_2@example.com"
user_3_id = "user_3@example.com"

user_1_session_1_id = "user_1_session_1"
user_1_session_2_id = "user_1_session_2"
user_2_session_1_id = "user_2_session_1"
user_3_session_1_id = "user_3_session_1"

chat_agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    update_memory_on_run=True,
)


async def run_chat_agent():
    await chat_agent.aprint_response(
        "My name is Mark Gonzales and I like anime and video games.",
        user_id=user_1_id,
        session_id=user_1_session_1_id,
    )
    await chat_agent.aprint_response(
        "I also enjoy reading manga and playing video games.",
        user_id=user_1_id,
        session_id=user_1_session_1_id,
    )

    # Chat with user 1 - Session 2
    await chat_agent.aprint_response(
        "I'm going to the movies tonight.",
        user_id=user_1_id,
        session_id=user_1_session_2_id,
    )

    # Chat with user 2
    await chat_agent.aprint_response(
        "Hi my name is John Doe.", user_id=user_2_id, session_id=user_2_session_1_id
    )
    await chat_agent.aprint_response(
        "I'm planning to hike this weekend.",
        user_id=user_2_id,
        session_id=user_2_session_1_id,
    )

    # Chat with user 3
    await chat_agent.aprint_response(
        "Hi my name is Jane Smith.", user_id=user_3_id, session_id=user_3_session_1_id
    )
    await chat_agent.aprint_response(
        "I'm going to the gym tomorrow.",
        user_id=user_3_id,
        session_id=user_3_session_1_id,
    )

    # Continue the conversation with user 1
    # The agent should take into account all memories of user 1.
    await chat_agent.aprint_response(
        "What do you suggest I do this weekend?",
        user_id=user_1_id,
        session_id=user_1_session_1_id,
    )


if __name__ == "__main__":
    # Chat with user 1 - Session 1
    asyncio.run(run_chat_agent())

    user_1_memories = chat_agent.get_user_memories(user_id=user_1_id)
    print("User 1's memories:")
    assert user_1_memories is not None
    for i, m in enumerate(user_1_memories):
        print(f"{i}: {m.memory}")

    user_2_memories = chat_agent.get_user_memories(user_id=user_2_id)
    print("User 2's memories:")
    assert user_2_memories is not None
    for i, m in enumerate(user_2_memories):
        print(f"{i}: {m.memory}")

    user_3_memories = chat_agent.get_user_memories(user_id=user_3_id)
    print("User 3's memories:")
    assert user_3_memories is not None
    for i, m in enumerate(user_3_memories):
        print(f"{i}: {m.memory}")
```

### 支持多用户、多会话同时聊天

本示例展示了如何同时运行多用户、多会话的聊天。


```python
"""
In this example, we have 3 users and 4 sessions.

User 1 has 2 sessions.
User 2 has 1 session.
User 3 has 1 session.
"""
import asyncio

from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.models.openai import OpenAIResponses

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

db = PostgresDb(db_url=db_url)

user_1_id = "user_1@example.com"
user_2_id = "user_2@example.com"
user_3_id = "user_3@example.com"

user_1_session_1_id = "user_1_session_1"
user_1_session_2_id = "user_1_session_2"
user_2_session_1_id = "user_2_session_1"
user_3_session_1_id = "user_3_session_1"

chat_agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    update_memory_on_run=True,
)


async def user_1_conversation():
    """Handle conversation with user 1 across multiple sessions"""
    # User 1 - Session 1
    await chat_agent.arun(
        "My name is Mark Gonzales and I like anime and video games.",
        user_id=user_1_id,
        session_id=user_1_session_1_id,
    )
    await chat_agent.arun(
        "I also enjoy reading manga and playing video games.",
        user_id=user_1_id,
        session_id=user_1_session_1_id,
    )

    # User 1 - Session 2
    await chat_agent.arun(
        "I'm going to the movies tonight.",
        user_id=user_1_id,
        session_id=user_1_session_2_id,
    )

    # Continue the conversation in session 1
    await chat_agent.arun(
        "What do you suggest I do this weekend?",
        user_id=user_1_id,
        session_id=user_1_session_1_id,
    )

    print("User 1 Done")


async def user_2_conversation():
    """Handle conversation with user 2"""
    await chat_agent.arun(
        "Hi my name is John Doe.", user_id=user_2_id, session_id=user_2_session_1_id
    )
    await chat_agent.arun(
        "I'm planning to hike this weekend.",
        user_id=user_2_id,
        session_id=user_2_session_1_id,
    )
    print("User 2 Done")


async def user_3_conversation():
    """Handle conversation with user 3"""
    await chat_agent.arun(
        "Hi my name is Jane Smith.", user_id=user_3_id, session_id=user_3_session_1_id
    )
    await chat_agent.arun(
        "I'm going to the gym tomorrow.",
        user_id=user_3_id,
        session_id=user_3_session_1_id,
    )
    print("User 3 Done")


async def run_concurrent_chat_agent():
    """Run all user conversations concurrently"""
    await asyncio.gather(
        user_1_conversation(), user_2_conversation(), user_3_conversation()
    )


if __name__ == "__main__":
    # Run all conversations concurrently
    asyncio.run(run_concurrent_chat_agent())

    user_1_memories = chat_agent.get_user_memories(user_id=user_1_id)
    print("User 1's memories:")
    assert user_1_memories is not None
    for i, m in enumerate(user_1_memories):
        print(f"{i}: {m.memory}")

    user_2_memories = chat_agent.get_user_memories(user_id=user_2_id)
    print("User 2's memories:")
    assert user_2_memories is not None
    for i, m in enumerate(user_2_memories):
        print(f"{i}: {m.memory}")

    user_3_memories = chat_agent.get_user_memories(user_id=user_3_id)
    print("User 3's memories:")
    assert user_3_memories is not None
    for i, m in enumerate(user_3_memories):
        print(f"{i}: {m.memory}")
```

### 特工之间共享记忆和历史

此示例展示了如何在代理之间共享内存和历史记录。

您可以设置add_history_to_context=True将历史记录添加到代理的上下文中。

您可以设置update_memory_on_run=True启用在每次运行结束时生成用户内存。

```python
from uuid import uuid4

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses

db = SqliteDb(db_file="tmp/agent_sessions.db")

session_id = str(uuid4())
user_id = "john_doe@example.com"

agent_1 = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="You are really friendly and helpful.",
    db=db,
    add_history_to_context=True,
    update_memory_on_run=True,
)

agent_2 = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="You are really grumpy and mean.",
    db=db,
    add_history_to_context=True,
    update_memory_on_run=True,
)

agent_1.print_response(
    "Hi! My name is John Doe.", session_id=session_id, user_id=user_id
)

agent_2.print_response("What is my name?", session_id=session_id, user_id=user_id)

agent_2.print_response(
    "I like to hike in the mountains on weekends.",
    session_id=session_id,
    user_id=user_id,
)

agent_1.print_response("What are my hobbies?", session_id=session_id, user_id=user_id)

agent_1.print_response(
    "What have we been discussing? Give me bullet points.",
    session_id=session_id,
    user_id=user_id,
)
```

## 拥有记忆力的团队
### 拥有记忆力的团队
在团队中使用持久内存。

该团队还可以像代理一样管理用户记忆：
```python
from agno.team import Team
from agno.db.sqlite import SqliteDb

db = SqliteDb(db_file="agno.db")

team_with_memory = Team(
    name="Team with Memory",
    members=[agent1, agent2],
    db=db,
    update_memory_on_run=True,
)

team_with_memory.print_response("Hi! My name is John Doe.")
team_with_memory.print_response("What is my name?")
```

要实现跨会话的持续学习，请使用 LearningMachine。


### 与内存管理器团队合作

本示例演示了如何在团队协作中使用持久内存。每次运行后，系统都会创建并更新用户记忆，使团队能够跨会话记住用户信息，并提供个性化体验。

```python
"""
This example shows you how to use persistent memory with an Agent.

After each run, user memories are created/updated.

To enable this, set `update_memory_on_run=True` in the Agent config.
"""

from uuid import uuid4

from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.memory import MemoryManager  # noqa: F401
from agno.models.openai import OpenAIResponses
from agno.team import Team

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
db = PostgresDb(db_url=db_url)

session_id = str(uuid4())
john_doe_id = "john_doe@example.com"

# 1. Create memories by setting `update_memory_on_run=True` in the Agent
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
)
team = Team(
    model=OpenAIResponses(id="gpt-5.2"),
    members=[agent],
    db=db,
    update_memory_on_run=True,
)

team.print_response(
    "My name is John Doe and I like to hike in the mountains on weekends.",
    stream=True,
    user_id=john_doe_id,
    session_id=session_id,
)

team.print_response(
    "What are my hobbies?", stream=True, user_id=john_doe_id, session_id=session_id
)

# 2. Set a custom MemoryManager on the agent
# memory_manager = MemoryManager(model=OpenAIResponses(id="gpt-5.2"))

# memory_manager.clear()

# agent = Agent(
#     model=OpenAIResponses(id="gpt-5.2"),
#     memory_manager=memory_manager,
# )

# team = Team(
#     model=OpenAIResponses(id="gpt-5.2"),
#     members=[agent],
#     db=db,
#     update_memory_on_run=True,
# )

# team.print_response(
#     "My name is John Doe and I like to hike in the mountains on weekends.",
#     stream=True,
#     user_id=john_doe_id,
#     session_id=session_id,
# )

# # You can also get the user memories from the agent
# memories = agent.get_user_memories(user_id=john_doe_id)
# print("John Doe's memories:")
# pprint(memories)
```

### 团队与 Agentic Memory
本示例演示了如何在团队中使用智能体记忆。与简单的记忆存储不同，智能体记忆允许人工智能根据对话上下文在每次运行期间主动创建、更新和删除用户记忆，从而实现智能记忆管理。

```python
"""
This example shows you how to use persistent memory with an Agent.

During each run the Agent can create/update/delete user memories.

To enable this, set `enable_agentic_memory=True` in the Agent config.
"""

from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.memory import MemoryManager  # noqa: F401
from agno.models.openai import OpenAIResponses
from agno.team import Team

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
db = PostgresDb(db_url=db_url)

john_doe_id = "john_doe@example.com"

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
)

team = Team(
    model=OpenAIResponses(id="gpt-5.2"),
    members=[agent],
    db=db,
    enable_agentic_memory=True,
)

team.print_response(
    "My name is John Doe and I like to hike in the mountains on weekends.",
    stream=True,
    user_id=john_doe_id,
)

team.print_response("What are my hobbies?", stream=True, user_id=john_doe_id)

# More examples:
# agent.print_response(
#     "Remove all existing memories of me.",
#     stream=True,
#     user_id=john_doe_id,
# )

# agent.print_response(
#     "My name is John Doe and I like to paint.", stream=True, user_id=john_doe_id
# )

# agent.print_response(
#     "I don't pain anymore, i draw instead.", stream=True, user_id=john_doe_id
# )
```

## 最佳实战

避免常见陷阱，优化成本，并确保生产环境中可靠的内存行为。

内存功能强大，但如果配置不当，可能会导致意外的令牌消耗、行为问题和高昂的成本。本指南将向您展示需要注意的事项以及如何优化生产环境中的内存使用。

### 快速参考
- 默认使用自动内存（update_memory_on_run=True），除非您有特殊原因需要代理控制。
- 务必提供 user_id，不要依赖默认的“default”用户。
- 使用代理内存时，应使用更便宜的内存操作模型。
- 对长时间运行的应用程序实施剪枝
- 监控生产环境中的令牌使用情况，以发现与内存相关的成本峰值。
- 使用真实数据进行测试：100+ 个记忆的表现与 5 个记忆的表现截然不同。

### 代理记忆令牌陷阱
问题：使用时enable_agentic_memory=True，每次内存操作都会触发一个单独的、嵌套的 LLM 调用。这种架构会导致令牌使用量激增，尤其是在内存不断累积的情况下。
以下是其内部工作原理：

- 用户发送消息 → 主 LLM 调用处理该消息
- 代理决定更新内存 → 调用update_user_memory工具
- 嵌套 LLM 调用触发：
    - 详细的系统提示（约50行）
    - 所有已加载到上下文中的现有用户记忆
    - 内存管理指令和工具
- 内存LLM执行工具调用（添加、更新、删除）
- 控制权返回主对话

实际影响：

```python
# Scenario: User with 100 existing memories
agent = Agent(
    db=db,
    enable_agentic_memory=True,
    model=OpenAIResponses(id="gpt-5.2")
)

# 10-message conversation where agent updates memory 7 times:
# Normal conversation: 10 × 500 tokens = 5,000 tokens
# With agentic memory: (10 × 500) + (7 × 5,000) = 40,000 tokens
# Cost increase: 8x more expensive!
```

随着内存占用增加，每次内存操作的开销也会越来越大。如果内存占用达到 200 个，那么仅加载上下文这一步，一次内存更新就可能消耗超过 10,000 个令牌。
​


### 缓解策略一：使用自动记忆
对于大多数使用场景来说，自动记忆是最佳选择——它的效率要高得多：
```python
# Recommended: Single memory processing after conversation
agent = Agent(
    db=db,
    update_memory_on_run=True  # Processes memories once at end
)

# Only use agentic memory when you specifically need:
# - Real-time memory updates during conversation
# - User-directed memory commands ("forget my address")
# - Complex memory reasoning within the conversation flow
```

### 缓解策略二：使用更便宜的内存操作模型
如果确实需要代理记忆，请使用成本较低的记忆管理模型，同时保留功能强大的对话模型

```python
from agno.memory import MemoryManager
from agno.models.openai import OpenAIResponses

# Cheap model for memory operations (60x less expensive)
memory_manager = MemoryManager(
    db=db,
    model=OpenAIResponses(id="gpt-5.2")
)

# Expensive model for main conversations
agent = Agent(
    db=db,
    model=OpenAIResponses(id="gpt-5.2"),
    memory_manager=memory_manager,
    enable_agentic_memory=True
)
```

这种方法可以在保持对话质量的同时，降低 98% 的内存相关成本。
​
### 缓解策略三：通过指令引导记忆行为
添加明确的指令以防止不必要的内存更新：

```python
agent = Agent(
    db=db,
    enable_agentic_memory=True,
    instructions=[
        "Only update memories when users share significant new information.",
        "Don't create memories for casual conversation or temporary states.",
        "Batch multiple memory updates together when possible."
    ]
)
```


### 缓解策略#4：实施内存剪枝
定期清理陈旧或无关的内存，防止内存膨胀：

```python
from datetime import datetime, timedelta

def prune_old_memories(db, user_id, days=90):
    """Remove memories older than 90 days"""
    cutoff_timestamp = int((datetime.now() - timedelta(days=days)).timestamp())
    
    memories = db.get_user_memories(user_id=user_id)
    for memory in memories:
        if memory.updated_at and memory.updated_at < cutoff_timestamp:
            db.delete_user_memory(memory_id=memory.memory_id)

# Run periodically or before high-cost operations
prune_old_memories(db, user_id="john_doe@example.com")
```

### 缓解策略五：设置工具调用次数限制
通过限制每次会话的工具调用次数来防止内存操作失控
```python
agent = Agent(
    db=db,
    enable_agentic_memory=True,
    tool_call_limit=5  # Prevents excessive memory operations
)
```


### 常见陷阱
​
#### 用户 ID 陷阱
问题：忘记设置user_id会导致所有记忆都默认为user_id="default"，将不同用户的记忆混在一起。

```python
# ❌ Bad: All users share the same memories
agent.print_response("I love pizza")
agent.print_response("I'm allergic to dairy")

# ✅ Good: Each user has isolated memories
agent.print_response("I love pizza", user_id="user_123")
agent.print_response("I'm allergic to dairy", user_id="user_456")
```

最佳实践：始终user_id显式传递参数，尤其是在多用户应用程序中。


### 双重启用陷阱
问题：同时使用两者update_memory_on_run=True并enable_agentic_memory=True不能同时实现两者——代理模式会覆盖自动模式。

```python
# ❌ Doesn't work as expected - automatic memory is disabled
agent = Agent(
    db=db,
    update_memory_on_run=True,
    enable_agentic_memory=True  # This disables automatic behavior
)

# ✅ Choose one approach
agent = Agent(db=db, update_memory_on_run=True)  # Automatic
# OR
agent = Agent(db=db, enable_agentic_memory=True)  # Agentic
```


### 记忆增长监测
跟踪内存使用情况以便及早发现问题：

```python

from agno.agent import Agent

agent = Agent(db=db, update_memory_on_run=True)

# Check memory count for a user
memories = agent.get_user_memories(user_id="user_123")
print(f"User has {len(memories)} memories")

# Alert if memory count is unusually high
if len(memories) > 500:
    print("⚠️ Warning: User has excessive memories. Consider pruning.")
```


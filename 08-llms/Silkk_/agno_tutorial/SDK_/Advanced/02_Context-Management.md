# 上下文管理
设计并控制发送给语言模型的信息，以指导其行为。

上下文工程是指设计和控制发送给语言模型的信息（上下文）以指导其行为和输出的过程。实际上，构建上下文归根结底就是一个问题：“哪些信息最有可能实现预期结果？”
在 Agno 中，这意味着要精心构建系统消息，其中包括代理或团队描述、说明和其他相关设置。通过周密地构建此上下文，您可以：

- 引导你的代理人或团队朝着特定的行为或角色发展。
- 限制或扩展您的代理人或团队的能力。
- 确保输出结果一致、相关，并符合应用程序的需求。
- 支持多步骤推理、工具使用或结构化输出等高级用例。


有效的上下文工程是一个迭代过程：完善系统消息，尝试不同的描述和指令，并使用模式、委托和工具集成等功能。

Agno代理的上下文包括以下内容：

- 系统消息：系统消息是发送给代理或团队的主要上下文，包括所有其他上下文。
- 用户消息：用户消息是发送给代理或团队的消息。
- 聊天记录：聊天记录是客服人员或团队与用户之间对话的历史记录。
- 补充说明：添加到上下文中的任何少量示例或其他补充说明。

### 上下文缓存
大多数模型提供商都支持系统和用户消息的缓存，但不同提供商的实现方式有所不同。
通常的做法是缓存重复内容和常用指令，然后在后续请求中将这些缓存内容作为系统消息的前缀。换句话说，如果模型支持，可以通过在系统消息开头添加静态内容来减少发送给模型的令牌数量。

Agno 的上下文构建机制旨在将最有可能出现的静态内容置于系统消息的开头。

如果您希望对此进行微调，建议手动设置系统消息。

提示缓存的一些示例：


## 上下文工程

为代理配置**系统消息**、**指令**和**上下文**。

上下文工程是指设计和控制发送给语言模型的信息（上下文）以指导其行为和输出的过程。实际上，构建上下文归根结底就是一个问题：“哪些信息最有可能实现预期结果？”

Agno代理的上下文包括以下内容：
- 系统消息：系统消息是发送给代理的主要上下文，包括所有其他上下文。
- 用户消息：用户消息是发送给代理的消息。
- 聊天记录：聊天记录是客服人员和用户之间对话的历史记录。
- 补充说明：添加到上下文中的任何少量示例或其他补充说明。


### 系统消息上下文
以下是用于创建系统消息的一些关键参数：
- 描述：指导代理整体行为的描述。
- 说明：一份关于如何实现目标的精确、针对具体任务的说明清单。
- 预期输出：对代理的预期输出的描述。
- 系统消息由代理的描述、指令和其他设置构成。


```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    description="You are a famous short story writer asked to write for a magazine",
    instructions=["Always write 2 sentence stories."],
    markdown=True,
    debug_mode=True,  # Set to True to view the detailed logs and see the compiled system message
)
agent.print_response("Tell me a horror story.", stream=True)
```

将生成以下系统消息：


```xml
You are a famous short story writer asked to write for a magazine                                                                          
<instructions>                                                                                                                             
- Always write 2 sentence stories.                                                                                                         
</instructions>                                                                                                                            
                                                                                                                                            
<additional_information>                                                                                                                   
- Use markdown to format your answer
</additional_information>
```

> 默认情况下，指令不会被包裹在<instructions>标签中。如果您希望将指令包裹在 XML 标签中（例如，在使用受益于 XML 结构的模型时），请设置add_instruction_tags=True：

```python
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    description="You are a famous short story writer",
    instructions=["Always write 2 sentence stories."],
    add_instruction_tags=True,  # Instructions will be wrapped in <instructions> tags
)
```

### 系统消息参数
代理程序会创建一个默认系统消息，可以使用以下代理程序参数对其进行自定义：


| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| description | str | None | 添加到系统消息开头的代理描述。 |
| instructions | List［str］ | None | 添加到系统提示符＜instructions＞标签中的指令列表。还会根据 ‘＜input＞’ markdown 、＂expected＿output＜output＞＂等参数的值创建默认指令。 |
| add＿instruction＿tags | bool | True | 如果为 True，则将指令包裏在＜instructions＞标签中。设置为 False 则按原样传递指令，不使用 XML 标签。 |
| additional＿context | str | None | 在系统消息末尾添加了更多上下文信息。 |
| expected＿output | str | None | 提供代理的预期输出。此输出将添加到系统消息的末尾。 |
| markdown | bool | False | 添加一条指令，使用 Markdown 格式化输出。 |
| add＿datetime＿to＿context | bool | False | 如果为真，则将当前日期时间添加到提示中，以便客服人员了解时间。这样就可以在提示中使用＂明天＂之类的相对时间。 |
| add＿name＿to＿context | bool | False | 如果为真，则将代理名称添加到上下文中。 |
| add＿location＿to＿context | bool | False | 如果为真，则将代理的位置添加到上下文中。这样可以实现位置感知响应和本地上下文。 |
| add＿session＿summary＿to＿co ntext | bool | False | 如果为 True，则将会话摘要添加到上下文中。有关更多信息，请参阅会话部分。 |
| add＿memories＿to＿context | bool | False | 如果为真，则将用户记忆添加到上下文中。有关更多信息，请参阅＂记忆＂部分。 |
| add＿session＿state＿to＿cont ext | bool | False | 如果为 True，则将会话状态添加到上下文中。有关更多信息，请参阅状态文档。 |
| enable＿agentic＿knowledge＿ filters | bool | False | 如果为真，则允许智能体选择知识过滤器。有关更多信息，请参阅＂知识＂部分。 |
| system＿message | str | None | 覆盖默认系统消息。 |
| build＿context | bool | True | （可选）禁用上下文构建。 |


请参阅完整的代理商参考资料以获取更多信息。


#### 系统消息是如何构建的
让我们以以下示例代理为例：

```python
from agno.agent import Agent

agent = Agent(
    name="Helpful Assistant",
    role="Assistant",
    description="You are a helpful assistant",
    instructions=["Help the user with their question"],
    additional_context="""
    Here is an example of how to answer the user's question: 
        Request: What is the capital of France?
        Response: The capital of France is Paris.
    """,
    expected_output="You should format your response with `Response: <response>`",
    markdown=True,
    add_datetime_to_context=True,
    add_location_to_context=True,
    add_name_to_context=True,
    add_session_summary_to_context=True,
    add_memories_to_context=True,
    add_session_state_to_context=True,
)
```
以下是即将生成的系统消息：

```xml
You are a helpful assistant
<your_role>
Assistant
</your_role>

<instructions>
  Help the user with their question
</instructions>

<additional_information>
Use markdown to format your answers.
The current time is 2025-09-30 12:00:00.
Your approximate location is: New York, NY, USA.
Your name is: Helpful Assistant.
</additional_information>

<expected_output>
  You should format your response with `Response: <response>`
</expected_output>

Here is an example of how to answer the user's question: 
    Request: What is the capital of France?
    Response: The capital of France is Paris.

You have access to memories from previous interactions with the user that you can use:

<memories_from_previous_interactions>
- User really likes Digimon and Japan.
- User really likes Japan.
- User likes coffee.
</memories_from_previous_interactions>

Note: this information is from previous interactions and may be updated in this conversation. You should always prefer information from this conversation over the past memories.

Here is a brief summary of your previous interactions:

<summary_of_previous_interactions>
The user asked about information about Digimon and Japan.
</summary_of_previous_interactions>

Note: this information is from previous interactions and may be outdated. You should ALWAYS prefer information from this conversation over the past summary.

<session_state> ... </session_state>
```

#### 补充背景信息
您可以使用参数在系统消息末尾添加其他上下文additional_context。

此处additional_context向系统消息添加一条注释，表明代理可以访问特定的数据库表。

```python
from textwrap import dedent

from agno.agent import Agent
from agno.models.langdb import LangDB
from agno.tools.duckdb import DuckDbTools

duckdb_tools = DuckDbTools(
    create_tables=False, export_tables=False, summarize_tables=False
)
duckdb_tools.create_table_from_path(
    path="https://phidata-public.s3.amazonaws.com/demo_data/IMDB-Movie-Data.csv",
    table="movies",
)

agent = Agent(
    model=LangDB(id="llama3-1-70b-instruct-v1.0"),
    tools=[duckdb_tools],
    markdown=True,
    additional_context=dedent("""\
    You have access to the following tables:
    - movies: contains information about movies from IMDB.
    """),
)
agent.print_response("What is the average rating of movies?", stream=True)
```


#### 工具使用说明
如果您在代理程序中使用工具包，则可以使用以下instructions参数将工具说明添加到系统消息中：

```python
from agno.agent import Agent
from agno.tools.slack import SlackTools

slack_tools = SlackTools(
    instructions=["Use `send_message` to send a message to the user.  If the user specifies a thread, use `send_message_thread` to send a message to the thread."],
    add_instructions=True,
)
agent = Agent(
    tools=[slack_tools],
)
```

这些指令会在标签<additional_information>之后注入到系统消息中。

### 代理记忆
如果您已在代理上enable_agentic_memory进行设置True，则代理将能够使用工具创建/更新用户记忆。

这会将以下内容添加到系统消息中：

```xml
<updating_user_memories>
- You have access to the `update_user_memory` tool that you can use to add new memories, update existing memories, delete memories, or clear all memories.
- If the user's message includes information that should be captured as a memory, use the `update_user_memory` tool to update your memory database.
- Memories should include details that could personalize ongoing interactions with the user.
- Use this tool to add new memories or update existing memories that you identify in the conversation.
- Use this tool if the user asks to update their memory, delete a memory, or clear all memories.
- If you use the `update_user_memory` tool, remember to pass on the response to the user.
</updating_user_memories>
```

#### 智能体知识过滤器
如果你的代理启用了知识库功能，你可以让代理使用enable_agentic_knowledge_filters参数选择知识过滤器。
这将向系统消息中添加以下内容：

```xml
The knowledge base contains documents with these metadata filters: [filter1, filter2, filter3].
Always use filters when the user query indicates specific metadata.

Examples:
1. If the user asks about a specific person like "Jordan Mitchell", you MUST use the search_knowledge_base tool with the filters parameter set to {{'<valid key like user_id>': '<valid value based on the user query>'}}.
2. If the user asks about a specific document type like "contracts", you MUST use the search_knowledge_base tool with the filters parameter set to {{'document_type': 'contract'}}.
4. If the user asks about a specific location like "documents from New York", you MUST use the search_knowledge_base tool with the filters parameter set to {{'<valid key like location>': 'New York'}}.

General Guidelines:
- Always analyze the user query to identify relevant metadata.
- Use the most specific filter(s) possible to narrow down results.
- If multiple filters are relevant, combine them in the filters parameter (e.g., {{'name': 'Jordan Mitchell', 'document_type': 'contract'}}).
- Ensure the filter keys match the valid metadata filters: [filter1, filter2, filter3].

You can use the search_knowledge_base tool to search the knowledge base and get the most relevant documents. Make sure to pass the filters as [Dict[str: Any]] to the tool. FOLLOW THIS STRUCTURE STRICTLY.
```
请在知识过滤器部分详细了解智能体知识过滤器。

#### 直接设置系统消息
您可以使用参数手动设置系统消息system_message。这将忽略所有其他设置，并使用您提供的系统消息。

```python
from agno.agent import Agent
agent.print_response("What is the capital of France?")

agent = Agent(system_message="Share a 2 sentence story about")
agent.print_response("Love in the year 12000.")
```

> 某些模型提供商（例如 Groq）提供的某些模型llama-3.2-11b-vision-preview不需要与其他消息一起使用系统消息。要移除系统消息，请设置 `setSystemMessage`build_context=False和 ` setSystemMessage` system_message=None。此外，如果markdown=True设置了 `setSystemMessage`，则会添加系统消息，因此请将其移除或显式禁用系统消息。

### 用户消息上下文
发送input到Agent.run()或Agent.print_response()用作用户消息。

#### 用户消息的其他上下文
您可以使用以下代理参数为用户消息添加更多上下文：
以下代理参数用于配置用户消息的构建方式：
- add_knowledge_to_context
- add_dependencies_to_context


```python
from agno.agent import Agent
agent = Agent(add_knowledge_to_context=True, add_dependencies_to_context=True)
agent.print_response("What is the capital of France?", dependencies={"name": "John Doe"})
```


发送给模型的用户消息将如下所示：


```xml
What is the capital of France?

Use the following references from the knowledge base if it helps:
<references>
- Reference 1
- Reference 2
</references>

<additional context>
{"name": "John Doe"}
</additional context>
```


请参阅依赖项，了解如何为用户消息进行依赖注入。
​
#### 聊天记录

如果您的代理启用了数据库存储，则会话历史记录会自动存储（请参阅会话）。

现在可以使用`add_history_to_context`以下方法将对话历史记录添加到上下文中。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses

db = SqliteDb(db_file="tmp/agent.db")

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    session_id="chat_history",
    instructions="You are a helpful assistant that can answer questions about space and oceans.",
    add_history_to_context=True,
    num_history_runs=2,
)

agent.print_response("Where is the sea of tranquility?", stream=True)

agent.print_response("What was my first question?", stream=True)
```

这将把对话历史记录添加到上下文中，该上下文可用于为下一条消息提供背景信息。


### 管理工具调用
max_tool_calls_from_history参数可用于仅将历史中最近的n个工具调用添加到上下文中。

这有助于管理上下文大小并降低代理运行期间的令牌成本。

请看以下示例：

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
import random

def get_weather_for_city(city: str) -> str:
    conditions = ["Sunny", "Cloudy", "Rainy", "Snowy", "Foggy", "Windy"]
    temperature = random.randint(-10, 35)
    condition = random.choice(conditions)

    return f"{city}: {temperature}°C, {condition}"

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[get_weather_for_city],
    db=SqliteDb(db_file="tmp/agent.db"),
    add_history_to_context=True,
    max_tool_calls_from_history=3,  # Keep only last 3 tool calls in context
)
agent.print_response("What's the weather in Tokyo?")
agent.print_response("What's the weather in Paris?")  
agent.print_response("What's the weather in London?")
agent.print_response("What's the weather in Berlin?")
agent.print_response("What's the weather in Mumbai?")
agent.print_response("What's the weather in Miami?")
agent.print_response("What's the weather in New York?")
agent.print_response("What's the weather in above cities?")
```


该模型给出了最后三个城市（孟买、迈阿密和纽约）的天气预报。

在这个例子中：

- 运行 1-3：模型看到工具调用 [1]、[1,2]、[1,2,3]
- 运行 4：模型检测到工具调用 [2,3,4]（已过滤掉工具调用 1）
- 运行 5：模型检测到工具调用 [3,4,5]（已过滤掉工具调用 1-2）

> 重要提示： max_tool_calls_from_history过滤掉由加载的运行中的工具调用num_history_runs。您的数据库始终包含完整的历史记录。

### 利用额外输入进行少样本学习
您可以使用参数向代理的上下文中添加完整的附加消息additional_input。这些消息会像对话历史记录的一部分一样添加到上下文中。

您可以给代理提供一些示例，说明它应该如何响应（也称为“少次提示”）：

```python
from agno.agent import Agent
from agno.models.message import Message
from agno.models.openai import OpenAIResponses

# Few-shot examples showing good support patterns
support_examples = [
    # Example 1: Simple issue resolution
    Message(role="user", content="I forgot my password and can't log in"),
    Message(
        role="assistant",
        content="""
    I'll help you reset your password right away.

    **Steps to Reset Your Password:**
    . Go to the login page and click "Forgot Password"
    . Enter your email address
    . Check your email for the reset link
    . Follow the link to create a new password
    . Use a strong password with uppercase, lowercase, numbers, and symbols

    f you don't receive the email within 5 minutes, please check your spam folder. Let me know if you need any assistance with these steps!
    """.strip(),
    ),
    # Example 2: Billing issue pattern
    Message(
        role="user",
        content="I've been charged twice for the same order and I'm frustrated!",
    ),
    Message(
        role="assistant",
        content="""
    I sincerely apologize for the billing error and the frustration this has caused you.

    **Immediate Action Plan:**
    1. I'll investigate your account to confirm the duplicate charge
    2. Process a full refund for the duplicate transaction
    3. Ensure this doesn't happen again by checking your payment method settings
    4. Provide you with a confirmation number once the refund is processed

    The refund typically takes 3-5 business days to appear on your statement. As an apology for this inconvenience, I'd also like to offer you a 10% discount on your next order.

    Is there anything else I can help you with regarding this issue?
    """.strip(),
    ),
    # Example 3: Technical support pattern
    Message(role="user", content="Your app keeps crashing when I try to upload photos"),
    Message(
        role="assistant",
        content="""
    I'm sorry you're experiencing crashes with photo uploads. Let me help you troubleshoot this issue.

    **Troubleshooting Steps:**
    1. **Check App Version:** Ensure you have the latest version from the app store
    2. **Restart the App:** Close completely and reopen the application
    3. **Check Storage:** Make sure you have sufficient device storage (at least 1GB free)
    4. **Photo Size:** Try uploading smaller photos (under 10MB each)
    5. **Network Connection:** Ensure you have a stable internet connection

    **If the issue persists:**
    - Try uploading one photo at a time instead of multiple
    - Clear the app cache in your device settings
    - Restart your device

    If none of these steps resolve the issue, please let me know your device type and OS version, and I'll escalate this to our technical team for further investigation.
    """.strip(),
    ),
]

if __name__ == "__main__":
    # Create agent with few-shot learning
    agent = Agent(
        name="Customer Support Specialist",
        model=OpenAIResponses(id="gpt-5.2"),
        add_name_to_context=True,
        additional_input=support_examples,  # few-shot learning examples
        instructions=[
            "You are an expert customer support specialist.",
            "Always be empathetic, professional, and solution-oriented.",
            "Provide clear, actionable steps to resolve customer issues.",
            "Follow the established patterns for consistent, high-quality support.",
        ],
        markdown=True,
    )

    for i, example in enumerate(support_examples, 1):
        print(f"Example {i}: {example}")
        print("-" * 50)
        agent.print_response(example)
```

### 上下文缓存
大多数模型提供商都支持系统和用户消息的缓存，但不同提供商的实现方式有所不同。

通常的做法是缓存重复内容和常用指令，然后在后续请求中将这些缓存内容作为系统消息的前缀。换句话说，如果模型支持，可以通过在系统消息开头添加静态内容来减少发送给模型的令牌数量。

Agno 的上下文构建机制旨在将最有可能出现的静态内容置于系统消息的开头。

如果您希望对此进行微调，建议手动设置系统消息。


## 基本说明

本示例演示了如何向代理提供基本指令，以指导其响应行为和叙事风格。

### 代码
```python
from agno.agent import Agent

agent = Agent(instructions="Share a 2 sentence story about")
agent.print_response("Love in the year 12000.")

```

## 动态指令
此示例演示了如何创建根据会话状态而改变的动态指令，从而为不同的用户提供个性化的代理行为。

### 代码

```python
from agno.agent import Agent
from agno.run import RunContext

def get_instructions(run_context: RunContext):
    if not run_context.session_state:
        run_context.session_state = {}

    if run_context.session_state.get("current_user_id"):
        return f"Make the story about {run_context.session_state.get('current_user_id')}."

    return "Make the story about the user."


agent = Agent(instructions=get_instructions)
agent.print_response("Write a 2 sentence story", user_id="john.doe")
```

## 通过功能发出指令
本示例演示了如何通过可访问代理属性的函数向代理提供指令，从而实现动态和个性化的指令生成。

### 代码

```python
from typing import List

from agno.agent import Agent


def get_instructions(agent: Agent) -> List[str]:
    return [
        f"Your name is {agent.name}!",
        "Talk in haiku's!",
        "Use poetry to answer questions.",
    ]


agent = Agent(
    name="AgentX",
    instructions=get_instructions,
    markdown=True,
)
agent.print_response("Who are you?", stream=True)
```

## Few-Shot Learning
本示例演示了如何使用 additional_input 和 Agent 通过少样本学习来教授正确的响应模式，特别是针对客户支持场景。

### 代码

```python
"""
This example demonstrates how to use additional_input with an Agent
to teach proper response patterns through few-shot learning.
"""

from agno.agent import Agent
from agno.models.message import Message
from agno.models.openai import OpenAIResponses

# Few-shot examples showing good support patterns
support_examples = [
    # Example 1: Simple issue resolution
    Message(role="user", content="I forgot my password and can't log in"),
    Message(
        role="assistant",
        content="""
    I'll help you reset your password right away.

    **Steps to Reset Your Password:**
    . Go to the login page and click "Forgot Password"
    . Enter your email address
    . Check your email for the reset link
    . Follow the link to create a new password
    . Use a strong password with uppercase, lowercase, numbers, and symbols

    f you don't receive the email within 5 minutes, please check your spam folder. Let me know if you need any assistance with these steps!
    """.strip(),
    ),
    # Example 2: Billing issue pattern
    Message(
        role="user",
        content="I've been charged twice for the same order and I'm frustrated!",
    ),
    Message(
        role="assistant",
        content="""
    I sincerely apologize for the billing error and the frustration this has caused you.

    **Immediate Action Plan:**
    1. I'll investigate your account to confirm the duplicate charge
    2. Process a full refund for the duplicate transaction
    3. Ensure this doesn't happen again by checking your payment method settings
    4. Provide you with a confirmation number once the refund is processed

    The refund typically takes 3-5 business days to appear on your statement. As an apology for this inconvenience, I'd also like to offer you a 10% discount on your next order.

    Is there anything else I can help you with regarding this issue?
    """.strip(),
    ),
    # Example 3: Technical support pattern
    Message(role="user", content="Your app keeps crashing when I try to upload photos"),
    Message(
        role="assistant",
        content="""
    I'm sorry you're experiencing crashes with photo uploads. Let me help you troubleshoot this issue.

    **Troubleshooting Steps:**
    1. **Check App Version:** Ensure you have the latest version from the app store
    2. **Restart the App:** Close completely and reopen the application
    3. **Check Storage:** Make sure you have sufficient device storage (at least 1GB free)
    4. **Photo Size:** Try uploading smaller photos (under 10MB each)
    5. **Network Connection:** Ensure you have a stable internet connection

    **If the issue persists:**
    - Try uploading one photo at a time instead of multiple
    - Clear the app cache in your device settings
    - Restart your device

    If none of these steps resolve the issue, please let me know your device type and OS version, and I'll escalate this to our technical team for further investigation.
    """.strip(),
    ),
]

if __name__ == "__main__":
    # Create agent with few-shot learning
    agent = Agent(
        name="Customer Support Specialist",
        model=OpenAIResponses(id="gpt-5.2"),
        add_name_to_context=True,
        additional_input=support_examples,  # few-shot learning examples
        instructions=[
            "You are an expert customer support specialist.",
            "Always be empathetic, professional, and solution-oriented.",
            "Provide clear, actionable steps to resolve customer issues.",
            "Follow the established patterns for consistent, high-quality support.",
        ],
        debug_mode=True,
        markdown=True,
    )

    for i, example in enumerate(support_examples, 1):
        print(f"Example {i}: {example}")
        print("-" * 50)
        agent.print_response(example)
```


## 提供日期时间

本示例演示了如何向代理指令添加当前日期和时间上下文，使代理能够提供与时间相关的响应。

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    add_datetime_to_context=True,
    timezone_identifier="Etc/UTC",
)
agent.print_response(
    "What is the current date and time? What is the current time in NYC?"
)
```

## 提供位置
本示例演示了如何向代理指令添加位置上下文，使代理能够提供特定位置的响应并搜索本地信息。


```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    add_location_to_context=True,
    tools=[HackerNewsTools(cache_results=True)],
)
agent.print_response("What city am I in?", stream=True)
agent.print_response("Search for tech news relevant to my location", stream=True)
```
​

## 管理工具调用


本示例演示了如何使用max_tool_calls_from_history限制代理上下文中包含的工具调用次数。

这有助于管理上下文大小并降低令牌成本，同时还能在数据库中保持完整的历史记录。

```python
import random

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses


def get_weather_for_city(city: str) -> str:
    """Get weather for a city"""
    conditions = ["Sunny", "Cloudy", "Rainy", "Snowy", "Foggy", "Windy"]
    temperature = random.randint(-10, 35)
    condition = random.choice(conditions)
    return f"{city}: {temperature}°C, {condition}"


agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[get_weather_for_city],
    instructions="You are a weather assistant. Get the weather using the get_weather_for_city tool.",
    # Only keep 3 most recent tool calls in context
    max_tool_calls_from_history=3,
    db=SqliteDb(db_file="tmp/weather_data.db"),
    add_history_to_context=True,
    markdown=True,
)

cities = [
    "Tokyo",
    "Delhi",
    "Shanghai",
    "São Paulo",
    "Mumbai",
    "Beijing",
    "Cairo",
    "London",
]

print(
    f"{'Run':<5} | {'City':<15} | {'History':<8} | {'Current':<8} | {'In Context':<11} | {'In DB':<8}"
)
print("-" * 90)

for i, city in enumerate(cities, 1):
    run_response = agent.run(f"What's the weather in {city}?")

    # Count tool calls in context
    history_tool_calls = sum(
        len(msg.tool_calls)
        for msg in run_response.messages
        if msg.role == "assistant"
        and msg.tool_calls
        and getattr(msg, "from_history", False)
    )

    # Count tool calls from current run
    current_tool_calls = sum(
        len(msg.tool_calls)
        for msg in run_response.messages
        if msg.role == "assistant"
        and msg.tool_calls
        and not getattr(msg, "from_history", False)
    )

    total_in_context = history_tool_calls + current_tool_calls

    # Total tool calls stored in database (unfiltered)
    saved_messages = agent.get_session_messages()
    saved_tool_calls = (
        sum(
            len(msg.tool_calls)
            for msg in saved_messages
            if msg.role == "assistant" and msg.tool_calls
        )
        if saved_messages
        else 0
    )

    print(
        f"{i:<5} | {city:<15} | {history_tool_calls:<8} | {current_tool_calls:<8} | {total_in_context:<11} | {saved_tool_calls:<8}"
    )
```

## 上下文工

为团队配置系统消息、指令和上下文。

上下文工程是指设计和控制发送给语言模型的信息（上下文）以指导其行为和输出的过程。实际上，构建上下文归根结底就是一个问题：“哪些信息最有可能实现预期结果？”
有效的上下文工程是一个迭代过程：完善系统消息，尝试不同的描述和指令，并使用模式、委托和工具集成等功能。

Agno团队的组成要素如下：
- 系统消息：系统消息是发送给团队的主要上下文，包含所有其他上下文。
- 用户消息：用户消息是发送给团队的消息。
- 聊天记录：聊天记录是团队与用户之间对话的历史记录。
- 补充说明：添加到上下文中的任何少量示例或其他补充说明。


### 系统消息上下文
以下是用于创建系统消息的一些关键参数：
- 描述：指导团队整体行为的描述。
- 说明：一份关于如何实现目标的精确、针对具体任务的说明清单。
- 预期输出：团队预期输出的描述。
- 成员：团队成员信息、角色和能力。
- 系统消息由团队描述、指令、成员详情和其他设置构成。团队领导的系统消息还包含委派规则和协调指南。例如：

```python
from agno.agent import Agent
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools

news_agent = Agent(
    name="News Researcher",
    role="You are a news researcher that can find information on HackerNews.",
    instructions=[
        "Use your HackerNews tool to find tech news and discussions.",
        "Provide a summary of the information found.",
    ],
    tools=[HackerNewsTools()],
    markdown=True,
    debug_mode=True,
)
finance_agent = Agent(
    name="Finance Researcher",
    role="You are a finance researcher that can get stock prices and market data.",
    instructions=[
        "Use your finance tools to get stock prices and financial data.",
        "Provide a summary of the information found.",
    ],
    tools=[YFinanceTools()],
    markdown=True,
    debug_mode=True,
)

team = Team(
    members=[news_agent, finance_agent],
    instructions=[
        "You are a team of researchers that can find tech news and financial data.",
        "After finding information about the topic, compile a joint report."
    ],
    markdown=True,
    debug_mode=True,
)
team.print_response("What is the latest news on AI and how is NVDA performing?", stream=True)
```

将生成以下系统消息：

```xml
You are the leader of a team and sub-teams of AI Agents.
Your task is to coordinate the team to complete the user's request.

Here are the members in your team:
<team_members>
- Agent 1:
    - ID: news-researcher
    - Name: News Researcher
    - Role: You are a news researcher that can find information on HackerNews.
    - Member tools:
        - get_top_hackernews_stories
        - get_user_details
- Agent 2:
    - ID: finance-researcher
    - Name: Finance Researcher
    - Role: You are a finance researcher that can get stock prices and market data.
    - Member tools:
        - get_stock_price
        - get_analyst_recommendations
</team_members>

<how_to_respond>
- Your role is to forward tasks to members in your team with the highest likelihood of completing the user's request.
- Carefully analyze the tools available to the members and their roles before delegating tasks.
- You cannot use a member tool directly. You can only delegate tasks to members.
- When you delegate a task to another member, make sure to include:
    - member_id (str): The ID of the member to delegate the task to. Use only the ID of the member, not the ID of the team followed by the ID of the
member.
    - task_description (str): A clear description of the task.
    - expected_output (str): The expected output.
- You can delegate tasks to multiple members at once.
- You must always analyze the responses from members before responding to the user.
- After analyzing the responses from the members, if you feel the task has been completed, you can stop and respond to the user.
- If you are not satisfied with the responses from the members, you should re-assign the task.
- For simple greetings, thanks, or questions about the team itself, you should respond directly.
- For all work requests, tasks, or questions requiring expertise, route to appropriate team members.
</how_to_respond>

<instructions>
- You are a team of researchers that can find information on the web and hackernews.
- After finding information about the topic, compile a joint report.
</instructions>

<additional_information>
- Use markdown to format your answers.
</additional_information>
```

> 默认情况下，指令不会被包裹在<instructions>标签中。如果您希望将指令包裹在 XML 标签中（例如，在使用受益于 XML 结构的模型时），请设置use_instruction_tags=True：


```python
team = Team(
    members=[news_agent, finance_agent],
    instructions=["Coordinate the team to provide comprehensive research"],
    use_instruction_tags=True,  # Instructions will be wrapped in <instructions> tags
)

```

### 系统消息参数
团队创建了一个默认系统消息，可以使用以下参数对其进行自定义：

| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| description | str | None | 添加到系统消息开头的团队描述。 |
| instructions | List［str］ | None | 添加到系统提示符＜instructions＞标签中的指令列表。还会根据＂＜input＞ markdown、ˋ expected＿output＜output＞’等参数的值创建默认指令。 |
| use＿instruction＿tags | bool | False | 如果为 True，则将指令包裹在＜instructions＞标签中。设置为 False 则按原样传递指令，不使用 XML 标签。 |
| additional＿context | str | None | 在系统消息末尾添加了更多上下文信息。 |
| expected＿output | str | None | 请提供团队的预期输出。此信息将添加到系统消息的末尾。 |
| markdown | bool | False | 添加一条指令，使用 Markdown 格式化输出。 |
| add＿datetime＿to＿context | bool | False | 如果为真，则将当前日期时间添加到提示中，以便团队对时间有个概念。这样就可以在提示中使用＂明天＂之类的相对时间。 |
| add＿name＿to＿context | bool | False | 如果为真，则将团队名称添加到上下文中。 |
| add＿location＿to＿context | bool | False | 如果为真，则将团队位置添加到上下文中。这样可以实现位置感知响应和本地上下文。 |
| timezone＿identifier | str | None | 允许为日期时间指令自定义时区，遵循 TZ 数据库格式（例如＂Etc／UTC＂）。 |
| add＿member＿tools＿to＿conte xt | bool | True | 如果为真，则将团队成员可用的工具添加到上下文中。 |
| add＿session＿summary＿to＿co ntext | bool | False | 如果为 True，则将会话摘要添加到上下文中。有关更多信息，请参阅会话部分。 |
| add＿memories＿to＿context | bool | False | 如果为真，则将用户记忆添加到上下文中。有关更多信息，请参阅＂记忆＂部分。 |
| add＿dependencies＿to＿conte xt | bool | False | 如果为 True，则将依赖项添加到上下文中。有关更多信息，请参阅依赖项。 |
| add＿session＿state＿to＿cont ext | bool | False | 如果为 True，则将会话状态添加到上下文中。有关更多信息，请参阅状态文档。 |
| add＿knowledge＿to＿context | bool | False | 如果为真，则将检索到的知识添加到上下文中，以启用红绿灯机制。有关更多信息，请参阅知识库。 |
| enable＿agentic＿knowledge＿ filters | bool | False | 如果为真，则允许团队选择知识筛选器。有关更多信息，请参阅知识库。 |
| system＿message | str | None | 藋盖默认系统消息。 |
| respond＿directly | bool | False | 如果为 True，团队领导将不会处理成员的回复，而是直接返回回复。不能与 delegate＿to＿all＿members＝True ． |
| delegate＿to＿all＿members | bool | False | 如果为真，团队领导会将任务同时分配给所有成员，而不是逐个分配。使用异步方式运行时，arun 成员将并发执行。不能与 respond＿directly＝True． |
| determine＿input＿for＿membe rs | bool | True | 如果要将运行输入直接发送给成员代理，请设置为 false。 |
| share＿member＿interaction s | bool | False | 如果为真，则将所有之前的成员互动发送给成员。 |
| get＿member＿information＿to ol | bool | False | 如果为真，则添加一个工具来获取团队成员的信息。 |


> 配置警告：同时设置delegate_to_all_members=True和respond_directly=True记录警告并禁用respond_directly。


### 编排参数
Team 类提供了多个参数来控制团队领导者如何协调团队成员之间的工作：

| 范围 | 类型 | 默认 | 描述 |
| :--- | :---: | :---: | :--- |
| share＿member＿interact <br> ions | bool | False | 如果为真，则之前的成员互动记录会共享给获得新委托的成员。 |

```python
from agno.agent import Agent
from agno.team import Team

# Create a team with orchestration controls
team = Team(
    members=[research_agent, analysis_agent],
    share_member_interactions=True,
)
```

#### 系统消息是如何构建的
让我们以以下团队为例：

```python
from agno.agent import Agent
from agno.team import Team

web_agent = Agent(
    name="Web Researcher",
    role="You are a web researcher that can find information on the web.",
    description="You are a helpful web research assistant",
    instructions=["Search for accurate information"],
    markdown=True,
)

team = Team(
    members=[web_agent],
    name="Research Team",
    role="Team Lead",
    description="You are a research team lead",
    instructions=["Coordinate the team to provide comprehensive research"],
    expected_output="You should format your response with detailed findings",
    markdown=True,
    add_datetime_to_context=True,
    add_location_to_context=True,
    add_name_to_context=True,
    add_session_summary_to_context=True,
    add_memories_to_context=True,
    add_session_state_to_context=True,
)
```

以下是即将生成的系统消息：

```xml
You are the leader of a team and sub-teams of AI Agents.
Your task is to coordinate the team to complete the user's request.

Here are the members in your team:
<team_members>
- Agent 1:
    - ID: web-researcher
    - Name: Web Researcher
    - Role: You are a web researcher that can find information on the web.
    - Member tools:
        (none)
</team_members>

<how_to_respond>
...
</how_to_respond>

You have access to memories from previous interactions with the user that you can use:

<memories_from_previous_interactions>
- User really likes Digimon and Japan.
- User really likes Japan.
- User likes coffee.
</memories_from_previous_interactions>

Note: this information is from previous interactions and may be updated in this conversation. You should always prefer information from this conversation over the past memories.

Here is a brief summary of your previous interactions:

<summary_of_previous_interactions>
The user asked about information about Digimon and Japan.
</summary_of_previous_interactions>

Note: this information is from previous interactions and may be outdated. You should ALWAYS prefer information from this conversation over the past summary.

<description>
You are a research team lead
</description>

<your_role>
Team Lead
</your_role>

<instructions>
- Coordinate the team to provide comprehensive research
</instructions>

<additional_information>
- Use markdown to format your answers.
- The current time is 2025-09-30 12:00:00.
- Your approximate location is: New York, NY, USA.
- Your name is: Research Team.
</additional_information>

<expected_output>
You should format your response with detailed findings
</expected_output>

<session_state> ... </session_state>
```

### 补充背景信息
您可以使用参数additional_context在系统消息末尾添加其他上下文。
此处additional_context向系统消息添加一条注释，表明该团队可以访问特定的数据库表。

```python
from textwrap import dedent

from agno.agent import Agent
from agno.team import Team
from agno.models.langdb import LangDB
from agno.tools.duckdb import DuckDbTools
from agno.tools.duckduckgo import DuckDuckGoTools

duckdb_tools = DuckDbTools(
    create_tables=False, export_tables=False, summarize_tables=False
)
duckdb_tools.create_table_from_path(
    path="https://phidata-public.s3.amazonaws.com/demo_data/IMDB-Movie-Data.csv",
    table="movies",
)

web_researcher = Agent(
    name="Web Researcher",
    role="You are a web researcher that can find information on the web.",
    tools=[DuckDuckGoTools()],
    instructions=[
        "Use your web search tool to find information on the web.",
        "Provide a summary of the information found.",
    ],
)

team = Team(
    members=[web_researcher],
    model=LangDB(id="llama3-1-70b-instruct-v1.0"),
    tools=[duckdb_tools],
    markdown=True,
    additional_context=dedent("""\
    You have access to the following tables:
    - movies: contains information about movies from IMDB.
    """),
)
team.print_response("What is the average rating of movies?", stream=True)
```

### 团队成员信息

成员信息会自动添加到系统消息中，包括成员 ID、姓名、角色和工具。您可以选择将其设置add_member_tools_to_context为 False，以减少此操作，这样即可从系统消息中移除成员工具信息。

你还可以给团队领导提供一个工具，让他/她了解团队成员的信息。

```python
from agno.agent import Agent
from agno.team import Team

web_agent = Agent(
    name="Web Researcher",
    role="You are a web researcher that can find information on the web."
)

team = Team(
    members=[web_agent],
    get_member_information_tool=True,  # Adds a tool to get information about team members
)
```

### 工具使用说明
如果您的团队正在使用工具包，您可以使用以下参数instructions将工具说明添加到系统消息中：

```python
from agno.agent import Agent
from agno.tools.slack import SlackTools

slack_tools = SlackTools(
    instructions=["Use `send_message` to send a message to the user.  If the user specifies a thread, use `send_message_thread` to send a message to the thread."],
    add_instructions=True,
)
team = Team(
    members=[...],
    tools=[slack_tools],
)

```

这些指令会在标签<additional_information>之后注入到系统消息中。


#### 代理记忆
如果您已在团队中`enable_agentic_memory`启用此功能True，则该团队可以使用工具创建/更新用户记忆。

这会将以下内容添加到系统消息中：

```xml
<updating_user_memories>
- You have access to the `update_user_memory` tool that you can use to add new memories, update existing memories, delete memories, or clear all memories.
- If the user's message includes information that should be captured as a memory, use the `update_user_memory` tool to update your memory database.
- Memories should include details that could personalize ongoing interactions with the user.
- Use this tool to add new memories or update existing memories that you identify in the conversation.
- Use this tool if the user asks to update their memory, delete a memory, or clear all memories.
- If you use the `update_user_memory` tool, remember to pass on the response to the user.
</updating_user_memories>
```

### 智能体知识过滤器
如果您的团队启用了知识库功能，您可以让团队成员使用该参数enable_agentic_knowledge_filters选择知识库筛选器。
这将向系统消息中添加以下内容：
```xml
The knowledge base contains documents with these metadata filters: [filter1, filter2, filter3].
Always use filters when the user query indicates specific metadata.

Examples:
1. If the user asks about a specific person like "Jordan Mitchell", you MUST use the search_knowledge_base tool with the filters parameter set to {{'<valid key like user_id>': '<valid value based on the user query>'}}.
2. If the user asks about a specific document type like "contracts", you MUST use the search_knowledge_base tool with the filters parameter set to {{'document_type': 'contract'}}.
4. If the user asks about a specific location like "documents from New York", you MUST use the search_knowledge_base tool with the filters parameter set to {{'<valid key like location>': 'New York'}}.

General Guidelines:
- Always analyze the user query to identify relevant metadata.
- Use the most specific filter(s) possible to narrow down results.
- If multiple filters are relevant, combine them in the filters parameter (e.g., {{'name': 'Jordan Mitchell', 'document_type': 'contract'}}).
- Ensure the filter keys match the valid metadata filters: [filter1, filter2, filter3].

You can use the search_knowledge_base tool to search the knowledge base and get the most relevant documents. Make sure to pass the filters as [Dict[str: Any]] to the tool. FOLLOW THIS STRUCTURE STRICTLY.
```

请在知识过滤器部分详细了解智能体知识过滤器。


#### 直接设置系统消息
您可以使用参数手动设置系统消息system_message。这将忽略所有其他设置，并使用您提供的系统消息。

```python
from agno.team import Team

team = Team(members=[], system_message="Share a 2 sentence story about")
team.print_response("Love in the year 12000.")
```

### 用户消息上下文
发送`input`到`Team.run()`或`Team.print_response()`用作用户消息。

请参阅依赖项，了解如何为用户消息进行依赖注入。

#### 用户消息的其他上下文
默认情况下，用户消息是使用input发送到Team.run()或Team.print_response()函数构建的。

以下团队参数用于配置用户消息的构建方式：

- add_knowledge_to_context
- add_dependencies_to_context


```python
from agno.agent import Agent
from agno.team import Team

web_agent = Agent(
    name="Web Researcher",
    role="You are a web researcher that can find information on the web."
)

team = Team(
    members=[web_agent],
    add_knowledge_to_context=True,
    add_dependencies_to_context=True
)
team.print_response("What is the capital of France?", dependencies={"name": "John Doe"})
```

发送给模型的用户消息将如下所示：

```xml
What is the capital of France?

Use the following references from the knowledge base if it helps:
<references>
- Reference 1
- Reference 2
</references>

<additional context>
{"name": "John Doe"}
</additional context>
```


### 聊天记录
如果您的团队启用了数据库存储，则会话历史记录会自动存储（请参阅会话）。
现在可以使用以下add_history_to_context方法将对话历史记录添加到上下文中。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools

db = SqliteDb(db_file="tmp/team.db")

news_researcher = Agent(
    name="News Researcher",
    role="You are a news researcher that can find information on HackerNews.",
    tools=[HackerNewsTools()],
    instructions=[
        "Use your HackerNews tool to find tech news and discussions.",
        "Provide a summary of the information found.",
    ],
)

team = Team(
    members=[news_researcher],
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    session_id="chat_history",
    instructions="You are a helpful assistant that can answer questions about technology.",
    add_history_to_context=True,
    num_history_runs=2,
)

team.print_response("What are the top stories on HackerNews?", stream=True)

team.print_response("What was my first question?", stream=True)
```

这将把对话历史记录添加到上下文中，该上下文可用于为下一条消息提供背景信息。


### 管理工具调用

该max_tool_calls_from_history参数可用于将n个历史记录中最近的工具调用添加到上下文中。

这有助于在团队运行期间管理上下文大小并降低令牌成本。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools

news_agent = Agent(
    name="News Researcher",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="You are a news researcher. Search HackerNews for tech news and discussions.",
    tools=[HackerNewsTools()],
)

finance_agent = Agent(
    name="Finance Researcher",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="You are a finance researcher. Get stock prices and financial data.",
    tools=[YFinanceTools()],
)

team = Team(
    members=[news_agent, finance_agent],
    model=OpenAIResponses(id="gpt-5.2"),
    db=SqliteDb(db_file="tmp/filter_history_tool_calls_team.db"),
    add_history_to_context=True,
    max_tool_calls_from_history=5,
    show_members_responses=True,
)

team.print_response("Search for AI news on HackerNews", stream=True)
team.print_response("Get the stock price for NVDA", stream=True)
team.print_response("Search for LLM discussions", stream=True)
# Older searches will be filtered from context
team.print_response("Get the stock price for AAPL", stream=True)
team.print_response("Search for startup news", stream=True)
team.print_response("What topics did I search for recently?", stream=True)
```

在这个例子中：
- 运行 1-3：团队看到工具调用 [1]、[1,2]、[1,2,3,4,5]
- 运行 4：团队查看最近 5 次工具调用范围内的工具调用（已过滤掉较早的工具调用）
- 运行 5：团队查看最近 5 次工具调用中的工具调用（较早的工具调用已被过滤掉）


### 补充输入
您可以使用参数向团队上下文添加完整的额外消息additional_input。这些消息会像对话历史记录的一部分一样添加到上下文中。

你可以给你的团队提供一些应对方式的例子（也称为“少次提示”）：

```python
from agno.team import Team
from agno.models.message import Message
from agno.models.openai import OpenAIResponses

# Few-shot examples showing good support patterns
support_examples = [
    # Example 1: Simple issue resolution
    Message(role="user", content="I forgot my password and can't log in"),
    Message(
        role="assistant",
        content="""
    I'll help you reset your password right away.

    **Steps to Reset Your Password:**
    . Go to the login page and click "Forgot Password"
    . Enter your email address
    . Check your email for the reset link
    . Follow the link to create a new password
    . Use a strong password with uppercase, lowercase, numbers, and symbols

    f you don't receive the email within 5 minutes, please check your spam folder. Let me know if you need any assistance with these steps!
    """.strip(),
    ),
    # Example 2: Billing issue pattern
    Message(
        role="user",
        content="I've been charged twice for the same order and I'm frustrated!",
    ),
    Message(
        role="assistant",
        content="""
    I sincerely apologize for the billing error and the frustration this has caused you.

    **Immediate Action Plan:**
    1. I'll investigate your account to confirm the duplicate charge
    2. Process a full refund for the duplicate transaction
    3. Ensure this doesn't happen again by checking your payment method settings
    4. Provide you with a confirmation number once the refund is processed

    The refund typically takes 3-5 business days to appear on your statement. As an apology for this inconvenience, I'd also like to offer you a 10% discount on your next order.

    Is there anything else I can help you with regarding this issue?
    """.strip(),
    ),
    # Example 3: Technical support pattern
    Message(role="user", content="Your app keeps crashing when I try to upload photos"),
    Message(
        role="assistant",
        content="""
    I'm sorry you're experiencing crashes with photo uploads. Let me help you troubleshoot this issue.

    **Troubleshooting Steps:**
    1. **Check App Version:** Ensure you have the latest version from the app store
    2. **Restart the App:** Close completely and reopen the application
    3. **Check Storage:** Make sure you have sufficient device storage (at least 1GB free)
    4. **Photo Size:** Try uploading smaller photos (under 10MB each)
    5. **Network Connection:** Ensure you have a stable internet connection

    **If the issue persists:**
    - Try uploading one photo at a time instead of multiple
    - Clear the app cache in your device settings
    - Restart your device

    If none of these steps resolve the issue, please let me know your device type and OS version, and I'll escalate this to our technical team for further investigation.
    """.strip(),
    ),
]

if __name__ == "__main__":
    # Create team with few-shot learning
    team = Team(
        members=[...],
        name="Customer Support Team",
        model=OpenAIResponses(id="gpt-5.2"),
        add_name_to_context=True,
        additional_input=support_examples,  # few-shot learning examples
        instructions=[
            "You are an expert customer support specialist.",
            "Always be empathetic, professional, and solution-oriented.",
            "Provide clear, actionable steps to resolve customer issues.",
            "Follow the established patterns for consistent, high-quality support.",
        ],
        markdown=True,
    )

    for i, example in enumerate(support_examples, 1):
        print(f"📞 Example {i}: {example}")
        print("-" * 50)
        team.print_response(example)
```

### 上下文缓存
大多数模型提供商都支持系统和用户消息的缓存，但不同提供商的实现方式有所不同。

通常的做法是缓存重复内容和常用指令，然后在后续请求中将缓存内容作为系统消息的前缀重用。换句话说，如果模型支持缓存，则可以通过在系统消息开头放置静态内容来减少发送的令牌数量。

Agno 的上下文构建机制旨在将最有可能出现的静态内容置于系统消息的开头。如果您需要更精细的控制，可以通过手动设置系统消息来微调此机制。

对于团队而言，成员信息、委派指示和协调准则通常是静态的，因此非常适合缓存。

```python
from textwrap import dedent

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools

# Create specialized research agents
tech_researcher = Agent(
    name="Alex",
    role="Technology Researcher",
    instructions=dedent("""
        You specialize in technology and AI research.
        - Focus on latest developments, trends, and breakthroughs
        - Provide concise, data-driven insights
    """).strip(),
)

business_analyst = Agent(
    name="Sarah",
    role="Business Analyst",
    instructions=dedent("""
        You specialize in business and market analysis.
        - Focus on companies, markets, and economic trends
        - Provide actionable business insights
        - Include relevant data and statistics
    """).strip(),
)

# Create research team with tools and context management
research_team = Team(
    name="Research Team",
    model=OpenAIResponses(id="gpt-5.2"),
    members=[tech_researcher, business_analyst],
    tools=[HackerNewsTools(), YFinanceTools()],
    description="Research team that investigates topics and provides analysis.",
    instructions=dedent("""
        You are a research coordinator that investigates topics comprehensively.

        Your Process:
        1. Use HackerNews to find tech discussions and YFinance for market data
        2. Delegate detailed analysis to the appropriate specialist
        3. Synthesize research findings with specialist insights

        Guidelines:
        - Use HackerNews for tech news and YFinance for financial data
        - Choose the right specialist based on the topic (tech vs business)
        - Combine your research with specialist analysis
        - Provide comprehensive responses
    """).strip(),
    db=SqliteDb(db_file="tmp/research_team.db"),
    session_id="research_session",
    add_history_to_context=True,
    num_history_runs=6,  # Load last 6 research queries
    max_tool_calls_from_history=3,  # Keep only last 3 research results
    markdown=True,
)

research_team.print_response("What are the latest developments in AI agents?", stream=True)
research_team.print_response("How is NVDA performing this quarter?", stream=True)
research_team.print_response("What are the trends in LLM applications?", stream=True)
research_team.print_response("What companies are leading in AI infrastructure?", stream=True)
```


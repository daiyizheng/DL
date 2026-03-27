#  状态管理

在代理运行、团队协调和工作流程执行过程中持久化和共享数据

状态是指在会话中多次运行后仍然保留的数据，使代理、团队和工作流能够维护上下文并记住信息。

常见用例包括管理用户特定数据，例如购物清单、待办事项清单、偏好设置或任何需要在交互过程中保持持久化的信息。状态通过session_state可访问和更新的工具进行管理，然后自动持久化到数据库中。


## 状态如何运作
Agno 中的状态遵循以下模式：
- 初始化-session_state创建代理、团队或工作流时设置默认值
- 访问- 通过run_context.session_state工具访问状态
- 更新- 修改会自动保存到数据库。
- 加载- 同一会话中的后续运行会检索已存储的状态


### 基本示例
这是一个简单的代理程序，用于维护购物清单：

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.run import RunContext

def add_item(run_context: RunContext, item: str) -> str:
    """Add an item to the shopping list."""
    run_context.session_state["shopping_list"].append(item)
    return f"Added {item}"

agent = Agent(
    db=SqliteDb(db_file="tmp/state.db"),
    session_state={"shopping_list": []},  # Default state
    tools=[add_item],
    instructions="Shopping list: {shopping_list}",  # State in instructions
)

agent.print_response("Add milk and eggs")
print(agent.get_session_state())  # {'shopping_list': ['milk', 'eggs']}
```

## 代理会话状态
### 基本状态

本示例演示如何创建一个具有基本会话状态管理的代理，并使用 SQLite 存储来维护跨交互的购物清单。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses


def add_item(session_state, item: str) -> str:
    """Add an item to the shopping list."""
    session_state["shopping_list"].append(item)
    return f"The shopping list is now {session_state['shopping_list']}"


agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    session_state={"shopping_list": []},
    db=SqliteDb(db_file="tmp/agents.db"),
    tools=[add_item],
    instructions="Current state (shopping list) is: {shopping_list}",
    markdown=True,
)

agent.print_response("Add milk, eggs, and bread to the shopping list", stream=True)
print(f"Final session state: {agent.get_session_state()}")
```

### 指令中的状态

本示例演示了如何在代理指令中直接使用会话状态变量。它展示了如何初始化会话状态以及如何在指令模板中引用这些变量。

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    session_state={"user_name": "John"},
    instructions="Users name is {user_name}",
    markdown=True,
)

agent.print_response("What is my name?", stream=True)
```
### 上下文中的状态

本示例演示了如何使用会话状态以及跨不同会话管理用户上下文。它展示了会话状态如何持久化，以及如何针对不同用户和会话检索会话状态。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses

db = SqliteDb(db_file="tmp/agent.db")

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="Users name is {user_name} and age is {age}",
    db=db,
)

# Sets the session state for the session with the id "user_1_session_1"
agent.print_response(
    "What is my name?",
    session_id="user_1_session_1",
    user_id="user_1",
    session_state={"user_name": "John", "age": 30},
    stream=True,
)

# Will load the session state from the session with the id "user_1_session_1"
agent.print_response("How old am I?", session_id="user_1_session_1", user_id="user_1", stream=True)

# Sets the session state for the session with the id "user_2_session_1"
agent.print_response(
    "What is my name?",
    session_id="user_2_session_1",
    user_id="user_2",
    session_state={"user_name": "Jane", "age": 25},
    stream=True,
)

# Will load the session state from the session with the id "user_2_session_1"
agent.print_response("How old am I?", session_id="user_2_session_1", user_id="user_2", stream=True)
```

### 高级状态

本示例演示了使用多个工具进行高级会话状态管理，以管理购物清单，包括添加、删除和列表操作。

```python
from textwrap import dedent

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.run import RunContext


# Define tools to manage our shopping list
def add_item(run_context: RunContext, item: str) -> str:
    """Add an item to the shopping list and return confirmation."""
    if not run_context.session_state:
        run_context.session_state = {}

    # Add the item if it's not already in the list
    if item.lower() not in [i.lower() for i in run_context.session_state["shopping_list"]]:
        run_context.session_state["shopping_list"].append(item)  # type: ignore
        return f"Added '{item}' to the shopping list"
    else:
        return f"'{item}' is already in the shopping list"


def remove_item(run_context: RunContext, item: str) -> str:
    """Remove an item from the shopping list by name."""
    if not run_context.session_state:
        run_context.session_state = {}

    # Case-insensitive search
    for i, list_item in enumerate(run_context.session_state["shopping_list"]):
        if list_item.lower() == item.lower():
            run_context.session_state["shopping_list"].pop(i)
            return f"Removed '{list_item}' from the shopping list"

    return f"'{item}' was not found in the shopping list"


def list_items(run_context: RunContext) -> str:
    """List all items in the shopping list."""
    if not run_context.session_state:
        run_context.session_state = {}

    shopping_list = run_context.session_state["shopping_list"]

    if not shopping_list:
        return "The shopping list is empty."

    items_text = "\n".join([f"- {item}" for item in shopping_list])
    return f"Current shopping list:\n{items_text}"


# Create a Shopping List Manager Agent that maintains state
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    # Initialize the session state with an empty shopping list (default session state for all sessions)
    session_state={"shopping_list": []},
    db=SqliteDb(db_file="tmp/example.db"),
    tools=[add_item, remove_item, list_items],
    # You can use variables from the session state in the instructions
    instructions=dedent("""\
        Your job is to manage a shopping list.

        The shopping list starts empty. You can add items, remove items by name, and list all items.

        Current shopping list: {shopping_list}
    """),
    markdown=True,
)

# Example usage
agent.print_response("Add milk, eggs, and bread to the shopping list", stream=True)
print(f"Session state: {agent.get_session_state()}")

agent.print_response("I got bread", stream=True)
print(f"Session state: {agent.get_session_state()}")

agent.print_response("I need apples and oranges", stream=True)
print(f"Session state: {agent.get_session_state()}")

agent.print_response("whats on my list?", stream=True)
print(f"Session state: {agent.get_session_state()}")

agent.print_response(
    "Clear everything from my list and start over with just bananas and yogurt",
    stream=True,
)
print(f"Session state: {agent.get_session_state()}")
```
### 多用户
本示例演示了如何在多用户环境中为多个用户维护单独的会话状态，每个用户都有自己的购物清单和会话。

```python
"""
This example demonstrates how to maintain state for each user in a multi-user environment.

The shopping list is stored in a dictionary, organized by user ID and session ID.

Agno automatically creates the "current_user_id" and "current_session_id" variables in the session state.

You can access these variables in your functions using the `agent.get_session_state()` dictionary.
"""

import json

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.run.base import run_context

# In-memory database to store user shopping lists
# Organized by user ID and session ID
shopping_list = {}


def add_item(run_context: RunContext, item: str) -> str:
    """Add an item to the current user's shopping list."""

    if not run_context.session_state:
        run_context.session_state = {}

    current_user_id = run_context.session_state["current_user_id"]
    current_session_id = run_context.session_state["current_session_id"]
    shopping_list.setdefault(current_user_id, {}).setdefault(
        current_session_id, []
    ).append(item)

    return f"Item {item} added to the shopping list"


def remove_item(run_context: RunContext, item: str) -> str:
    """Remove an item from the current user's shopping list."""

    if not run_context.session_state:
        run_context.session_state = {}

    current_user_id = run_context.session_state["current_user_id"]
    current_session_id = run_context.session_state["current_session_id"]

    if (
        current_user_id not in shopping_list
        or current_session_id not in shopping_list[current_user_id]
    ):
        return f"No shopping list found for user {current_user_id} and session {current_session_id}"

    if item not in shopping_list[current_user_id][current_session_id]:
        return f"Item '{item}' not found in the shopping list for user {current_user_id} and session {current_session_id}"

    shopping_list[current_user_id][current_session_id].remove(item)
    return f"Item {item} removed from the shopping list"


def get_shopping_list(run_context: RunContext) -> str:
    """Get the current user's shopping list."""

    if not run_context.session_state:
        run_context.session_state = {}

    current_user_id = run_context.session_state["current_user_id"]
    current_session_id = run_context.session_state["current_session_id"]
    return f"Shopping list for user {current_user_id} and session {current_session_id}: \n{json.dumps(shopping_list[current_user_id][current_session_id], indent=2)}"


# Create an Agent that maintains state
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=SqliteDb(db_file="tmp/data.db"),
    tools=[add_item, remove_item, get_shopping_list],
    # Reference the in-memory database
    instructions=[
        "Current User ID: {current_user_id}",
        "Current Session ID: {current_session_id}",
    ],
    markdown=True,
)

user_id_1 = "john_doe"
user_id_2 = "mark_smith"
user_id_3 = "carmen_sandiago"

# Example usage
agent.print_response(
    "Add milk, eggs, and bread to the shopping list",
    stream=True,
    user_id=user_id_1,
    session_id="user_1_session_1",
)
agent.print_response(
    "Add tacos to the shopping list",
    stream=True,
    user_id=user_id_2,
    session_id="user_2_session_1",
)
agent.print_response(
    "Add apples and grapes to the shopping list",
    stream=True,
    user_id=user_id_3,
    session_id="user_3_session_1",
)
agent.print_response(
    "Remove milk from the shopping list",
    stream=True,
    user_id=user_id_1,
    session_id="user_1_session_1",
)
agent.print_response(
    "Add minced beef to the shopping list",
    stream=True,
    user_id=user_id_2,
    session_id="user_2_session_1",
)

# What is on Mark Smith's shopping list?
agent.print_response(
    "What is on Mark Smith's shopping list?",
    stream=True,
    user_id=user_id_2,
    session_id="user_2_session_1",
)

# New session, so new shopping list
agent.print_response(
    "Add chicken and soup to my list.",
    stream=True,
    user_id=user_id_2,
    session_id="user_3_session_2",
)

print(f"Final shopping lists: \n{json.dumps(shopping_list, indent=2)}")
```

### Agentic State
此示例演示了如何启用代理会话状态管理，使代理能够根据对话上下文自动更新和管理会话状态。代理可以根据用户交互修改购物清单。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses

db = SqliteDb(db_file="tmp/agents.db")
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    session_state={"shopping_list": []},
    add_session_state_to_context=True,
    enable_agentic_state=True,
)

agent.print_response("Add milk, eggs, and bread to the shopping list", stream=True)

agent.print_response("I picked up the eggs, now what's on my list?", stream=True)

print(f"Session state: {agent.get_session_state()}")
```

### 动态状态

本示例演示了如何使用工具钩子动态管理会话状态。它展示了如何创建一个客户管理系统，该系统通过工具交互而非直接修改来更新会话状态。

```python
import json
from typing import Any, Callable, Dict

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.openai import OpenAIResponses
from agno.tools.toolkit import Toolkit
from agno.utils.log import log_info, log_warning
from agno.run import RunContext


class CustomerDBTools(Toolkit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register(self.process_customer_request)

    def process_customer_request(
        self,
        agent: Agent,
        customer_id: str,
        action: str = "retrieve",
        name: str = "John Doe",
    ):
        log_warning("Tool called, this shouldn't happen.")
        return "This should not be seen."


def customer_management_hook(
    run_context: RunContext,
    function_name: str,
    function_call: Callable,
    arguments: Dict[str, Any],
):
    if not run_context.session_state:
        run_context.session_state = {}

    action = arguments.get("action", "retrieve")
    cust_id = arguments.get("customer_id")
    name = arguments.get("name", None)

    if not cust_id:
        raise ValueError("customer_id is required.")

    if action == "create":
        run_context.session_state["customer_profiles"][cust_id] = {"name": name}
        log_info(f"Hook: UPDATED session_state for customer '{cust_id}'.")
        return f"Success! Customer {cust_id} has been created."

    if action == "retrieve":
        profile = run_context.session_state.get("customer_profiles", {}).get(cust_id)
        if profile:
            log_info(f"Hook: FOUND customer '{cust_id}' in session_state.")
            return f"Profile for {cust_id}: {json.dumps(profile)}"
        else:
            raise ValueError(f"Customer '{cust_id}' not found.")

    log_info(f"Session state: {run_context.session_state}")


def run_test():
    agent = Agent(
        model=OpenAIResponses(id="gpt-5.2"),
        tools=[CustomerDBTools()],
        tool_hooks=[customer_management_hook],
        session_state={"customer_profiles": {"123": {"name": "Jane Doe"}}},
        instructions="Your profiles: {customer_profiles}. Use `process_customer_request`. Use either create or retrieve as action for the tool.",
        resolve_in_context=True,
        db=InMemoryDb(),
    )

    prompt = "First, create customer 789 named 'Tom'. Then, retrieve Tom's profile. Step by step."
    log_info(f"Prompting: '{prompt}'")
    agent.print_response(prompt, stream=False)

    log_info("\n--- TEST ANALYSIS ---")
    log_info(
        "Check logs for the second tool call. The system prompt will NOT contain customer '789'."
    )


if __name__ == "__main__":
    run_test()
```

### 运行中更改状态

此示例演示了如何管理不同用户在不同运行中的会话状态。它展示了会话状态如何在同一会话内保持持久性，但在不同的会话和用户之间相互隔离。


```python
from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.openai import OpenAIResponses

agent = Agent(
    db=InMemoryDb(),
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="Users name is {user_name} and age is {age}",
    debug_mode=True,
)

# Sets the session state for the session with the id "user_1_session_1"
agent.print_response(
    "What is my name?",
    session_id="user_1_session_1",
    user_id="user_1",
    session_state={"user_name": "John", "age": 30},
)

# Will load the session state from the session with the id "user_1_session_1"
agent.print_response("How old am I?", session_id="user_1_session_1", user_id="user_1")

# Sets the session state for the session with the id "user_2_session_1"
agent.print_response(
    "What is my name?",
    session_id="user_2_session_1",
    user_id="user_2",
    session_state={"user_name": "Jane", "age": 25},
)

# Will load the session state from the session with the id "user_2_session_1"
agent.print_response("How old am I?", session_id="user_2_session_1", user_id="user_2")
```

### 最后 N 条消息

此示例演示了如何配置座席以搜索历史会话并限制上下文中包含的历史会话数量。这有助于在保持相关对话历史记录的同时，控制上下文长度。

```python
import os

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses

# Remove the tmp db file before running the script
os.remove("tmp/data.db")

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    user_id="user_1",
    db=SqliteDb(db_file="tmp/data.db"),
    add_history_to_context=True,
    num_history_runs=3,
    search_session_history=True,  # allow searching previous sessions
    num_history_sessions=2,  # only include the last 2 sessions in the search to avoid context length issues
)

session_1_id = "session_1_id"
session_2_id = "session_2_id"
session_3_id = "session_3_id"
session_4_id = "session_4_id"
session_5_id = "session_5_id"

agent.print_response("What is the capital of South Africa?", session_id=session_1_id)
agent.print_response("What is the capital of China?", session_id=session_2_id)
agent.print_response("What is the capital of France?", session_id=session_3_id)
agent.print_response("What is the capital of Japan?", session_id=session_4_id)
agent.print_response(
    "What did I discuss in my previous conversations?", session_id=session_5_id
)  # It should only include the last 2 sessions
```


## 团队会话状态

在团队中共享和协调多个代理的状态

团队会话状态功能支持代理团队之间共享和更新状态数据。团队通常需要就共享信息进行协调。

### 如何使用共享状态
你可以在Team上设置session_state参数来设置初始会话状态数据。这些状态数据将在团队领导和成员之间共享。

所有团队成员均可访问此状态，并且该状态在他们之间同步。

例如：
```python
team = Team(
    members=[agent1, agent2, agent3],
    session_state={"shopping_list": []},
)
```

成员可以使用工具中的 `run_context.session_state` 访问共享状态。

```python
from agno.run import RunContext

def add_item(run_context: RunContext, item: str) -> str:
    """Add an item to the shopping list and return confirmation.

    Args:
        item (str): The item to add to the shopping list.
    """
    # Add the item if it's not already in the list
    if item.lower() not in [
        i.lower() for i in run_context.session_state["shopping_list"]
    ]:
        run_context.session_state["shopping_list"].append(item)
        return f"Added '{item}' to the shopping list"
    else:
        return f"'{item}' is already in the shopping list"
```


### 例子
以下是一个团队管理共享购物清单的简单示例：

```python
from agno.models.openai import OpenAIResponses
from agno.agent import Agent
from agno.team import Team
from agno.run import RunContext


# Define tools that work with shared team state
def add_item(run_context: RunContext, item: str) -> str:
    """Add an item to the shopping list."""
    if not run_context.session_state:
        run_context.session_state = {}

    if item.lower() not in [
        i.lower() for i in run_context.session_state["shopping_list"]
    ]:
        run_context.session_state["shopping_list"].append(item)
        return f"Added '{item}' to the shopping list"
    else:
        return f"'{item}' is already in the shopping list"


def remove_item(run_context: RunContext, item: str) -> str:
    """Remove an item from the shopping list."""
    if not run_context.session_state:
        run_context.session_state = {}

    for i, list_item in enumerate(run_context.session_state["shopping_list"]):
        if list_item.lower() == item.lower():
            run_context.session_state["shopping_list"].pop(i)
            return f"Removed '{list_item}' from the shopping list"

    return f"'{item}' was not found in the shopping list"


# Create an agent that manages the shopping list
shopping_agent = Agent(
    name="Shopping List Agent",
    role="Manage the shopping list",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[add_item, remove_item],
)


# Define team-level tools
def list_items(run_context: RunContext) -> str:
    """List all items in the shopping list."""
    if not run_context.session_state:
        run_context.session_state = {}

    # Access shared state (not private state)
    shopping_list = run_context.session_state["shopping_list"]

    if not shopping_list:
        return "The shopping list is empty."

    items_text = "\n".join([f"- {item}" for item in shopping_list])
    return f"Current shopping list:\n{items_text}"


def add_chore(run_context: RunContext, chore: str) -> str:
    """Add a completed chore to the team's private log."""
    if not run_context.session_state:
        run_context.session_state = {}

    # Access team's private state
    if "chores" not in run_context.session_state:
        run_context.session_state["chores"] = []

    run_context.session_state["chores"].append(chore)
    return f"Logged chore: {chore}"


# Create a team with both shared and private state
shopping_team = Team(
    name="Shopping Team",
    model=OpenAIResponses(id="gpt-5.2"),
    members=[shopping_agent],
    session_state={"shopping_list": [], "chores": []},
    tools=[list_items, add_chore],
    instructions=[
        "You manage a shopping list.",
        "Forward add/remove requests to the Shopping List Agent.",
        "Use list_items to show the current list.",
        "Log completed tasks using add_chore.",
    ],
)

# Example usage
shopping_team.print_response("Add milk, eggs, and bread", stream=True)
print(f"Shared state: {shopping_team.get_session_state()}")

shopping_team.print_response("What's on my list?", stream=True)

shopping_team.print_response("I got the eggs", stream=True)
print(f"Shared state: {shopping_team.get_session_state()}")
```


### 在指令中使用状态
您可以在指令中引用会话状态中的变量。

```python
from agno.team.team import Team

team = Team(
    members=[],
    # Initialize the session state with a variable
    session_state={"user_name": "John"},
    instructions="Users name is {user_name}",
    markdown=True,
)

team.print_response("What is my name?", stream=True)
```

### 运行中状态的改变
当你在team.run（）上把session_id传递给团队时，它会切换到给定session_id的会话，并加载该会话中设置的状态。

```python
from agno.team.team import Team
from agno.models.openai import OpenAIResponses
from agno.db.in_memory import InMemoryDb

team = Team(
    db=InMemoryDb(),
    model=OpenAIResponses(id="gpt-5.2"),
    members=[],
    instructions="Users name is {user_name} and age is {age}",
)

# Sets the session state for the session with the id "user_1_session_1"
team.print_response("What is my name?", session_id="user_1_session_1", user_id="user_1", session_state={"user_name": "John", "age": 30})

# Will load the session state from the session with the id "user_1_session_1"
team.print_response("How old am I?", session_id="user_1_session_1", user_id="user_1")

# Sets the session state for the session with the id "user_2_session_1"
team.print_response("What is my name?", session_id="user_2_session_1", user_id="user_2", session_state={"user_name": "Jane", "age": 25})

# Will load the session state from the session with the id "user_2_session_1"
team.print_response("How old am I?", session_id="user_2_session_1", user_id="user_2")
```

### 覆盖数据库中的状态
默认情况下，如果将此新状态传递session_state给运行方法，则会将其与session_state数据库中的状态合并。

如果您想覆盖session_state数据库中的数据，可以更改该行为：
```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses

# Create an Agent that maintains state
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=SqliteDb(db_file="tmp/agents.db"),
    markdown=True,
    # Set the default session_state. The values set here won't be overwritten.
    session_state={},
    # Adding the session_state to context for the agent to easily access it
    add_session_state_to_context=True,
    # Allow overwriting the stored session state with the session state provided in the run
    overwrite_db_session_state=True,
)

# Let's run the agent providing a session_state. This session_state will be stored in the database.
agent.print_response(
    "Can you tell me what's in your session_state?",
    session_state={"shopping_list": ["Potatoes"]},
    stream=True,
)
print(f"Stored session state: {agent.get_session_state()}")

# Now if we pass a new session_state, it will overwrite the stored session_state.
agent.print_response(
    "Can you tell me what is in your session_state?",
    session_state={"secret_number": 43},
    stream=True,
)
print(f"Stored session state: {agent.get_session_state()}")
```

### 团队成员互动
智能体团队可以共享成员之间的交互，使智能体能够从彼此的输出中学习：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.team.team import Team

from agno.db.sqlite import SqliteDb
from agno.tools.duckduckgo import DuckDuckGoTools

db = SqliteDb(db_file="tmp/agents.db")

web_research_agent = Agent(
    name="Web Research Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[DuckDuckGoTools()],
    instructions="You are a web research agent that can answer questions from the web.",
)

report_agent = Agent(
    name="Report Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="You are a report agent that can write a report from the web research.",
)

team = Team(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    members=[web_research_agent, report_agent],
    share_member_interactions=True,
    instructions=[
        "You are a team of agents that can research the web and write a report.",
        "First, research the web for information about the topic.",
        "Then, use your report agent to write a report from the web research.",
    ],
    show_members_responses=True,
    debug_mode=True,
)

team.print_response("How are LEDs made?")
```


## 工作流会话状态

协调工作流步骤、代理、团队和自定义函数之间的状态

工作流会话状态支持在工作流中的所有组件（代理、团队和自定义函数）之间共享和更新状态数据。

如果数据库可用，会话状态数据将被持久化，并在后续工作流运行中从该数据库加载。


<img src="https://mintcdn.com/agno-v2/JYIBgMrzFEujZh3_/images/workflows-session-state-light.png?w=2500&fit=max&auto=format&n=JYIBgMrzFEujZh3_&q=85&s=c5d4f68f953670a714b4a5d334f9b5fb">

### 工作流会话状态的工作原理
​
1. 状态初始化
创建工作流时初始化会话状态。会话状态可以初始为空，也可以包含所有工作流组件均可访问和修改的预定义数据。

```python
shopping_workflow = Workflow(
    name="Shopping List Workflow",
    steps=[manage_items_step, view_list_step],
    session_state={"shopping_list": []},  # Initialize with structured data
)
```

2. 状态持久化

所有工作流组件（包括代理、团队和功能）都可以读取和写入共享会话状态。这实现了整个工作流执行过程中的持久数据流和协调。
通过工具，您可以通过以下run_context.session_state方式访问会话状态。
示例：购物清单管理

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.workflow.step import Step
from agno.workflow.workflow import Workflow
from agno.run import RunContext

db = SqliteDb(db_file="tmp/workflow.db")


# Define tools to manage a shopping list in workflow session state
def add_item(run_context: RunContext, item: str) -> str:
    """Add an item to the shopping list in workflow session state.

    Args:
        item (str): The item to add to the shopping list
    """
    if not run_context.session_state:
        run_context.session_state = {}

    # Check if item already exists (case-insensitive)
    existing_items = [
        existing_item.lower() for existing_item in run_context.session_state["shopping_list"]
    ]
    if item.lower() not in existing_items:
        run_context.session_state["shopping_list"].append(item)
        return f"Added '{item}' to the shopping list."
    else:
        return f"'{item}' is already in the shopping list."


def remove_item(run_context: RunContext, item: str) -> str:
    """Remove an item from the shopping list in workflow session state.

    Args:
        item (str): The item to remove from the shopping list
    """
    if not run_context.session_state:
        run_context.session_state = {}

    if len(run_context.session_state["shopping_list"]) == 0:
        return f"Shopping list is empty. Cannot remove '{item}'."

    # Find and remove item (case-insensitive)
    shopping_list = run_context.session_state["shopping_list"]
    for i, existing_item in enumerate(shopping_list):
        if existing_item.lower() == item.lower():
            removed_item = shopping_list.pop(i)
            return f"Removed '{removed_item}' from the shopping list."

    return f"'{item}' not found in the shopping list."


def remove_all_items(run_context: RunContext) -> str:
    """Remove all items from the shopping list in workflow session state."""
    if not run_context.session_state:
        run_context.session_state = {}

    run_context.session_state["shopping_list"] = []
    return "Removed all items from the shopping list."


def list_items(run_context: RunContext) -> str:
    """List all items in the shopping list from workflow session state."""
    if not run_context.session_state:
        run_context.session_state = {}

    if len(run_context.session_state["shopping_list"]) == 0:
        return "Shopping list is empty."

    items = run_context.session_state["shopping_list"]
    items_str = "\n".join([f"- {item}" for item in items])
    return f"Shopping list:\n{items_str}"


# Create agents with tools that use workflow session state
shopping_assistant = Agent(
    name="Shopping Assistant",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[add_item, remove_item, list_items],
    instructions=[
        "You are a helpful shopping assistant.",
        "You can help users manage their shopping list by adding, removing, and listing items.",
        "Always use the provided tools to interact with the shopping list.",
        "Be friendly and helpful in your responses.",
    ],
)

list_manager = Agent(
    name="List Manager",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[list_items, remove_all_items],
    instructions=[
        "You are a list management specialist.",
        "You can view the current shopping list and clear it when needed.",
        "Always show the current list when asked.",
        "Confirm actions clearly to the user.",
    ],
)

# Create steps
manage_items_step = Step(
    name="manage_items",
    description="Help manage shopping list items (add/remove)",
    agent=shopping_assistant,
)

view_list_step = Step(
    name="view_list",
    description="View and manage the complete shopping list",
    agent=list_manager,
)

# Create workflow with workflow_session_state
shopping_workflow = Workflow(
    name="Shopping List Workflow",
    db=db,
    steps=[manage_items_step, view_list_step],
    session_state={"shopping_list": []},
)

if __name__ == "__main__":
    # Example 1: Add items to the shopping list
    print("=== Example 1: Adding Items ===")
    shopping_workflow.print_response(
        input="Please add milk, bread, and eggs to my shopping list."
    )
    print("Workflow session state:", shopping_workflow.get_session_state())

    # Example 2: Add more items and view list
    print("\n=== Example 2: Adding More Items ===")
    shopping_workflow.print_response(
        input="Add apples and bananas to the list, then show me the complete list."
    )
    print("Workflow session state:", shopping_workflow.get_session_state())

    # Example 3: Remove items
    print("\n=== Example 3: Removing Items ===")
    shopping_workflow.print_response(
        input="Remove bread from the list and show me what's left."
    )
    print("Workflow session state:", shopping_workflow.get_session_state())

    # Example 4: Clear the entire list
    print("\n=== Example 4: Clearing List ===")
    shopping_workflow.print_response(
        input="Clear the entire shopping list and confirm it's empty."
    )
    print("Final workflow session state:", shopping_workflow.get_session_state())
```



3. run_context作为工作流程中自定义 Python 函数步骤的参数

你可以把run_context参数添加到你用的Python函数中，作为自定义步骤。

运行函数时，run_context对象会自动注入。

你可以用它来读取和修改会话状态，通过 run_context.session_state。

```python
from agno.run import RunContext

def custom_function_step(step_input: StepInput, run_context: RunContext):
    """Update the workflow session state"""
    run_context.session_state["test"] = test_1
```


该run_context也可以作为参数出现在条件和路由器步骤的评估器和选择函数中：

```python
from agno.run import RunContext

def evaluator_function(step_input: StepInput, run_context: RunContext):
    return run_context.session_state["test"] == "test_1"

condition_step = Condition(
    name="condition_step",
    evaluator=evaluator_function,
    steps=[step_1, step_2],
)
```

```python
from agno.run import RunContext

def selector_function(step_input: StepInput, run_context: RunContext):
    return run_context.session_state["test"] == "test_1"

router_step = Router(
    name="router_step",
    selector=selector_function,
    choices=[step_1, step_2],
)
```

### 自定义函数中的状态

此示例演示如何在自定义 Python 函数步骤中访问运行上下文。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.run import RunContext
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.workflow.step import Step, StepInput, StepOutput
from agno.workflow.workflow import Workflow

# Define agents
hackernews_agent = Agent(
    name="Hackernews Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
    instructions="Extract key insights and content from Hackernews posts",
)

finance_agent = Agent(
    name="Finance Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[YFinanceTools()],
    instructions="Get financial data and market trends",
)

# Define research team for complex analysis
research_team = Team(
    name="Research Team",
    model=OpenAIResponses(id="gpt-5.2"),
    members=[hackernews_agent, finance_agent],
    instructions="Analyze content and create comprehensive social media strategy",
)

content_planner = Agent(
    name="Content Planner",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=[
        "Plan a content schedule over 4 weeks for the provided topic and research content",
        "Ensure that I have posts for 3 posts per week",
    ],
)


def custom_content_planning_function(
    step_input: StepInput, run_context: RunContext
) -> StepOutput:
    """
    Custom function that does intelligent content planning with context awareness
    and maintains a content plan history in session_state
    """
    message = step_input.input
    previous_step_content = step_input.previous_step_content

    # Initialize content history if not present
    if "content_plans" not in run_context.session_state:
        run_context.session_state["content_plans"] = []

    if "plan_counter" not in run_context.session_state:
        run_context.session_state["plan_counter"] = 0

    # Increment plan counter
    run_context.session_state["plan_counter"] += 1
    current_plan_id = run_context.session_state["plan_counter"]

    # Create intelligent planning prompt
    planning_prompt = f"""
        STRATEGIC CONTENT PLANNING REQUEST:

        Core Topic: {message}
        Plan ID: #{current_plan_id}

        Research Results: {previous_step_content[:500] if previous_step_content else "No research results"}

        Previous Plans Count: {len(run_context.session_state["content_plans"])}

        Planning Requirements:
        1. Create a comprehensive content strategy based on the research
        2. Leverage the research findings effectively
        3. Identify content formats and channels
        4. Provide timeline and priority recommendations
        5. Include engagement and distribution strategies

        Please create a detailed, actionable content plan.
    """

    try:
        response = content_planner.run(planning_prompt)

        # Store this plan in session state
        plan_data = {
            "id": current_plan_id,
            "topic": message,
            "content": response.content,
            "timestamp": f"Plan #{current_plan_id}",
            "has_research": bool(previous_step_content),
        }
        run_context.session_state["content_plans"].append(plan_data)

        enhanced_content = f"""
            ## Strategic Content Plan #{current_plan_id}

            **Planning Topic:** {message}

            **Research Integration:** {"Research-based" if previous_step_content else "No research foundation"}
            **Total Plans Created:** {len(run_context.session_state["content_plans"])}

            **Content Strategy:**
            {response.content}

            **Custom Planning Enhancements:**
            - Research Integration: {"High" if previous_step_content else "Baseline"}
            - Strategic Alignment: Optimized for multi-channel distribution
            - Execution Ready: Detailed action items included
            - Session History: {len(run_context.session_state["content_plans"])} plans stored

            **Plan ID:** #{current_plan_id}
        """.strip()

        return StepOutput(content=enhanced_content)

    except Exception as e:
        return StepOutput(
            content=f"Custom content planning failed: {str(e)}",
            success=False,
        )


def content_summary_function(step_input: StepInput, run_context: RunContext) -> StepOutput:
    """
    Custom function that summarizes all content plans created in the session
    """
    if run_context.session_state is None or run_context.session_state.get("content_plans") is None:
        return StepOutput(
            content="No content plans found in session state.", success=False
        )

    plans = run_context.session_state["content_plans"]
    summary = f"""
        ## Content Planning Session Summary

        **Total Plans Created:** {len(plans)}
        **Session Statistics:**
        - Plans with research: {len([p for p in plans if p["has_research"]])}
        - Plans without research: {len([p for p in plans if not p["has_research"]])}

        **Plan Overview:**
    """

    for plan in plans:
        summary += f"""

        ### Plan #{plan["id"]} - {plan["topic"]}
        - Research Available: {"Yes" if plan["has_research"] else "No"}
        - Status: Completed
        """

    # Update session state with summary info
    run_context.session_state["session_summarized"] = True
    run_context.session_state["total_plans_summarized"] = len(plans)

    return StepOutput(content=summary.strip())


# Define steps using different executor types

research_step = Step(
    name="Research Step",
    team=research_team,
)

content_planning_step = Step(
    name="Content Planning Step",
    executor=custom_content_planning_function,
)

content_summary_step = Step(
    name="Content Summary Step",
    executor=content_summary_function,
)


# Define and use examples
if __name__ == "__main__":
    content_creation_workflow = Workflow(
        name="Content Creation Workflow",
        description="Automated content creation with custom execution options and session state",
        db=SqliteDb(
            session_table="workflow_session",
            db_file="tmp/workflow.db",
        ),
        # Define the sequence of steps
        # First run the research_step, then the content_planning_step, then the summary_step
        # You can mix and match agents, teams, and even regular python functions directly as steps
        steps=[research_step, content_planning_step, content_summary_step],
        # Initialize session state with empty content plans
        session_state={"content_plans": [], "plan_counter": 0},
    )

    print("=== First Workflow Run ===")
    content_creation_workflow.print_response(
        input="AI trends in 2024",
        markdown=True,
    )

    print(
        f"\nSession State After First Run: {content_creation_workflow.get_session_state()}"
    )

    print("\n" + "=" * 60 + "\n")

    print("=== Second Workflow Run (Same Session) ===")
    content_creation_workflow.print_response(
        input="Machine Learning automation tools",
        markdown=True,
    )

    print(f"\nFinal Session State: {content_creation_workflow.get_session_state()}")
```


### 处于状态
此示例演示如何在条件步骤的评估器函数中访问运行上下文。

这个例子表明：

1. 如何run_context在条件评估器函数中使用
2. run_context.session_state基于条件逻辑的读取和修改
3. 访问user_id和session_id来自run_context.session_state
4. 基于以下条件做出决策run_context.session_state

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.workflow.condition import Condition
from agno.workflow.step import Step, StepInput, StepOutput
from agno.workflow.workflow import Workflow
from agno.run import RunContext


def check_user_has_context(step_input: StepInput, run_context: RunContext) -> bool:
    """
    Condition evaluator that checks if user has been greeted before.

    Args:
        step_input: The input for this step (contains workflow context)
        run_context: The run context object

    Returns:
        bool: True if user has context, False otherwise
    """
    print("\n=== Evaluating Condition ===")
    print(f"User ID: {run_context.session_state.get('current_user_id')}")
    print(f"Session ID: {run_context.session_state.get('current_session_id')}")
    print(f"Has been greeted: {run_context.session_state.get('has_been_greeted', False)}")

    # Check if user has been greeted before
    return run_context.session_state.get("has_been_greeted", False)


def mark_user_as_greeted(step_input: StepInput, run_context: RunContext) -> StepOutput:
    """Custom function that marks user as greeted in session state."""
    print("\n=== Marking User as Greeted ===")
    run_context.session_state["has_been_greeted"] = True
    run_context.session_state["greeting_count"] = run_context.session_state.get("greeting_count", 0) + 1

    return StepOutput(
        content=f"User has been greeted. Total greetings: {run_context.session_state['greeting_count']}"
    )


# Create agents
greeter_agent = Agent(
    name="Greeter",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="Greet the user warmly and introduce yourself.",
    markdown=True,
)

contextual_agent = Agent(
    name="Contextual Assistant",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="Continue the conversation with context. You already know the user.",
    markdown=True,
)

# Create workflow with condition
workflow = Workflow(
    name="Conditional Greeting Workflow",
    steps=[
        # First, check if user has been greeted before
        Condition(
            name="Check If New User",
            description="Check if this is a new user who needs greeting",
            # Condition returns True if user has context, so we negate it
            evaluator=lambda step_input, run_context: not check_user_has_context(
                step_input, run_context
            ),
            steps=[
                # Only execute these steps for new users
                Step(
                    name="Greet User",
                    description="Greet the new user",
                    agent=greeter_agent,
                ),
                Step(
                    name="Mark as Greeted",
                    description="Mark user as greeted in session",
                    executor=mark_user_as_greeted,
                ),
            ],
        ),
        # This step always executes
        Step(
            name="Handle Query",
            description="Handle the user's query with or without greeting",
            agent=contextual_agent,
        ),
    ],
    session_state={
        "has_been_greeted": False,
        "greeting_count": 0,
    },
)


def run_example():
    """Run the example workflow multiple times to see conditional behavior."""

    print("=" * 80)
    print("First Run - New User (Condition will be True, greeting will happen)")
    print("=" * 80)

    workflow.print_response(
        input="Hi, can you help me with something?",
        session_id="user-123",
        user_id="user-123",
        stream=True,
    )

    print("\n" + "=" * 80)
    print("Second Run - Same Session (Skips greeting)")
    print("=" * 80)

    workflow.print_response(
        input="Tell me a joke",
        session_id="user-123",
        user_id="user-123",
        stream=True,
    )


if __name__ == "__main__":
    run_example()
```

### 路由器中的状态

此示例演示如何在路由步骤的选择器函数中访问运行上下文。

这个例子表明：

1. run_context.session_state在路由器选择器函数中使用
2. 基于会话状态数据做出路由决策
3. 从用户偏好设置和历史记录中访问用户run_context.session_state
4. 根据用户上下文动态选择不同的代理

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.workflow.router import Router
from agno.workflow.step import Step, StepInput, StepOutput
from agno.workflow.workflow import Workflow
from agno.run import RunContext


def route_based_on_user_preference(step_input: StepInput, run_context: RunContext) -> Step:
    """
    Router selector that chooses an agent based on user preferences in session_state.

    Args:
        step_input: The input for this step (contains user query)
        run_context: The run context object

    Returns:
        Step: The step to execute based on user preference
    """
    print("\n=== Routing Decision ===")
    print(f"User ID: {run_context.session_state.get('current_user_id')}")
    print(f"Session ID: {run_context.session_state.get('current_session_id')}")

    # Get user preference from session state
    user_preference = run_context.session_state.get("agent_preference", "general")
    interaction_count = run_context.session_state.get("interaction_count", 0)

    print(f"User Preference: {user_preference}")
    print(f"Interaction Count: {interaction_count}")

    # Update interaction count
    run_context.session_state["interaction_count"] = interaction_count + 1

    # Route based on preference
    if user_preference == "technical":
        print("→ Routing to Technical Expert")
        return technical_step
    elif user_preference == "friendly":
        print("→ Routing to Friendly Assistant")
        return friendly_step
    else:
        # For first interaction, route to onboarding
        if interaction_count == 0:
            print("→ Routing to Onboarding (first interaction)")
            return onboarding_step
        else:
            print("→ Routing to General Assistant")
            return general_step


def set_user_preference(step_input: StepInput, run_context: RunContext) -> StepOutput:
    """Custom function that sets user preference based on onboarding."""
    print("\n=== Setting User Preference ===")

    # In a real scenario, this would analyze the user's response
    # For demo purposes, we'll set it based on interaction count
    interaction_count = run_context.session_state.get("interaction_count", 0)

    if interaction_count % 3 == 1:
        run_context.session_state["agent_preference"] = "technical"
        preference = "technical"
    elif interaction_count % 3 == 2:
        run_context.session_state["agent_preference"] = "friendly"
        preference = "friendly"
    else:
        run_context.session_state["agent_preference"] = "general"
        preference = "general"

    print(f"Set preference to: {preference}")
    return StepOutput(content=f"Preference set to: {preference}")


# Create specialized agents
onboarding_agent = Agent(
    name="Onboarding Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=(
        "Welcome new users and ask about their preferences. "
        "Determine if they prefer technical or friendly assistance."
    ),
    markdown=True,
)

technical_agent = Agent(
    name="Technical Expert",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=(
        "You are a technical expert. Provide detailed, technical answers with code examples and best practices."
    ),
    markdown=True,
)

friendly_agent = Agent(
    name="Friendly Assistant",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=(
        "You are a friendly, casual assistant. Use simple language, emojis, and make the conversation fun."
    ),
    markdown=True,
)

general_agent = Agent(
    name="General Assistant",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=(
        "You are a balanced assistant. Provide helpful answers that are neither too technical nor too casual."
    ),
    markdown=True,
)

# Create steps for routing
onboarding_step = Step(
    name="Onboard User",
    description="Onboard new user and set preferences",
    agent=onboarding_agent,
)

technical_step = Step(
    name="Technical Response",
    description="Provide technical assistance",
    agent=technical_agent,
)

friendly_step = Step(
    name="Friendly Response",
    description="Provide friendly assistance",
    agent=friendly_agent,
)

general_step = Step(
    name="General Response",
    description="Provide general assistance",
    agent=general_agent,
)

# Create workflow with router
workflow = Workflow(
    name="Adaptive Assistant Workflow",
    steps=[
        # Router that selects agent based on session state
        Router(
            name="Route to Appropriate Agent",
            description="Route to the appropriate agent based on user preferences",
            selector=route_based_on_user_preference,
            choices=[
                onboarding_step,
                technical_step,
                friendly_step,
                general_step,
            ],
        ),
        # After first interaction, update preferences
        Step(
            name="Update Preferences",
            description="Update user preferences based on interaction",
            executor=set_user_preference,
        ),
    ],
    session_state={
        "agent_preference": "general",
        "interaction_count": 0,
    },
)


def run_example():
    """Run the example workflow multiple times to see dynamic routing."""

    queries = [
        "Hello! I'm new here.",
        "How do I implement a binary search tree in Python?",
        "What's the best pizza topping?",
        "Explain quantum computing",
    ]

    for i, query in enumerate(queries, 1):
        print("\n" + "=" * 80)
        print(f"Interaction {i}: {query}")
        print("=" * 80)

        workflow.print_response(
            input=query,
            session_id="user-456",
            user_id="user-456",
            stream=True,
        )


if __name__ == "__main__":
    run_example()
```


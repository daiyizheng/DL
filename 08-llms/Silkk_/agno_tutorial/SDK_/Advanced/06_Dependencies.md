# 依赖关系

将变量及其依赖关系注入到代理和团队上下文中。


依赖项是一种将变量注入到代理或团队上下文中的方法。该dependencies参数接受一个字典，其中包含函数或静态变量，这些函数或变量会在代理或团队运行之前自动解析。


## 基本用法
您可以在代理指令或用户消息中引用依赖项。

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    dependencies={"name": "John Doe"},
    instructions="You are a story writer. The current user is {name}."
)

agent.print_response("Write a 5 second short story about {name}")
```

### 依赖关系如何运作
依赖关系在运行时解决，就在您的代理或团队执行之前。流程如下：
- 定义依赖关系：提供一个键值对字典，其中值可以是静态数据或可调用函数。
- 解决方法：当代理/团队运行时，Agno 会调用所有可调用依赖项，并将其替换为它们的返回值。
- 模板替换：已解析的依赖项可通过{dependency_name}语法在您的指令中使用。
- 上下文注入：启用后add_dependencies_to_context=True，依赖项将自动添加到用户消息中。

## 与代理的依赖关系

将具有依赖关系的变量注入到代理上下文中。

依赖项是一种将变量注入到代理上下文中的方法。该dependencies参数接受一个字典，其中包含函数或静态变量，这些函数或变量会在代理运行之前自动解析。

### 基本用法
您可以在代理指令或用户消息中引用依赖项。

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    dependencies={"name": "John Doe"},
    instructions="You are a story writer. The current user is {name}."
)

agent.print_response("Write a 5 second short story about {name}")
```

### 向上下文添加依赖项
设置add_dependencies_to_context=True为将所有依赖项列表添加到用户消息中。这样您就无需手动将依赖项添加到说明中。

```python
import json

from agno.agent import Agent
from agno.models.openai import OpenAIResponses


def get_user_profile() -> str:
    """Fetch and return the user profile.

    Returns:
        JSON string containing user profile information
    """
    # Get the user profile from the database (this is a placeholder)
    user_profile = {
        "name": "John Doe",
        "experience_level": "senior",
    }

    return json.dumps(user_profile, indent=4)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    dependencies={"user_profile": get_user_profile},
    # We can add the entire dependencies dictionary to the user message
    add_dependencies_to_context=True,
    markdown=True,
)

agent.print_response(
    "Get the user profile and tell me about their experience level.",
    stream=True,
)
# Optionally pass the dependencies to the print_response method
# agent.print_response(
#     "Get the user profile and tell me about their experience level.",
#     dependencies={"user_profile": get_user_profile},
#     stream=True,
# )
```

### 在工具调用和钩子中访问依赖项
您可以使用该RunContext对象访问工具调用和钩子中的依赖项。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.run import RunContext

def get_user_profile(run_context: RunContext) -> str:
    """Get the user profile."""
    return run_context.dependencies["user_profiles"][run_context.user_id]

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=SqliteDb(db_file="tmp/agents.db"),
    tools=[get_user_profile],
    dependencies={
        "user_profiles": {
            "user_1001": {"name": "John Doe", "experience_level": "senior"},
            "user_1002": {"name": "Jane Doe", "experience_level": "junior"},
        }
    },
)

agent.print_response("Get the user profile for the current user and tell me about their experience level.", user_id="user_1001", stream=True)
```

## 向代理运行添加依赖项

此示例演示了如何将依赖项注入代理运行，从而允许代理访问动态上下文，例如用户配置文件和当前时间信息，以提供个性化响应。

```python
from datetime import datetime

from agno.agent import Agent
from agno.models.openai import OpenAIResponses


def get_user_profile(user_id: str = "john_doe") -> dict:
    """Get user profile information that can be referenced in responses.

    Args:
        user_id: The user ID to get profile for
    Returns:
        Dictionary containing user profile information
    """
    profiles = {
        "john_doe": {
            "name": "John Doe",
            "preferences": {
                "communication_style": "professional",
                "topics_of_interest": ["AI/ML", "Software Engineering", "Finance"],
                "experience_level": "senior",
            },
            "location": "San Francisco, CA",
            "role": "Senior Software Engineer",
        }
    }

    return profiles.get(user_id, {"name": "Unknown User"})


def get_current_context() -> dict:
    """Get current contextual information like time, weather, etc."""
    return {
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "PST",
        "day_of_week": datetime.now().strftime("%A"),
    }


agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    markdown=True,
)

response = agent.run(
    "Please provide me with a personalized summary of today's priorities based on my profile and interests.",
    dependencies={
        "user_profile": get_user_profile,
        "current_context": get_current_context,
    },
    add_dependencies_to_context=True,
    debug_mode=True,
)

print(response.content)
```

### 向代理上下文添加依赖项

本示例演示了如何创建一个上下文感知代理，该代理可以通过依赖注入访问实时 HackerNews 数据，从而使代理能够提供最新信息。

```python
import json

import httpx

from agno.agent import Agent
from agno.models.openai import OpenAIResponses


def get_top_hackernews_stories(num_stories: int = 5) -> str:
    """Fetch and return the top stories from HackerNews.

    Args:
        num_stories: Number of top stories to retrieve (default: 5)
    Returns:
        JSON string containing story details (title, url, score, etc.)
    """
    stories = [
        {
            k: v
            for k, v in httpx.get(
                f"https://hacker-news.firebaseio.com/v0/item/{id}.json"
            )
            .json()
            .items()
            if k != "kids"
        }
        for id in httpx.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json"
        ).json()[:num_stories]
    ]
    return json.dumps(stories, indent=4)


agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    dependencies={"top_hackernews_stories": get_top_hackernews_stories},
    add_dependencies_to_context=True,
    markdown=True,
)

agent.print_response(
    "Summarize the top stories on HackerNews and identify any interesting trends.",
    stream=True,
)
```

### 访问工具中的依赖关系

此示例演示了工具如何访问传递给代理的依赖项，从而使工具能够利用动态上下文（例如用户配置文件和当前时间信息）来增强功能。

```python
from typing import Dict, Any, Optional
from datetime import datetime

from agno.agent import Agent
from agno.models.openai import OpenAIResponses


def get_current_context() -> dict:
    """Get current contextual information like time, weather, etc."""
    return {
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "PST",
        "day_of_week": datetime.now().strftime("%A"),
    }

def analyze_user(user_id: str, dependencies: Optional[Dict[str, Any]] = None) -> str:
    """
    Analyze a specific user's profile and provide insights.

    This tool analyzes user behavior and preferences using available data sources.
    Call this tool with the user_id you want to analyze.

    Args:
        user_id: The user ID to analyze (e.g., 'john_doe', 'jane_smith')
        dependencies: Available data sources (automatically provided)

    Returns:
        Detailed analysis and insights about the user
    """
    if not dependencies:
        return "No data sources available for analysis."

    print(f"--> Tool received data sources: {list(dependencies.keys())}")

    results = [f"=== USER ANALYSIS FOR {user_id.upper()} ==="]

    if "user_profile" in dependencies:
        profile_data = dependencies["user_profile"]
        results.append(f"Profile Data: {profile_data}")

        if profile_data.get("role"):
            results.append(f"Professional Analysis: {profile_data['role']} with expertise in {', '.join(profile_data.get('preferences', []))}")

    if "current_context" in dependencies:
        context_data = dependencies["current_context"]
        results.append(f"Current Context: {context_data}")
        results.append(f"Time-based Analysis: Analysis performed on {context_data['day_of_week']} at {context_data['current_time']}")

    print(f"--> Tool returned results: {results}")

    return "\n\n".join(results)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[analyze_user],
    name="User Analysis Agent",
    description="An agent specialized in analyzing users using integrated data sources.",
    instructions=[
        "You are a user analysis expert with access to user analysis tools.",
        "When asked to analyze any user, use the analyze_user tool.",
        "This tool has access to user profiles and current context through integrated data sources.",
        "After getting tool results, provide additional insights and recommendations based on the analysis.",
        "Be thorough in your analysis and explain what the tool found."
    ],
)

print("=== Tool Dependencies Access Example ===\n")

response = agent.run(
    input="Please analyze user 'john_doe' and provide insights about their professional background and preferences.",
    dependencies={
        "user_profile": {
            "name": "John Doe",
            "preferences": ["AI/ML", "Software Engineering", "Finance"],
            "location": "San Francisco, CA",
            "role": "Senior Software Engineer",
        },
        "current_context": get_current_context,
    },
    session_id="test_tool_dependencies",
)

print(f"\nAgent Response: {response.content}")

```


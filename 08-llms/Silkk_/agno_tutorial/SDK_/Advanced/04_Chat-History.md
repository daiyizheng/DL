# 聊天记录

保存并访问多轮互动中的对话历史记录。

聊天记录使您的客服人员、团队和工作流程能够记住并参考以前的对话，从而创建智能且具有上下文感知能力的交互。


聊天记录功能允许您在每次互动时无需重新开始：
- 保持对话的连贯性——在会话期间，基于之前的交流进行深入探讨。
- 提供个性化回复——参考过往互动来调整输出内容
- 避免重复提问- 获取之前提供的信息
- 支持长时间对话- 支持具有持久内存的多轮对话


## 代理中的聊天记录

配置和访问客服人员对话历史记录。

启用存储功能的代理会自动访问会话的运行历史记录（也称为“对话历史记录”或“聊天历史记录”）。

我们可以通过以下方式授予代理访问聊天记录的权限：

### 代理级历史记录
- 你可以设置 `add_history_to_context=True` 和 `num_history_runs=5`，自动将最近 5 次运行的输入和响应添加到每个发送给代理的请求中。
- 你可以通过设置`num_history_messages`来更细致地决定要添加多少消息到发送到模型的列表中。
- 你可以设置`read_chat_history=True`，为你的座席提供一个`get_chat_history（）`工具，让它能读取整个聊天记录中的任何一条消息。
- 你可以设置 `read_tool_call_history=True`，为代理提供一个 `get_tool_call_history（）` 工具，使其能够按倒序读取工具调用。
- 你可以启用`search_session_history`，允许搜索之前的会话。

### 历史参考
```python
agent = Agent(
    db=SqliteDb(db_file="tmp/agent.db"),
    add_history_to_context=True,
    num_history_runs=5,
)
```

```python
agent = Agent(
    db=SqliteDb(db_file="tmp/agent.db"),
    read_chat_history=True,  # Agent decides when to look up
)
```

```python
agent = Agent(
    db=SqliteDb(db_file="tmp/agent.db"),
    search_session_history=True,
    num_history_sessions=2,  # Keep low
)
```

### 将历史记录添加到代理上下文
要将对话历史添加到上下文中，可以设置add_history_to_context=True。这会将最近三次运行（即默认）的输入和响应添加到代理的上下文中。你可以通过设置 num_history_runs=n 来更改运行次数，其中 n 是要包含的运行次数。

你可以直接在代理上设置 add_history_to_context=True，或者直接在 run（） 方法上设置。完整实现请参见带历史的持久会话示例。

更多信息请参见上下文工程文档。

### 阅读聊天记录
要读取聊天记录，您可以进行设置read_chat_history=True。这将为get_chat_history()您的代理提供一个工具，使其能够读取整个聊天记录中的任何消息。
请参阅“聊天记录管理”页面以了解完整的实现方法。

### 搜索会话历史记录
在某些情况下，您可能需要从多个会话中获取消息，以便在对话中提供上下文或连续性。

要启用从最近 N 个会话中获取消息的功能，您需要使用以下标志：
- 启用search_session_history此选项True可允许搜索以前的会话。
- num_history_sessions：指定要包含在搜索中的历史会话数量。在下面的示例中，设置为2仅包含最近 2 个会话。

## 聊天记录

本示例演示如何管理和检索客服对话中的聊天记录，从而可以访问以前的对话消息和上下文。

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
)

agent.print_response("Tell me a new interesting fact about space", stream=True)
print(agent.get_chat_history())

agent.print_response("Tell me a new interesting fact about oceans", stream=True)
print(agent.get_chat_history())
```

### Teams中的聊天记录

管理团队会议历史记录和对话上下文。

启用存储功能的团队会自动访问会话的运行历史记录（也称为“对话历史记录”或“聊天历史记录”）。

我们可以通过以下方式授予团队访问聊天记录的权限：

团队历史：

- 您可以设置add_history_to_context=True自动num_history_runs=5将最近 5 次运行的输入和响应添加到发送给团队领导的每个请求中。
- 您可以通过设置来更精细地控制要添加到发送给模型的列表中的消息数量num_history_messages。
- 您可以设置向您的团队read_chat_history=True提供一个get_chat_history()工具，允许他们阅读整个聊天记录中的任何消息。
- 您可以设置向您的团队read_tool_call_history=True提供一个get_tool_call_history()工具，使其能够按时间倒序读取工具调用。
- 您可以启用此功能search_session_history以允许搜索以前的会话。
- 您可以设置add_team_history_to_members=True将num_team_history_runs=5最近 5 次运行的输入和响应（即团队级别的输入和响应）自动添加到发送给团队成员的每条消息中。

会员级别历史记录：

您还可`add_history_to_context`以为单个团队成员启用此功能。启用后，只会将该成员的输入和输出添加到发送给该成员的所有请求中，使其能够访问自己的历史记录。


### 历史参考

首先从团队历史背景入手，以便进行基本的对话衔接：

```python
team = Team(
    members=[...],
    db=SqliteDb(db_file="tmp/team.db"),
    add_history_to_context=True,
    num_history_runs=5,
)


team = Team(
    members=[german_agent, spanish_agent],
    db=SqliteDb(db_file="tmp/team.db"),
    add_team_history_to_members=True,
    num_team_history_runs=3,
)

team = Team(
    members=[profile_agent, billing_agent],
    db=SqliteDb(db_file="tmp/team.db"),
    share_member_interactions=True,
)

team = Team(
    members=[...],
    db=SqliteDb(db_file="tmp/team.db"),
    read_chat_history=True,  # Agent decides when to look up
)

team = Team(
    members=[...],
    db=SqliteDb(db_file="tmp/team.db"),
    search_session_history=True,
    num_history_sessions=2,  # Keep low
)
```


## 工作流程历史记录和持续执行

工作流历史记录使您的 Agno 工作流能够记住和引用以前的对话，将孤立的执行转变为连续的、具有上下文感知能力的交互。

使用工作流历史记录，您无需每次都从头开始：

- 在前人互动的基础上继续——参考过去互动的背景
- 避免重复提问——避免询问之前已提供的信息。
- 保持语境连贯性——营造对话体验
- 从模式中学习——分析历史数据以做出更好的决策


### 工作原理
启用工作流历史记录后，之前的消息将作为结构化上下文自动注入到代理/团队输入中：

```python
<workflow_history_context>
[run-1]
input: Create content about AI in healthcare
response: # AI in Healthcare: Transforming Patient Care...

[run-2] 
input: Make it more family-focused
response: # AI in Family Healthcare: A Parent's Guide...
</workflow_history_context>

Your current input goes here...
```

此外，在使用带有自定义函数的步骤时，您可以通过以下方式访问此历史记录：
1. 如上所示的格式化上下文字符串
2. 以结构化格式呈现，以便更好地控制

例子-

```python

def custom_function(step_input: StepInput) -> StepOutput:
    # Option 1: Structured data for analysis
    history_tuples = step_input.get_workflow_history(num_runs=3)
    for user_input, workflow_output in history_tuples:
        # Process each conversation turn

    # Option 2: Formatted context for agents  
    context_string = step_input.get_workflow_history_context(num_runs=3)

    return StepOutput(content="Analysis complete")
```

您可以使用以下辅助函数访问历史记录：
- step_input.get_workflow_history(num_runs=3)
- step_input.get_workflow_history_context(num_runs=3)

更多详情请参阅StepInput参考文档。

### 控制水平
您可以具体指定要将历史记录添加到哪些步骤：
​
#### 工作流级别历史记录
将工作流历史记录添加到工作流中的所有步骤：

```python
workflow = Workflow(
    steps=[research_step, analysis_step, writing_step],
    add_workflow_history_to_steps=True  # All steps get history
)
```

#### 逐步历史记录
仅为特定步骤添加工作流程历史记录：
```python
Step(
    name="Content Creator", 
    agent=content_agent,
    add_workflow_history=True  # Only this step gets history
)
```

### 优先级逻辑
步骤级设置始终优先于工作流级设置：

```python
workflow = Workflow(
    steps=[
        Step("Research", agent=research_agent),                              # None → inherits workflow setting
        Step("Analysis", agent=analysis_agent, add_workflow_history=False),  # False → overrides workflow  
        Step("Writing", agent=writing_agent, add_workflow_history=True),     # True → overrides workflow
    ],
    add_workflow_history_to_steps=True  # Default for all steps
)
```

### 历史长度控制
默认情况下，所有可用历史记录都会被包含（无限制）。建议设置固定的历史记录运行次数限制，以避免LLM上下文窗口过大。
您可以在两个层面上控制这一点：

```python
# Workflow-level: limit history for all steps
workflow = Workflow(
    add_workflow_history_to_steps=True,
    num_history_runs=5  # Only last 5 runs
)

# Step-level: override for specific steps
Step("Analysis", agent=analysis_agent, 
     add_workflow_history=True,
     num_history_runs=3  # Only last 3 runs for this step
)
```

## 单步工作流程
此示例演示了一个包含单个步骤的工作流程，该工作流程会持续执行，并可访问工作流程历史记录。

此示例展示了如何使用该add_workflow_history_to_steps标志将工作流历史记录添加到工作流中的所有步骤。

在这种情况下，我们采用的是一个单步骤工作流程，只有一个代理。

代理可以访问工作流程历史记录，并利用这些历史记录提供个性化的教育支持。

```python

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.workflow.step import Step
from agno.workflow.workflow import Workflow

tutor_agent = Agent(
    name="AI Tutor",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=[
        "You are an expert tutor who provides personalized educational support.",
        "You have access to our full conversation history.",
        "Build on previous discussions - don't repeat questions or information.",
        "Reference what the student has told you earlier in our conversation.",
        "Adapt your teaching style based on what you've learned about the student.",
        "Be encouraging, patient, and supportive.",
        "When asked about conversation history, provide a helpful summary.",
        "Focus on helping the student understand concepts and improve their skills.",
    ],
)

tutor_workflow = Workflow(
    name="Simple AI Tutor",
    description="Single-step conversational tutoring with history awareness",
    db=SqliteDb(db_file="tmp/simple_tutor_workflow.db"),
    steps=[
        Step(name="AI Tutoring", agent=tutor_agent),
    ],
    add_workflow_history_to_steps=True,  # This adds the workflow history
)


def demo_simple_tutoring_cli():
    """Demo simple single-step tutoring workflow"""
    print("Simple AI Tutor Demo - Type 'exit' to quit")
    print("Try asking about:")
    print("- 'I'm struggling with calculus derivatives'")
    print("- 'Can you help me with algebra?'")
    print("-" * 60)

    tutor_workflow.cli_app(
        session_id="simple_tutor_demo",
        user="Student",
        stream=True,
        show_step_details=True,
    )


if __name__ == "__main__":
    demo_simple_tutoring_cli()
```

## 多步骤工作流程

此示例演示了启用特定步骤历史记录的工作流程。

此示例展示了如何使用该add_workflow_history_to_steps标志将工作流历史记录添加到工作流中的多个步骤。在本例中，我们有一个包含三个步骤的工作流。

第一步是提供餐食建议，推荐餐食类别和菜系。

第二步是偏好分析步骤，分析对话历史以了解用户的食物偏好。

第三步是由食谱专家根据用户的喜好提供食谱推荐。

```python

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.workflow.step import Step, StepInput, StepOutput
from agno.workflow.workflow import Workflow

# Define specialized agents for meal planning conversation
meal_suggester = Agent(
    name="Meal Suggester",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=[
        "You are a friendly meal planning assistant who suggests meal categories and cuisines.",
        "Consider the time of day, day of the week, and any context from the conversation.",
        "Keep suggestions broad (Italian, Asian, healthy, comfort food, quick meals, etc.)",
        "Ask follow-up questions to understand preferences better.",
    ],
)

recipe_specialist = Agent(
    name="Recipe Specialist",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=[
        "You are a recipe expert who provides specific, detailed recipe recommendations.",
        "Pay close attention to the full conversation to understand user preferences and restrictions.",
        "If the user mentioned avoiding certain foods or wanting healthier options, respect that.",
        "Provide practical, easy-to-follow recipe suggestions with ingredients and basic steps.",
        "Reference the conversation naturally (e.g., 'Since you mentioned wanting something healthier...')",
    ],
)


def analyze_food_preferences(step_input: StepInput) -> StepOutput:
    """
    Smart function that analyzes conversation history to understand user food preferences
    """
    current_request = step_input.input
    conversation_context = step_input.previous_step_content or ""

    # Simple preference analysis based on conversation
    preferences = {
        "dietary_restrictions": [],
        "cuisine_preferences": [],
        "avoid_list": [],
        "cooking_style": "any",
    }

    # Analyze conversation for patterns
    full_context = f"{conversation_context} {current_request}".lower()

    # Dietary restrictions and preferences
    if any(word in full_context for word in ["healthy", "healthier", "light", "fresh"]):
        preferences["dietary_restrictions"].append("healthy")
    if any(word in full_context for word in ["vegetarian", "veggie", "no meat"]):
        preferences["dietary_restrictions"].append("vegetarian")
    if any(word in full_context for word in ["quick", "fast", "easy", "simple"]):
        preferences["cooking_style"] = "quick"
    if any(word in full_context for word in ["comfort", "hearty", "filling"]):
        preferences["cooking_style"] = "comfort"

    # Foods/cuisines to avoid (mentioned recently)
    if "italian" in full_context and (
        "had" in full_context or "yesterday" in full_context
    ):
        preferences["avoid_list"].append("Italian")
    if "chinese" in full_context and (
        "had" in full_context or "recently" in full_context
    ):
        preferences["avoid_list"].append("Chinese")

    # Preferred cuisines mentioned positively
    if "love asian" in full_context or "like asian" in full_context:
        preferences["cuisine_preferences"].append("Asian")
    if "mediterranean" in full_context:
        preferences["cuisine_preferences"].append("Mediterranean")

    # Create guidance for the recipe agent
    guidance = []
    if preferences["dietary_restrictions"]:
        guidance.append(
            f"Focus on {', '.join(preferences['dietary_restrictions'])} options"
        )
    if preferences["avoid_list"]:
        guidance.append(
            f"Avoid {', '.join(preferences['avoid_list'])} cuisine since user had it recently"
        )
    if preferences["cuisine_preferences"]:
        guidance.append(
            f"Consider {', '.join(preferences['cuisine_preferences'])} options"
        )
    if preferences["cooking_style"] != "any":
        guidance.append(f"Prefer {preferences['cooking_style']} cooking style")

    analysis_result = f"""
        PREFERENCE ANALYSIS:
        Current Request: {current_request}

        Detected Preferences:
        {chr(10).join(f"• {g}" for g in guidance) if guidance else "• No specific preferences detected"}

        RECIPE AGENT GUIDANCE:
        Based on the conversation history, please provide recipe recommendations that align with these preferences.
        Reference the conversation naturally and explain why these recipes fit their needs.
    """.strip()

    return StepOutput(content=analysis_result)


# Define workflow steps
suggestion_step = Step(
    name="Meal Suggestion",
    agent=meal_suggester,
)

preference_analysis_step = Step(
    name="Preference Analysis",
    executor=analyze_food_preferences,
)

recipe_step = Step(
    name="Recipe Recommendations",
    agent=recipe_specialist,
)

# Create conversational meal planning workflow
meal_workflow = Workflow(
    name="Conversational Meal Planner",
    description="Smart meal planning with conversation awareness and preference learning",
    db=SqliteDb(
        session_table="workflow_session",
        db_file="tmp/meal_workflow.db",
    ),
    steps=[suggestion_step, preference_analysis_step, recipe_step],
    add_workflow_history_to_steps=True,
)


def demonstrate_conversational_meal_planning():
    """Demonstrate natural conversational meal planning"""
    session_id = "meal_planning_demo"

    print("Conversational Meal Planning Demo")
    print("=" * 60)

    # First interaction
    print("\nUser: What should I cook for dinner tonight?")
    meal_workflow.print_response(
        input="What should I cook for dinner tonight?",
        session_id=session_id,
        markdown=True,
    )

    # Second interaction - user provides preferences
    print(
        "\nUser: I had Italian yesterday, and I'm trying to eat healthier these days"
    )
    meal_workflow.print_response(
        input="I had Italian yesterday, and I'm trying to eat healthier these days",
        session_id=session_id,
        markdown=True,
    )

    # Third interaction - more specific request
    print(
        "\nUser: Actually, do you have something with fish? I love Asian flavors too"
    )
    meal_workflow.print_response(
        input="Actually, do you have something with fish? I love Asian flavors too",
        session_id=session_id,
        markdown=True,
    )


if __name__ == "__main__":
    demonstrate_conversational_meal_planning()
```

### 逐步历史记录

此示例演示了启用特定步骤历史记录的工作流程。

此示例演示如何使用add_workflow_history标志将工作流历史记录添加到工作流中的特定步骤。在本例中，我们有一个包含三个步骤的工作流。
第一步是聘请研究专家收集相关主题的信息。
第二步是寻找能够创作引人入胜内容的创作者。
第三步是内容发布者准备发布内容。

```python
"""
This example shows step-level add_workflow_history control.
Only the Content Creator step gets workflow history to avoid repeating previous content.

Workflow: Research → Content Creation (with history) → Publishing
"""

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.workflow.step import Step
from agno.workflow.workflow import Workflow

research_agent = Agent(
    name="Research Specialist",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=[
        "You are a research specialist who gathers information on topics.",
        "Conduct thorough research and provide key facts, trends, and insights.",
        "Focus on current, accurate information from reliable sources.",
        "Organize your findings in a clear, structured format.",
        "Provide citations and context for your research.",
    ],
)

content_creator = Agent(
    name="Content Creator",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=[
        "You are an expert content creator who writes engaging content.",
        "Use the research provided and CREATE UNIQUE content that stands out.",
        "IMPORTANT: Review workflow history to understand:",
        "- What content topics have been covered before",
        "- What writing styles and formats were used previously",
        "- User preferences and content patterns",
        "- Avoid repeating similar content or approaches",
        "Build on previous themes while keeping content fresh and original.",
        "Reference the conversation history to maintain consistency in tone and style.",
        "Create compelling headlines, engaging intros, and valuable content.",
    ],
)

publisher_agent = Agent(
    name="Content Publisher",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=[
        "You are a content publishing specialist.",
        "Review the created content and prepare it for publication.",
        "Add appropriate hashtags, formatting, and publishing recommendations.",
        "Suggest optimal posting times and distribution channels.",
        "Ensure content meets platform requirements and best practices.",
    ],
)

workflow = Workflow(
    name="Smart Content Creation Pipeline",
    description="Research → Content Creation (with history awareness) → Publishing",
    db=SqliteDb(db_file="tmp/content_workflow.db"),
    steps=[
        Step(
            name="Research Phase",
            agent=research_agent,
            add_workflow_history=True,  # Specifically add history to this step
        ),
        # Content creation step - uses workflow history to avoid repetition and give better results
        Step(
            name="Content Creation",
            agent=content_creator,
            add_workflow_history=True,  # Specifically add history to this step
        ),
        Step(
            name="Content Publishing",
            agent=publisher_agent,
        ),
    ],
)


if __name__ == "__main__":
    print("Content Creation Demo - Step-Level History Control")
    print("Only the Content Creator step sees previous workflow history!")
    print("")
    print("Try these content requests:")
    print("- 'Create a LinkedIn post about AI trends in 2024'")
    print("- 'Write a Twitter thread about productivity tips'")
    print("- 'Create a blog intro about remote work benefits'")
    print("")
    print(
        "Notice how the Content Creator references previous content to avoid repetition!"
    )
    print("Type 'exit' to quit")
    print("-" * 70)

    workflow.cli_app(
        session_id="content_demo",
        user="Content Requester",
        stream=True,
    )
```

### 函数中的历史
本示例演示如何在自定义函数中获取工作流历史记录。

此示例展示了如何在自定义函数中获取工作流历史记录。

- 通过这种方法，`step_input.get_workflow_history(num_runs=5)`我们可以将历史记录获取为元组列表的形式。
- 我们还可以使用`step_input.get_workflow_history_context(num_runs=5)`字符串来获取历史记录。


```python
import json

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.workflow.step import Step
from agno.workflow.types import StepInput, StepOutput
from agno.workflow.workflow import Workflow


def analyze_content_strategy(step_input: StepInput) -> StepOutput:
    current_topic = step_input.input or ""
    research_data = step_input.get_last_step_content() or ""
    history_data = step_input.get_workflow_history(
        num_runs=5
    )  # history as a list of tuples

    # use this if you need history as a string for direct use.
    # history_str = step_input.get_workflow_history_context(num_runs=5)

    def extract_keywords(text: str) -> set:
        stop_words = {
            "create",
            "content",
            "about",
            "write",
            "the",
            "a",
            "an",
            "how",
            "is",
            "of",
            "this",
            "that",
            "in",
            "on",
            "for",
            "to",
        }
        words = set(text.lower().split()) - stop_words

        keyword_map = {
            "ai": ["ai", "artificial", "intelligence"],
            "ml": ["machine", "learning", "ml"],
            "healthcare": ["medical", "health", "healthcare", "medicine"],
            "blockchain": ["crypto", "cryptocurrency", "blockchain"],
        }

        expanded_keywords = set(words)
        for word in list(words):
            for key, synonyms in keyword_map.items():
                if word in synonyms:
                    expanded_keywords.update([word])

        return expanded_keywords

    current_keywords = extract_keywords(current_topic)
    max_possible_overlap = len(current_keywords)
    topic_overlaps = []
    covered_topics = []

    for input_request, content_output in history_data:
        if input_request:
            covered_topics.append(input_request.lower())
            previous_keywords = extract_keywords(input_request)

            overlap = len(current_keywords.intersection(previous_keywords))
            if overlap > 0:
                topic_overlaps.append(overlap)

    topic_overlap = max(topic_overlaps) if topic_overlaps else 0
    overlap_percentage = (topic_overlap / max(max_possible_overlap, 1)) * 100
    diversity_score = len(set(covered_topics)) / max(len(covered_topics), 1)

    recommendations = []
    if overlap_percentage > 60:
        recommendations.append(
            "HIGH OVERLAP detected - consider a fresh angle or advanced perspective"
        )
    elif overlap_percentage > 30:
        recommendations.append(
            "MODERATE OVERLAP detected - differentiate your approach"
        )
    if diversity_score < 0.6:
        recommendations.append(
            "Low content diversity - explore different aspects of the topic"
        )
    if len(history_data) > 0:
        recommendations.append(
            f"Building on {len(history_data)} previous content pieces - ensure progression"
        )

    # Structure the analysis with better metrics
    strategy_analysis = {
        "content_topic": current_topic,
        "historical_coverage": {
            "previous_topics": covered_topics[-3:],
            "topic_overlap_score": topic_overlap,
            "overlap_percentage": round(overlap_percentage, 1),
            "content_diversity": diversity_score,
        },
        "strategic_recommendations": recommendations,
        "research_summary": research_data[:500] + "..."
        if len(research_data) > 500
        else research_data,
        "suggested_angle": "unique perspective"
        if overlap_percentage > 30
        else "comprehensive overview",
        "content_gap_analysis": {
            "avoid_repeating": [
                topic
                for topic in covered_topics
                if any(word in current_topic.lower() for word in topic.split()[:2])
            ],
            "build_upon": "previous insights"
            if len(history_data) > 0
            else "foundational knowledge",
        },
    }

    # Format with proper metrics
    formatted_analysis = f"""
        CONTENT STRATEGY ANALYSIS
        ========================

        STRATEGIC OVERVIEW:
        - Topic: {strategy_analysis["content_topic"]}
        - Previous Content Count: {len(history_data)}
        - Keyword Overlap: {strategy_analysis["historical_coverage"]["topic_overlap_score"]} keywords ({strategy_analysis["historical_coverage"]["overlap_percentage"]}%)
        - Content Diversity: {strategy_analysis["historical_coverage"]["content_diversity"]:.2f}

        RECOMMENDATIONS:
        {chr(10).join([f"- {rec}" for rec in strategy_analysis["strategic_recommendations"]])}

        RESEARCH FOUNDATION:
        {strategy_analysis["research_summary"]}

        CONTENT POSITIONING:
        - Suggested Angle: {strategy_analysis["suggested_angle"]}
        - Build Upon: {strategy_analysis["content_gap_analysis"]["build_upon"]}
        - Differentiate From: {", ".join(strategy_analysis["content_gap_analysis"]["avoid_repeating"]) if strategy_analysis["content_gap_analysis"]["avoid_repeating"] else "No similar content found"}

        CREATIVE DIRECTION:
        Based on historical analysis, focus on providing {strategy_analysis["suggested_angle"]} while ensuring the content complements rather than duplicates previous work.

        STRUCTURED_DATA: {json.dumps(strategy_analysis, indent=2)}
    """

    return StepOutput(content=formatted_analysis.strip())


def create_content_workflow():
    """Professional content creation workflow with strategic analysis"""

    # Step 1: Research Agent gathers comprehensive information
    research_step = Step(
        name="Content Research",
        agent=Agent(
            name="Research Specialist",
            model=OpenAIResponses(id="gpt-5.2"),
            instructions=[
                "You are an expert research specialist for content creation.",
                "Conduct thorough research on the requested topic.",
                "Gather current trends, key insights, statistics, and expert perspectives.",
                "Structure your research with clear sections: Overview, Key Points, Recent Developments, Expert Insights.",
                "Prioritize accurate, up-to-date information from credible sources.",
                "Keep research comprehensive but concise for content creators to use.",
            ],
        ),
    )

    # Step 2: Custom function analyzes content strategy and prevents duplication
    strategy_step = Step(
        name="Content Strategy Analysis",
        executor=analyze_content_strategy,
        description="Analyze content strategy using historical data to prevent duplication and identify opportunities",
    )

    # Step 3: Strategic Writer creates final content with full context
    writer_step = Step(
        name="Strategic Content Creation",
        agent=Agent(
            name="Content Strategist",
            model=OpenAIResponses(id="gpt-5.2"),
            instructions=[
                "You are a strategic content writer who creates high-quality, unique content.",
                "Use the research and strategic analysis to create compelling content.",
                "Follow the strategic recommendations to ensure content uniqueness.",
                "Structure content with: Hook, Main Content, Key Takeaways, Call-to-Action.",
                "Ensure your content builds upon previous work rather than repeating it.",
                "Include 'Target Audience:' and 'Content Type:' at the end for tracking.",
                "Make content engaging, actionable, and valuable to readers.",
            ],
        ),
    )

    return Workflow(
        name="Strategic Content Creation",
        description="Research → Strategic Analysis → Content Creation with historical awareness",
        db=SqliteDb(db_file="tmp/content_workflow.db"),
        steps=[research_step, strategy_step, writer_step],
        add_workflow_history_to_steps=True,
    )


def demo_content_workflow():
    """Demo the strategic content creation workflow"""
    workflow = create_content_workflow()

    print("Strategic Content Creation Workflow")
    print("Flow: Research -> Strategy Analysis -> Content Writing")
    print("")
    print(
        "This workflow prevents duplicate content and ensures strategic progression"
    )
    print("")
    print("Try these content requests:")
    print("- 'Create content about AI in healthcare'")
    print("- 'Write about machine learning applications' (will detect overlap)")
    print("- 'Content on blockchain technology' (different topic)")
    print("")
    print("Type 'exit' to quit")
    print("-" * 70)

    workflow.cli_app(
        session_id="content_strategy_demo",
        user="Content Manager",
        stream=True,
    )


if __name__ == "__main__":
    demo_content_workflow()
```

## 多用途命令行界面

本示例演示如何在多用途 CLI 中使用工作流历史记录。

此示例展示了如何使用该add_workflow_history_to_steps标志为工作流步骤添加历史记录。在本例中，我们有一个包含单个代理的多步骤工作流。

我们展示了工作流程持续执行的不同场景。我们有 5 个不同的演示：
- 客户支持
- 医疗咨询
- 辅导


```python

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.workflow.step import Step
from agno.workflow.workflow import Workflow

# ==============================================================================
# 1. CUSTOMER SUPPORT WORKFLOW
# ==============================================================================


def create_customer_support_workflow():
    """Multi-step customer support with escalation and context retention"""

    intake_agent = Agent(
        name="Support Intake Specialist",
        model=OpenAIResponses(id="gpt-5.2"),
        instructions=[
            "You are a friendly customer support intake specialist.",
            "Gather initial problem details, customer info, and urgency level.",
            "Ask clarifying questions to understand the issue completely.",
            "Classify issues as: technical, billing, account, or general inquiry.",
            "Be empathetic and professional.",
        ],
    )

    technical_specialist = Agent(
        name="Technical Support Specialist",
        model=OpenAIResponses(id="gpt-5.2"),
        instructions=[
            "You are a technical support expert with deep product knowledge.",
            "Review the full conversation history to understand the customer's issue.",
            "Reference what the intake specialist learned to avoid repeating questions.",
            "Provide step-by-step troubleshooting or technical solutions.",
            "If you can't solve it, escalate with detailed context.",
        ],
    )

    resolution_manager = Agent(
        name="Resolution Manager",
        model=OpenAIResponses(id="gpt-5.2"),
        instructions=[
            "You are a customer success manager who ensures resolution.",
            "Review the entire support conversation to understand what happened.",
            "Provide final resolution, follow-up steps, and ensure customer satisfaction.",
            "Reference specific details from earlier in the conversation.",
            "Be solution-oriented and customer-focused.",
        ],
    )

    return Workflow(
        name="Customer Support Pipeline",
        description="Multi-agent customer support with conversation continuity",
        db=SqliteDb(db_file="tmp/support_workflow.db"),
        steps=[
            Step(name="Support Intake", agent=intake_agent),
            Step(name="Technical Resolution", agent=technical_specialist),
            Step(name="Final Resolution", agent=resolution_manager),
        ],
        add_workflow_history_to_steps=True,
    )


# ==============================================================================
# 2. MEDICAL CONSULTATION WORKFLOW
# ==============================================================================


def create_medical_consultation_workflow():
    """Medical consultation with symptom analysis and specialist referral"""

    triage_nurse = Agent(
        name="Triage Nurse",
        model=OpenAIResponses(id="gpt-5.2"),
        instructions=[
            "You are a professional triage nurse conducting initial assessment.",
            "Gather symptoms, medical history, and current medications.",
            "Ask about pain levels, duration, and severity.",
            "Document everything clearly for the consulting physician.",
            "Be thorough but compassionate.",
        ],
    )

    consulting_physician = Agent(
        name="Consulting Physician",
        model=OpenAIResponses(id="gpt-5.2"),
        instructions=[
            "You are an experienced physician reviewing the patient case.",
            "Review all information gathered by the triage nurse.",
            "Build on the conversation - don't repeat questions already asked.",
            "Provide differential diagnosis and recommend next steps.",
            "Explain medical reasoning in patient-friendly terms.",
        ],
    )

    care_coordinator = Agent(
        name="Care Coordinator",
        model=OpenAIResponses(id="gpt-5.2"),
        instructions=[
            "You coordinate follow-up care based on the full consultation.",
            "Reference specific details from the nurse assessment and physician recommendations.",
            "Provide clear next steps, appointment scheduling, and care instructions.",
            "Ensure continuity of care with detailed documentation.",
        ],
    )

    return Workflow(
        name="Medical Consultation",
        description="Comprehensive medical consultation with care coordination",
        db=SqliteDb(db_file="tmp/medical_workflow.db"),
        steps=[
            Step(name="Triage Assessment", agent=triage_nurse),
            Step(name="Physician Consultation", agent=consulting_physician),
            Step(name="Care Coordination", agent=care_coordinator),
        ],
        add_workflow_history_to_steps=True,
    )


# ==============================================================================
# 4. EDUCATIONAL TUTORING WORKFLOW
# ==============================================================================


def create_tutoring_workflow():
    """Personalized tutoring with adaptive learning"""

    learning_assessor = Agent(
        name="Learning Assessment Specialist",
        model=OpenAIResponses(id="gpt-5.2"),
        instructions=[
            "You are an educational assessment specialist.",
            "Evaluate the student's current knowledge level and learning style.",
            "Ask about specific topics they're struggling with.",
            "Identify knowledge gaps and learning preferences.",
            "Be encouraging and supportive.",
        ],
    )

    subject_tutor = Agent(
        name="Subject Matter Tutor",
        model=OpenAIResponses(id="gpt-5.2"),
        instructions=[
            "You are an expert tutor in the student's subject area.",
            "Build on the assessment discussion - don't repeat questions.",
            "Teach using methods that match the student's identified learning style.",
            "Reference specific gaps and challenges mentioned earlier.",
            "Provide clear explanations and check for understanding.",
        ],
    )

    progress_coach = Agent(
        name="Learning Progress Coach",
        model=OpenAIResponses(id="gpt-5.2"),
        instructions=[
            "You are a learning coach focused on student success.",
            "Review the entire tutoring session for context.",
            "Provide study strategies based on what was discussed.",
            "Reference specific learning challenges and successes from the conversation.",
            "Create actionable next steps and encourage continued learning.",
        ],
    )

    return Workflow(
        name="Personalized Tutoring Session",
        description="Adaptive educational support with learning continuity",
        db=SqliteDb(db_file="tmp/tutoring_workflow.db"),
        steps=[
            Step(name="Learning Assessment", agent=learning_assessor),
            Step(name="Subject Tutoring", agent=subject_tutor),
            Step(name="Progress Planning", agent=progress_coach),
        ],
        add_workflow_history_to_steps=True,
    )


# ==============================================================================
# DEMO FUNCTIONS USING CLI
# ==============================================================================


def demo_customer_support_cli():
    """Demo customer support workflow with CLI"""
    support_workflow = create_customer_support_workflow()

    print("Customer Support Demo - Type 'exit' to quit")
    print("Try: 'My account is locked and I can't access my billing information'")
    print("-" * 60)

    support_workflow.cli_app(
        session_id="support_demo",
        user="Customer",
        stream=True,
    )


def demo_medical_consultation_cli():
    """Demo medical consultation workflow with CLI"""
    medical_workflow = create_medical_consultation_workflow()

    print("Medical Consultation Demo - Type 'exit' to quit")
    print("Try: 'I've been having chest pain and shortness of breath for 2 days'")
    print("-" * 60)

    medical_workflow.cli_app(
        session_id="medical_demo",
        user="Patient",
        stream=True,
    )


def demo_tutoring_cli():
    """Demo tutoring workflow with CLI"""
    tutoring_workflow = create_tutoring_workflow()

    print("Tutoring Session Demo - Type 'exit' to quit")
    print("Try: 'I'm struggling with calculus derivatives and have a test next week'")
    print("-" * 60)

    tutoring_workflow.cli_app(
        session_id="tutoring_demo",
        user="Student",
        stream=True,
    )


if __name__ == "__main__":
    import sys

    demos = {
        "support": demo_customer_support_cli,
        "medical": demo_medical_consultation_cli,
        "tutoring": demo_tutoring_cli,
    }

    if len(sys.argv) > 1 and sys.argv[1] in demos:
        demos[sys.argv[1]]()
    else:
        print("Conversational Workflow Demos")
        print("Choose a demo to run:")
        print("")
        for key, func in demos.items():
            print(f"{key:<10} - {func.__doc__}")
        print("")
        print("Or run all demos interactively:")
        choice = input("Enter demo name (or 'all'): ").strip().lower()

        if choice == "all":
            for demo_func in demos.values():
                demo_func()
        elif choice in demos:
            demos[choice]()
        else:
            print("Invalid choice!")
```

## 意图路由

此示例演示如何在意图路由中使用工作流历史记录。

此示例演示了：

1. 一个简单的路由器，可以将请求路由到不同的专业代理。
2. 为了保持上下文的连贯性，所有代理共享相同的对话历史记录。
3. 不同主体间共享上下文的力量

路由器使用基本的意图检测，但真正的价值在于共享历史记录。

```python
from typing import List

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.workflow.router import Router
from agno.workflow.step import Step
from agno.workflow.types import StepInput
from agno.workflow.workflow import Workflow

# Define specialized customer service agents
tech_support_agent = Agent(
    name="Technical Support Specialist",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=[
        "You are a technical support specialist with deep product knowledge.",
        "You have access to the full conversation history with this customer.",
        "Reference previous interactions to provide better help.",
        "Build on any troubleshooting steps already attempted.",
        "Be patient and provide step-by-step technical guidance.",
    ],
)

billing_agent = Agent(
    name="Billing & Account Specialist",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=[
        "You are a billing and account specialist.",
        "You have access to the full conversation history with this customer.",
        "Reference any account details or billing issues mentioned previously.",
        "Build on any payment or account information already discussed.",
        "Be helpful with billing questions, refunds, and account changes.",
    ],
)

general_support_agent = Agent(
    name="General Customer Support",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=[
        "You are a general customer support representative.",
        "You have access to the full conversation history with this customer.",
        "Handle general inquiries, product information, and basic support.",
        "Reference the conversation context - build on what was discussed.",
        "Be friendly and acknowledge their previous interactions.",
    ],
)


# Create steps with shared history
tech_support_step = Step(
    name="Technical Support",
    agent=tech_support_agent,
    add_workflow_history=True,
)

billing_support_step = Step(
    name="Billing Support",
    agent=billing_agent,
    add_workflow_history=True,
)

general_support_step = Step(
    name="General Support",
    agent=general_support_agent,
    add_workflow_history=True,
)


def simple_intent_router(step_input: StepInput) -> List[Step]:
    """
    Simple intent-based router with basic keyword detection.
    The focus is on shared history, not complex routing logic.
    """
    current_message = step_input.input or ""
    current_message_lower = current_message.lower()

    # Simple keyword matching for intent detection
    tech_keywords = [
        "api",
        "error",
        "bug",
        "technical",
        "login",
        "not working",
        "broken",
        "crash",
    ]
    billing_keywords = [
        "billing",
        "payment",
        "refund",
        "charge",
        "subscription",
        "invoice",
        "plan",
    ]

    # Simple routing logic
    if any(keyword in current_message_lower for keyword in tech_keywords):
        print("Routing to Technical Support")
        return [tech_support_step]
    elif any(keyword in current_message_lower for keyword in billing_keywords):
        print("Routing to Billing Support")
        return [billing_support_step]
    else:
        print("Routing to General Support")
        return [general_support_step]


def create_smart_customer_service_workflow():
    """Customer service workflow with simple routing and shared history"""

    return Workflow(
        name="Smart Customer Service",
        description="Simple routing to specialists with shared conversation history",
        db=SqliteDb(db_file="tmp/smart_customer_service.db"),
        steps=[
            Router(
                name="Customer Service Router",
                selector=simple_intent_router,
                choices=[tech_support_step, billing_support_step, general_support_step],
                description="Routes to appropriate specialist based on simple intent detection",
            )
        ],
        add_workflow_history_to_steps=True,  # Enable history for the workflow
    )


def demo_smart_customer_service_cli():
    """Demo the smart customer service workflow with CLI"""
    workflow = create_smart_customer_service_workflow()

    print("Smart Customer Service Demo")
    print("=" * 60)
    print("")
    print("This workflow demonstrates:")
    print("- Simple routing between Technical, Billing, and General support")
    print("- Shared conversation history across ALL agents")
    print("- Context continuity - agents remember your entire conversation")
    print("")
    print("TRY THESE CONVERSATIONS:")
    print("")
    print("TECHNICAL SUPPORT:")
    print("   - 'My API is not working'")
    print("   - 'I'm getting an error message'")
    print("   - 'There's a technical bug'")
    print("")
    print("BILLING SUPPORT:")
    print("   - 'I need help with billing'")
    print("   - 'Can I get a refund?'")
    print("   - 'My payment was charged twice'")
    print("")
    print("GENERAL SUPPORT:")
    print("   - 'Hello, I have a question'")
    print("   - 'What features do you offer?'")
    print("   - 'I need general help'")
    print("")
    print("Type 'exit' to quit")
    print("-" * 60)

    workflow.cli_app(
        session_id="smart_customer_service_demo",
        user="Customer",
        stream=True,
        show_step_details=True,
    )


if __name__ == "__main__":
    demo_smart_customer_service_cli()
```
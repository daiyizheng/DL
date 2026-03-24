# 工作流程
## 什么是工作流程？
工作流通过定义步骤协调代理、团队和功能，以完成可重复的任务。

工作流将代理、团队和功能协调成一系列步骤。步骤可以按顺序、并行、循环或根据结果条件执行。每个步骤的输出都会流向下一个步骤，从而为复杂任务创建可预测的流程。

![](https://i-blog.csdnimg.cn/direct/751fb35852e945a6986bfc00e69d7e56.png)

## 你的第一个工作流程
以下是一个简单的工作流程，包括选定主题、进行研究并撰写文章：

```python
from agno.agent import Agent
from agno.workflow import Workflow
from agno.tools.hackernews import HackerNewsTools

researcher = Agent(
    name="Researcher",
    instructions="Find relevant information about the topic",
    tools=[HackerNewsTools()]
)

writer = Agent(
    name="Writer",
    instructions="Write a clear, engaging article based on the research"
)

content_workflow = Workflow(
    name="Content Creation",
    steps=[researcher, writer]
)

content_workflow.print_response("Write an article about AI trends", stream=True)
```

### 何时使用工作流
在以下情况下使用工作流程：
- 你需要可预测、可重复的执行。
- 任务具有清晰的顺序步骤，以及明确的输入和输出。
- 您需要审计跟踪和跨运行一致的结果
当您需要灵活、协作式的问题解决方式，并且需要代理人动态协调时，请使用团队模式。


## 构建工作流程

在工作流中定义步骤、循环、条件和并行执行。

工作流是协调客服人员和团队的强大工具。它是一系列按顺序执行的步骤，您可以完全控制这些步骤的执行流程。

### 积木

1. 该类Workflow是最高级别的协调器，负责管理整个执行过程。
2. Step是工作流系统中的基本工作单元。每个步骤都封装了一个执行器executor——可以是普通的执行器Agent、自Team定义的 Python 函数或自定义的 Python 函数。这种设计既保证了清晰性和可维护性，又保留了每个执行器的独特特性。
3. Loop是一种允许您多次执行一个或多个步骤的结构。当您需要重复执行一组步骤直到满足特定条件时，这非常有用。
4. Parallel是一种允许您并行执行一个或多个步骤的结构。当您需要同时执行一组步骤并将它们的输出合并在一起时，这非常有用。
5. Condition根据您指定的条件，使某个步骤具有条件性。
6. Router允许您指定下一步要执行的步骤，从而在工作流程中有效地创建分支逻辑。

### 如何创建你的第一个工作流程？
您可以使用不同类型的模式来构建工作流程。例如，您可以将代理、团队和功能组合起来构建工作流程。
```python
from agno.workflow import Step, Workflow, StepOutput

def data_preprocessor(step_input):
    # Custom preprocessing logic

    # Or you can also run any agent/team over here itself
    # response = some_agent.run(...)
    return StepOutput(content=f"Processed: {step_input.input}") # <-- Now pass the agent/team response in content here

workflow = Workflow(
    name="Mixed Execution Pipeline",
    steps=[
        research_team,      # Team
        data_preprocessor,  # Function
        content_agent,      # Agent
    ]
)

workflow.print_response("Analyze the competitive landscape for fintech startups", markdown=True)
```

## 运行工作流

使用 `Workflow.run()` 执行工作流并处理其输出。

该`Workflow.run()`函数运行代理并生成响应，响应可以是`WorkflowRunOutput`对象或对象流`WorkflowRunOutput`。

我们的许多示例都使用了`workflow.print_response()`一个辅助工具，用于在终端中打印响应。这`workflow.run()`在底层使用了其他方法。

### 运行您的工作流程
以下是如何运行您的工作流程。响应被捕获在`response.`

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.db.sqlite import SqliteDb
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.workflow import Step, Workflow
from agno.run.workflow import WorkflowRunOutput
from agno.utils.pprint import pprint_run_response

# Define agents
hackernews_agent = Agent(
    name="Hackernews Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
    role="Extract key insights and content from Hackernews posts",
)
finance_agent = Agent(
    name="Finance Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[YFinanceTools()],
    role="Get stock prices and financial data",
)

# Define research team for complex analysis
research_team = Team(
    name="Research Team",
    members=[hackernews_agent, finance_agent],
    instructions="Research tech topics and related stocks",
)

content_planner = Agent(
    name="Content Planner",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=[
        "Plan a content schedule over 4 weeks for the provided topic and research content",
        "Ensure that I have posts for 3 posts per week",
    ],
)

content_creation_workflow = Workflow(
    name="Content Creation Workflow",
    description="Automated content creation from blog posts to social media",
    db=SqliteDb(db_file="tmp/workflow.db"),
    steps=[research_team, content_planner],
)

# Create and use workflow
if __name__ == "__main__":
    response: WorkflowRunOutput = content_creation_workflow.run(
        input="AI trends in 2024",
        markdown=True,
    )

    pprint_run_response(response, markdown=True)
```

### 异步执行
该`Workflow.arun()`函数是异步版本`Workflow.run()`。

以下是使用示例：

```python
from typing import AsyncIterator
import asyncio

from agno.agent import Agent
from agno.tools.hackernews import HackerNewsTools
from agno.workflow import Condition, Step, Workflow, StepInput
from agno.run.workflow import WorkflowRunOutput, WorkflowRunOutputEvent, WorkflowRunEvent

# === BASIC AGENTS ===
researcher = Agent(
    name="Researcher",
    instructions="Research the given topic and provide detailed findings.",
    tools=[HackerNewsTools()],
)

summarizer = Agent(
    name="Summarizer",
    instructions="Create a clear summary of the research findings.",
)

fact_checker = Agent(
    name="Fact Checker",
    instructions="Verify facts and check for accuracy in the research.",
    tools=[HackerNewsTools()],
)

writer = Agent(
    name="Writer",
    instructions="Write a comprehensive article based on all available research and verification.",
)

# === CONDITION EVALUATOR ===
def needs_fact_checking(step_input: StepInput) -> bool:
    """Determine if the research contains claims that need fact-checking"""
    summary = step_input.previous_step_content or ""

    # Look for keywords that suggest factual claims
    fact_indicators = [
        "study shows",
        "breakthroughs",
        "research indicates",
        "according to",
        "statistics",
        "data shows",
        "survey",
        "report",
        "million",
        "billion",
        "percent",
        "%",
        "increase",
        "decrease",
    ]

    return any(indicator in summary.lower() for indicator in fact_indicators)


# === WORKFLOW STEPS ===
research_step = Step(
    name="research",
    description="Research the topic",
    agent=researcher,
)

summarize_step = Step(
    name="summarize",
    description="Summarize research findings",
    agent=summarizer,
)

# Conditional fact-checking step
fact_check_step = Step(
    name="fact_check",
    description="Verify facts and claims",
    agent=fact_checker,
)

write_article = Step(
    name="write_article",
    description="Write final article",
    agent=writer,
)

# === BASIC LINEAR WORKFLOW ===
basic_workflow = Workflow(
    name="Basic Linear Workflow",
    description="Research -> Summarize -> Condition(Fact Check) -> Write Article",
    steps=[
        research_step,
        summarize_step,
        Condition(
            name="fact_check_condition",
            description="Check if fact-checking is needed",
            evaluator=needs_fact_checking,
            steps=[fact_check_step],
        ),
        write_article,
    ],
)

async def main():
    try:
        response: WorkflowRunOutput = await basic_workflow.arun(
            input="Recent breakthroughs in quantum computing",
        )
        pprint_run_response(response, markdown=True)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```



### 流式执行器事件
工作流中使用的代理和团队的事件会在工作流流式传输期间自动生成。您可以通过设置`stream_executor_events=False`来选择不流式传输这些执行器事件。

以下工作流事件在所有情况下都会被流式传输：
- WorkflowStarted
- WorkflowCompleted
- StepStarted
- StepCompleted

请看以下示例：
```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.workflow.step import Step
from agno.workflow.workflow import Workflow

agent = Agent(
    name="ResearchAgent",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="You are a helpful research assistant. Be concise.",
)

workflow = Workflow(
    name="Research Workflow",
    steps=[Step(name="Research", agent=agent)],
    stream=True,
    stream_executor_events=False,  # <- Filter out internal executor events
)

print("\n" + "=" * 70)
print("Workflow Streaming Example: stream_executor_events=False")
print("=" * 70)
print(
    "\nThis will show only workflow and step events and will not yield RunContent and TeamRunContent events"
)
print("filtering out internal agent/team events for cleaner output.\n")

# Run workflow and display events
for event in workflow.run(
    "What is Python?",
    stream=True,
    stream_events=True,
):
    event_name = event.event if hasattr(event, "event") else type(event).__name__
    print(f"  → {event_name}")
```

### 异步流
它Workflow.arun(stream=True)返回的是一个异步WorkflowRunOutputEvent对象迭代​​器，而不是单个响应。例如，如果您想流式传输响应，可以执行以下操作：

```python
# Define your workflow
...

async def main():
    try:
        response: AsyncIterator[WorkflowRunOutputEvent] = basic_workflow.arun(
            message="Recent breakthroughs in quantum computing",
            stream=True,
            stream_events=True,
        )
        async for event in response:
            if event.event == WorkflowRunEvent.condition_execution_started.value:
                print(event)
                print()
            elif event.event == WorkflowRunEvent.condition_execution_completed.value:
                print(event)
                print()
            elif event.event == WorkflowRunEvent.workflow_started.value:
                print(event)
                print()
            elif event.event == WorkflowRunEvent.step_started.value:
                print(event)
                print()
            elif event.event == WorkflowRunEvent.step_completed.value:
                print(event)
                print()
            elif event.event == WorkflowRunEvent.workflow_completed.value:
                print(event)
                print()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
```

### 事件类型
根据工作流的配置，以下事件由Workflow.run()and Workflow.arun()函数产生：
#### 核心事件

| 事件类型 | 描述 |
| :--- | :--- |
| WorkflowStarted | 表示工作流运行的开始 |
| WorkflowCompleted | 表示工作流程运行已成功完成 |
| WorkflowError | 表示工作流运行期间发生错误 |


​#### 步骤事件
| 事件类型 | 描述 |
| :--- | :--- |
| StepStarted | 表示步骤的开始 |
| StepCompleted | 表示步骤已成功完成 |
| StepError | 表示在某个步骤中发生错误。 |

#### 步骤输出事件（用于自定义函数）
| 事件类型 | 描述 |
| :--- | :--- |
| StepOutput | 指示步骤的输出 |

#### 并行执行事件
| 事件类型 | 描述 |
| :--- | :--- |
| ParallelExecutionStarted | 表示并行步骤的开始 |
| ParallelExecutionCompleted | 表示并行步骤已成功完成 |

#### 条件执行事件
| 事件类型 | 描述 |
| :--- | :--- |
| ConditionExecutionStarted | 表示某种状态的开始 |
| ConditionExecutionCompleted | 表示该条件已成功完成 |

#### 循环执行事件
| 事件类型 | 描述 |
| :--- | :--- |
| LoopExecutionStarted | 表示循环的开始 |
| LoopIterationStartedEvent | 表示循环迭代的开始 |
| LoopIterationCompletedEvent | 表示循环迭代成功完成 |
| LoopExecutionCompleted | 表示循环成功完成 |

#### 路由器执行事件
| 事件类型 | 描述 |
| :--- | :--- |
| RouterExecutionStarted | 表示路由器已启动 |
| RouterExecutionCompleted | 路由器成功完成的信号 |


#### 步骤执行事件
| 事件类型 | 描述 |
| :--- | :--- |
| StepsExecutionStarted | 表示开始 Steps 执行 |
| StepsExecutionCompleted | Steps 表示执行成功完成 |


#### 存储事件
工作流可以自动存储所有执行事件，用于分析、调试和审计。通过筛选特定事件类型，可以在保留关键执行记录的同时，减少干扰信息和存储开销。

通过workflow.run_response.events访问存储事件，以及你工作流会话数据库（SQLite、PostgreSQL等）的运行列。

- store_events=True自动将所有工作流事件存储到数据库中
- events_to_skip=[]过滤掉特定类型的事件，以减少存储空间和噪音

通过以下方式workflow.run_response.events访问所有已存储的事件

可跳过的活动：

```python
from agno.run.workflow import WorkflowRunEvent

# Common events you might want to skip
events_to_skip = [
    WorkflowRunEvent.workflow_started,
    WorkflowRunEvent.workflow_completed,
    WorkflowRunEvent.workflow_cancelled,
    WorkflowRunEvent.step_started,
    WorkflowRunEvent.step_completed,
    WorkflowRunEvent.parallel_execution_started,
    WorkflowRunEvent.parallel_execution_completed,
    WorkflowRunEvent.condition_execution_started,
    WorkflowRunEvent.condition_execution_completed,
    WorkflowRunEvent.loop_execution_started,
    WorkflowRunEvent.loop_execution_completed,
    WorkflowRunEvent.router_execution_started,
    WorkflowRunEvent.router_execution_completed,
]
```

用例

- 调试：存储所有事件以分析工作流执行流程
- 审计跟踪：记录所有工作流程活动，以确保合规性。
- 性能分析：分析时间和执行模式
- 错误调查：审查导致故障的事件序列
- 降噪：跳过冗长的事件，step_started专注于结果

配置示例

```python
# store everything
debug_workflow = Workflow(
    name="Debug Workflow",
    store_events=True,
    steps=[...]
)

# store only important events
production_workflow = Workflow(
    name="Production Workflow",
    store_events=True,
    events_to_skip=[
        WorkflowRunEvent.step_started,
        WorkflowRunEvent.parallel_execution_started,
        # keep step_completed and workflow_completed
    ],
    steps=[...]
)

# No event storage
fast_workflow = Workflow(
    name="Fast Workflow",
    store_events=False,
    steps=[...]
)
```

### 自主遥测
Agno 日志用于模拟工作流程，以便我们优先更新最常用的提供商。您可以通过AGNO_TELEMETRY=false在环境变量或telemetry=False工作流程中进行设置来禁用此功能。

```shell
export AGNO_TELEMETRY=false
```

或者：

```python
workflow = Workflow(..., telemetry=False)
```
请参阅工作流类参考文档了解更多详情。


## 工作流程模式

掌握确定性工作流模式，包括顺序执行、并行执行、条件执行和循环执行，以实现可靠的多智能体自动化。

构建可预测的、生产就绪的工作流，以协调代理、团队和功能，实现可预测的执行模式。本指南全面涵盖所有工作流类型，从简单的顺序流程到具有并行执行和动态路由的复杂分支逻辑。

与自由形式的代理交互不同，这些模式提供了结构化的自动化，可产生一致、可重复的结果，非常适合生产系统。

### 积木
Agno Workflows 的核心构建模块包括

| 成分 | 目的 |
| :--- | :--- |
| 步 | 基本执行单元 |
| 代理人 | 具有特定角色的AI助手 |
| 团队 | 协调的代理人小组 |
| 功能 | 自定义 Python 逻辑 |
| 平行线 | 并发执行 |
| 健康）状况 | 条件执行 |
| 环形 | 迭代执行 |
| 路由器 | 动态路由 |

Agno Workflows 支持多种执行模式，这些模式可以组合起来构建复杂的自动化系统。每种模式都适用于特定的用例，并且可以组合使用以构建复杂的工作流程。

## 顺序工作流程


线性确定性过程，其中每一步都取决于前一步的输出。

顺序工作流程可确保可预测的执行顺序和步骤之间清晰的数据流。

流程示例：研究 → 数据处理 → 内容创作 → 最终审核

顺序工作流程可确保可预测的执行顺序和步骤之间清晰的数据流。

```python
from agno.workflow import Step, Workflow, StepOutput

def data_preprocessor(step_input):
    # Custom preprocessing logic

    # Or you can also run any agent/team over here itself
    # response = some_agent.run(...)
    return StepOutput(content=f"Processed: {step_input.input}") # <-- Now pass the agent/team response in content here

workflow = Workflow(
    name="Mixed Execution Pipeline",
    steps=[
        research_team,      # Team
        data_preprocessor,  # Function
        content_agent,      # Agent
    ]
)

workflow.print_response("Analyze the competitive landscape for fintech startups", markdown=True)
```

## 完全 Python 工作流程
用纯 Python 保持简洁，采用 v1 工作流风格

使用纯 Python 实现简洁：如果您更喜欢 Workflows 1.0 的方式或需要最大的灵活性，您仍然可以使用单个 Python 函数来处理所有事情。这种方式让您可以完全控制执行流程，同时还能受益于 Workflow 的存储、流式传输和会话管理等功能。

将工作流程中的所有步骤替换为一个可执行的单一函数，您可以在其中控制所有操作。

```python

from agno.workflow import Workflow, WorkflowExecutionInput

def custom_workflow_function(workflow: Workflow, execution_input: WorkflowExecutionInput):
    # Custom orchestration logic
    research_result = research_team.run(execution_input.message)
    analysis_result = analysis_agent.run(research_result.content)
    return f"Final: {analysis_result.content}"

workflow = Workflow(
    name="Function-Based Workflow",
    steps=custom_workflow_function  # Single function replaces all steps
)

workflow.print_response("Evaluate the market potential for quantum computing applications", markdown=True)
```

### 基于步骤的工作流程
AgentOS聊天页面上的日志记录和支持的具体步骤

您可以为步骤命名，以便更好地记录日志并方便日后在 Agno 平台上进行支持。此外，在对象内部访问步骤的输出时，步骤的名称也会随之更改StepInput。

```python   
from agno.workflow import Step, Workflow

# Named steps for better tracking
workflow = Workflow(
    name="Content Creation Pipeline",
    steps=[
        Step(name="Research Phase", team=researcher),
        Step(name="Analysis Phase", executor=custom_function),
        Step(name="Writing Phase", agent=writer),
    ]
)

workflow.print_response(
    "AI trends in 2024",
    markdown=True,
)
```

### 工作流中的自定义函数
如何在工作流中使用自定义函数

自定义函数提供最大的灵活性，允许您为步骤执行定义特定逻辑。您可以使用它们来预处理输入、协调代理和团队，以及后处理输出，并实现完全的程序化控制。

主要能力
- 自定义逻辑：实现复杂的业务规则和数据转换
- 代理集成：在您的自定义处理逻辑中呼叫代理和团队
- 数据流控制：转换各步骤之间的输出，以实现最佳数据处理。

实现模式： 定义一个Step带有自定义函数的接口executor。该函数必须接受一个StepInput对象并返回一个StepOutput对象，以确保与工作流系统无缝集成。

![](https://i-blog.csdnimg.cn/direct/52246f9b722f4bb9b4412ce3e45420e7.png)


#### 例子
```python

content_planning_step = Step(
    name="Content Planning Step",
    executor=custom_content_planning_function,
)

def custom_content_planning_function(step_input: StepInput) -> StepOutput:
    """
    Custom function that does intelligent content planning with context awareness
    """
    message = step_input.input
    previous_step_content = step_input.previous_step_content

    # Create intelligent planning prompt
    planning_prompt = f"""
        STRATEGIC CONTENT PLANNING REQUEST:

        Core Topic: {message}

        Research Results: {previous_step_content[:500] if previous_step_content else "No research results"}

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

        enhanced_content = f"""
            ## Strategic Content Plan

            **Planning Topic:** {message}

            **Research Integration:** {"✓ Research-based" if previous_step_content else "✗ No research foundation"}

            **Content Strategy:**
            {response.content}

            **Custom Planning Enhancements:**
            - Research Integration: {"High" if previous_step_content else "Baseline"}
            - Strategic Alignment: Optimized for multi-channel distribution
            - Execution Ready: Detailed action items included
        """.strip()

        return StepOutput(content=enhanced_content)

    except Exception as e:
        return StepOutput(
            content=f"Custom content planning failed: {str(e)}",
            success=False,
        )
```

#### 基于类的执行器
您还可以通过定义一个实现该方法的类来使用基于类的执行器__call__。
```python
class CustomExecutor:
    def __call__(self, step_input: StepInput) -> StepOutput:
        # 1. Custom preprocessing
        # 2. Call agents/teams as needed
        # 3. Custom postprocessing
        return StepOutput(content=enhanced_content)

content_planning_step = Step(
    name="Content Planning Step",
    executor=CustomExecutor(),
)

```

什么时候用得上？
- 初始化配置：创建执行器时传入设置、API 密钥或行为标志。
- 有状态执行：在多个工作流运行中维护计数器、缓存或跟踪信息
- 可重用组件：创建可在多个工作流之间共享的已配置执行器实例

```python
class CustomExecutor:
    def __init__(self, max_retries: int = 3, use_cache: bool = True):
        # Configuration passed during instantiation
        self.max_retries = max_retries
        self.use_cache = use_cache
        self.call_count = 0  # Stateful tracking

    def __call__(self, step_input: StepInput) -> StepOutput:
        self.call_count += 1

        # Access instance configuration and state
        if self.use_cache and self.call_count > 1:
            return StepOutput(content="Using cached result")

        # Your custom logic with access to self.max_retries, etc.
        return StepOutput(content=enhanced_content)

# Instantiate with specific configuration
content_planning_step = Step(
    name="Content Planning Step",
    executor=CustomExecutor(max_retries=5, use_cache=False),
)
```

也支持通过将`__call__`方法定义为异步函数来实现异步执行。

```python

class CustomExecutor:
    async def __call__(self, step_input: StepInput) -> StepOutput:
        # 1. Custom preprocessing
        # 2. Call agents/teams as needed
        # 3. Custom postprocessing
        return StepOutput(content=enhanced_content)

content_planning_step = Step(
    name="Content Planning Step",
    executor=CustomExecutor(),
)
```

### 在 AgentOS 上使用自定义函数步骤进行流式执行：
如果你在自定义函数步骤中运行代理或团队，可以在 AgentOS 聊天页面上通过调用 run（） 或 arun（） 时设置 stream=True 和 stream_events=True，从而实现事件。

```python
content_planner = Agent(
    name="Content Planner",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=[
        "Plan a content schedule over 4 weeks for the provided topic and research content",
        "Ensure that I have posts for 3 posts per week",
    ],
    db=InMemoryDb(),
)

async def custom_content_planning_function(
    step_input: StepInput,
) -> AsyncIterator[Union[WorkflowRunOutputEvent, StepOutput]]:
    """
    Custom function that does intelligent content planning with context awareness.

    Note: This function calls content_planner.arun() internally, and all events
    from that agent call will automatically get workflow context injected by
    the workflow execution system - no manual intervention required!
    """
    message = step_input.input
    previous_step_content = step_input.previous_step_content

    # Create intelligent planning prompt
    planning_prompt = f"""
        STRATEGIC CONTENT PLANNING REQUEST:

        Core Topic: {message}

        Research Results: {previous_step_content[:500] if previous_step_content else "No research results"}

        Planning Requirements:
        1. Create a comprehensive content strategy based on the research
        2. Leverage the research findings effectively
        3. Identify content formats and channels
        4. Provide timeline and priority recommendations
        5. Include engagement and distribution strategies

        Please create a detailed, actionable content plan.
    """

    try:
        response_iterator = content_planner.arun(
            planning_prompt, stream=True, stream_events=True
        )
        async for event in response_iterator:
            yield event

        response = content_planner.get_last_run_output()

        enhanced_content = f"""
            ## Strategic Content Plan

            **Planning Topic:** {message}

            **Research Integration:** {"✓ Research-based" if previous_step_content else "✗ No research foundation"}

            **Content Strategy:**
            {response.content}

            **Custom Planning Enhancements:**
            - Research Integration: {"High" if previous_step_content else "Baseline"}
            - Strategic Alignment: Optimized for multi-channel distribution
            - Execution Ready: Detailed action items included
        """.strip()

        yield StepOutput(content=enhanced_content)

    except Exception as e:
        yield StepOutput(
            content=f"Custom content planning failed: {str(e)}",
            success=False,
        )
```

### 条件工作流

基于输入分析或业务规则的确定性分支

示例用例：内容类型路由、主题特定处理、基于质量的决策

条件工作流在保持确定性执行路径的同时，提供可预测的分支逻辑。

![](https://i-blog.csdnimg.cn/direct/9d79710724ce43b78d6ceff34d00d305.png)


#### 工作原理
该类Condition会评估一个函数，并根据结果执行不同的步骤：
- 如果 branch( steps)：当评估器返回时执行True
- 否则分支( else_steps)：当评估器返回时执行False（可选）

如果条件为真False且未else_steps提供任何参数，则跳过该条件，工作流程继续进行到下一步。

#### 基本示例
```python
from agno.workflow import Condition, Step, Workflow

def is_tech_topic(step_input) -> bool:
    topic = step_input.input.lower()
    return any(keyword in topic for keyword in ["ai", "tech", "software"])

workflow = Workflow(
    name="Conditional Research",
    steps=[
        Condition(
            name="Tech Topic Check",
            evaluator=is_tech_topic,
            steps=[Step(name="Tech Research", agent=tech_researcher)]
        ),
        Step(name="General Analysis", agent=general_analyst),
    ]
)

workflow.print_response("Comprehensive analysis of AI and machine learning trends", markdown=True)
```

#### If/Else 分支
当条件为 False 时，使用 else_steps 来定义备用执行路径：
```python
from agno.workflow import Condition, Step, Workflow

def is_technical_issue(step_input) -> bool:
    text = (step_input.input or "").lower()
    tech_keywords = ["error", "bug", "crash", "not working", "api", "timeout"]
    return any(kw in text for kw in tech_keywords)

workflow = Workflow(
    name="Customer Support Router",
    steps=[
        Condition(
            name="TechnicalTriage",
            evaluator=is_technical_issue,
            # If branch: technical pipeline
            steps=[
                Step(name="Diagnose", agent=diagnostic_agent),
                Step(name="Engineer", agent=engineering_agent),
            ],
            # Else branch: general support
            else_steps=[
                Step(name="GeneralSupport", agent=general_support_agent),
            ],
        ),
        Step(name="FollowUp", agent=followup_agent),
    ],
)

# Technical query -> executes Diagnose, Engineer, then FollowUp
workflow.print_response("My app keeps crashing with a timeout error")

# Non-technical query -> executes GeneralSupport, then FollowUp
workflow.print_response("How do I change my shipping address?")
```

### 并行工作流程

独立且可并发执行的任务，可以同时执行以提高效率

示例应用场景：多源研究、并行分析、并发数据处理

并行工作流在保持确定性结果的同时，大幅缩短了独立操作的执行时间。

![](https://i-blog.csdnimg.cn/direct/ba8a8b2876b0420296b32cd08abfdcb6.png)

#### 例子

```python
from agno.workflow import Parallel, Step, Workflow

workflow = Workflow(
    name="Parallel Research Pipeline",
    steps=[
        Parallel(
            Step(name="HackerNews Research", agent=hn_researcher),
            Step(name="Web Research", agent=web_researcher),
            Step(name="Academic Research", agent=academic_researcher),
            name="Research Step"
        ),
        Step(name="Synthesis", agent=synthesizer),  # Combines the results and produces a report
    ]
)

workflow.print_response("Write about the latest AI developments", markdown=True)
```
#### 分步骤处理会话状态数据
在步骤中使用自定义 Python 函数时，可以通过run_context参数访问和更新 Workfklow 会话状态。

如果您在并行步骤中执行会话状态更新，请注意，对共享状态的并发访问需要协调以避免竞争条件。



### 迭代工作流程

以质量为导向的流程，需要重复执行直至满足特定条件。

示例用例：质量改进循环、重试机制、迭代改进

迭代工作流程提供可控的重复操作和确定的退出条件，从而确保一致的质量标准。

![](https://i-blog.csdnimg.cn/direct/ba8a8b2876b0420296b32cd08abfdcb6.png)

```python
from agno.workflow import Loop, Step, Workflow

def quality_check(outputs) -> bool:
    # Return True to break loop, False to continue
    return any(len(output.content) > 500 for output in outputs)

workflow = Workflow(
    name="Quality-Driven Research",
    steps=[
        Loop(
            name="Research Loop",
            steps=[Step(name="Deep Research", agent=researcher)],
            end_condition=quality_check,
            max_iterations=3
        ),
        Step(name="Final Analysis", agent=analyst),
    ]
)

workflow.print_response("Research the impact of renewable energy on global markets", markdown=True)
```

### 分支工作流程


需要基于内容分析进行动态路径选择的复杂决策树

示例用例：专家路由、内容类型检测、多路径处理

动态路由工作流程提供智能路径选择，同时保持每个选定分支内可预测的执行。

![](https://i-blog.csdnimg.cn/direct/e153223532844b939b5b87fe29660a21.png)

#### 选择器灵活性
路由选择器函数支持多种返回类型：

- 字符串：返回步骤名称 - 路由器从选项中解析它
- 步骤：直接返回步骤对象
- List[Step]：返回链式调用的步骤列表

选择器还可以接收step_choices一个可选的第二个参数，用于动态选择。
​
#### 示例：基于字符串的选择器
最简单的方法是——将步骤名称作为字符串返回：

```python
from typing import Union, List

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.workflow.router import Router
from agno.workflow.step import Step
from agno.workflow.types import StepInput
from agno.workflow.workflow import Workflow

tech_expert = Agent(
    name="tech_expert",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="You are a tech expert. Provide technical analysis.",
)

biz_expert = Agent(
    name="biz_expert",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="You are a business expert. Provide business insights.",
)

generalist = Agent(
    name="generalist",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="You are a generalist. Provide general information.",
)

tech_step = Step(name="Tech Research", agent=tech_expert)
business_step = Step(name="Business Research", agent=biz_expert)
general_step = Step(name="General Research", agent=generalist)


def route_by_topic(step_input: StepInput) -> Union[str, Step, List[Step]]:
    """Selector can return step name as string - Router resolves it."""
    topic = step_input.input.lower()

    if "tech" in topic or "ai" in topic or "software" in topic:
        return "Tech Research"  # Return name as string
    elif "business" in topic or "market" in topic or "finance" in topic:
        return "Business Research"
    else:
        return "General Research"


workflow = Workflow(
    name="Expert Routing",
    steps=[
        Router(
            name="Topic Router",
            selector=route_by_topic,
            choices=[tech_step, business_step, general_step],
        ),
    ],
)

workflow.print_response("Latest developments in artificial intelligence", markdown=True)
```

#### 示例：使用 step_choices 参数
动态获取可用选项，实现更灵活的路线规划：

```python
def dynamic_selector(step_input: StepInput, step_choices: list) -> Union[str, Step, List[Step]]:
    """
    Selector receives step_choices - can select by name or return Step directly.
    step_choices contains the prepared Step objects from Router.choices.
    """
    user_input = step_input.input.lower()

    # Build name map from step_choices
    step_map = {s.name: s for s in step_choices if hasattr(s, "name") and s.name}

    # Can return step name as string
    if "research" in user_input:
        return "researcher"

    # Can return Step object directly
    if "write" in user_input:
        return step_map.get("writer", step_choices[0])

    # Can return list of Steps for chaining
    if "full" in user_input:
        return [step_map["researcher"], step_map["writer"], step_map["reviewer"]]

    # Default
    return step_choices[0]


workflow = Workflow(
    name="Dynamic Routing",
    steps=[
        Router(
            name="Dynamic Router",
            selector=dynamic_selector,
            choices=[researcher, writer, reviewer],
        ),
    ],
)
```

### 分组步骤工作流程


将多个步骤组织成可重用的、逻辑清晰的序列，以构建复杂的工作流程，并明确区分关注点。

主要优势：可重用序列、更清晰的分支逻辑、模块化工作流程设计

分组步骤实现了模块化工作流程架构，具有可重用的组件和清晰的逻辑边界。

#### 基本示例

```python
from agno.workflow import Steps, Step, Workflow

# Create a reusable content creation sequence
article_creation_sequence = Steps(
    name="ArticleCreation",
    description="Complete article creation workflow from research to final edit",
    steps=[
        Step(name="research", agent=researcher),
        Step(name="writing", agent=writer),
        Step(name="editing", agent=editor),
    ],
)

# Use the sequence in a workflow
workflow = Workflow(
    name="Article Creation Workflow",
    steps=[article_creation_sequence]  # Single sequence
)

workflow.print_response("Write an article about renewable energy", markdown=True)
```

### 高级工作流程模式


结合多种工作流程模式，构建复杂且可用于生产的自动化系统

模式组合：条件逻辑 + 并行执行 + 迭代循环 + 自定义处理 + 动态路由

这个例子展示了如何组合确定性模式来创建复杂但可预测的工作流程。

```python
from agno.workflow import Condition, Loop, Parallel, Router, Step, Workflow

def research_post_processor(step_input) -> StepOutput:
    """Post-process and consolidate research data from parallel conditions"""
    research_data = step_input.previous_step_content or ""

    try:
        # Analyze research quality and completeness
        word_count = len(research_data.split())
        has_tech_content = any(keyword in research_data.lower()
                              for keyword in ["technology", "ai", "software", "tech"])
        has_business_content = any(keyword in research_data.lower()
                                  for keyword in ["market", "business", "revenue", "strategy"])

        # Create enhanced research summary
        enhanced_summary = f"""
            ## Research Analysis Report

            **Data Quality:** {"✓ High-quality" if word_count > 200 else "⚠ Limited data"}

            **Content Coverage:**
            - Technical Analysis: {"✓ Completed" if has_tech_content else "✗ Not available"}
            - Business Analysis: {"✓ Completed" if has_business_content else "✗ Not available"}

            **Research Findings:**
            {research_data}
        """.strip()

        return StepOutput(
            content=enhanced_summary,
            success=True,
        )

    except Exception as e:
        return StepOutput(
            content=f"Research post-processing failed: {str(e)}",
            success=False,
            error=str(e)
        )

# Complex workflow combining multiple patterns
workflow = Workflow(
    name="Advanced Multi-Pattern Workflow",
    steps=[
        Parallel(
            Condition(
                name="Tech Check",
                evaluator=is_tech_topic,
                steps=[Step(name="Tech Research", agent=tech_researcher)]
            ),
            Condition(
                name="Business Check",
                evaluator=is_business_topic,
                steps=[
                    Loop(
                        name="Deep Business Research",
                        steps=[Step(name="Market Research", agent=market_researcher)],
                        end_condition=research_quality_check,
                        max_iterations=3
                    )
                ]
            ),
            name="Conditional Research Phase"
        ),
        Step(
            name="Research Post-Processing",
            executor=research_post_processor,
            description="Consolidate and analyze research findings with quality metrics"
        ),
        Router(
            name="Content Type Router",
            selector=content_type_selector,
            choices=[blog_post_step, social_media_step, report_step]
        ),
        Step(name="Final Review", agent=reviewer),
    ]
)

workflow.print_response("Create a comprehensive analysis of sustainable technology trends and their business impact for 2024", markdown=True)
```


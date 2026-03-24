# 用法
## 核心模式

使用命名步骤进行顺序执行，并进行清晰的跟踪。

此示例演示了如何使用命名步骤对象来更好地跟踪和组织工作流。这种模式在保持简单顺序执行的同时，提供了清晰的步骤标识和增强的日志记录。


## 模式：顺序命名步骤
适用场景：线性流程，需要清晰的步骤标识、更完善的日志记录以及对未来平台的支持。尤其适用于具有明显阶段且需要命名的情况。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.workflow.step import Step
from agno.workflow.workflow import Workflow

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

# Define steps
research_step = Step(
    name="Research Step",
    team=research_team,
)

content_planning_step = Step(
    name="Content Planning Step",
    agent=content_planner,
)

content_creation_workflow = Workflow(
    name="Content Creation Workflow",
    description="Automated content creation from blog posts to social media",
    db=SqliteDb(
        session_table="workflow_session",
        db_file="tmp/workflow.db",
    ),
    steps=[research_step, content_planning_step],
)

# Create and use workflow
if __name__ == "__main__":
    content_creation_workflow.print_response(
        input="AI trends in 2024",
        markdown=True,
    )
```

## 并行工作流程

同时执行独立任务以减少总执行时间。

此示例演示了Workflows 2.0如何并行执行可同时运行的独立任务。它展示了如何通过并行执行非依赖步骤来优化工作流性能，从而显著缩短总执行时间。

适用场景：当您有多个独立任务，这些任务彼此独立，互不依赖，但都能为实现同一最终目标做出贡献时。非常适合来自多个数据源的研究、并行数据处理，或任何任务可以同时运行的场景。

```python

from agno.agent import Agent
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.workflow import Step, Workflow
from agno.workflow.parallel import Parallel

# Create agents
news_researcher = Agent(name="News Researcher", tools=[HackerNewsTools()])
finance_researcher = Agent(name="Finance Researcher", tools=[YFinanceTools()])
writer = Agent(name="Writer")
reviewer = Agent(name="Reviewer")

# Create individual steps
research_news_step = Step(name="Research News", agent=news_researcher)
research_finance_step = Step(name="Research Finance", agent=finance_researcher)
write_step = Step(name="Write Article", agent=writer)
review_step = Step(name="Review Article", agent=reviewer)

# Create workflow with parallel research
workflow = Workflow(
    name="Content Creation Pipeline",
    steps=[
        Parallel(research_news_step, research_finance_step, name="Research Phase"),
        write_step,
        review_step,
    ],
)

workflow.print_response("Write about the latest AI developments and stock trends")
```

## 条件工作流

根据内容分析或业务逻辑执行步骤。

本示例演示了Workflows 2.0 的条件执行模式。它展示了如何基于内容分析有条件地执行步骤，从而根据实际处理的数据智能地选择步骤。

适用场景：当您需要基于内容分析而非简单的输入参数或其他业务逻辑来智能选择步骤时。非常适合用于质量门控、特定内容处理或响应中间结果的自适应工作流。

```python
from agno.agent import Agent
from agno.tools.hackernews import HackerNewsTools
from agno.workflow.condition import Condition
from agno.workflow.step import Step
from agno.workflow.types import StepInput
from agno.workflow.workflow import Workflow

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

if __name__ == "__main__":
    print("Running Basic Linear Workflow Example")
    print("=" * 50)

    try:
        basic_workflow.print_response(
            input="Recent breakthroughs in quantum computing",
            stream=True,
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
```

## 循环步骤工作流程

此示例演示了工作流 2.0循环执行，以实现质量驱动的迭代流程。

此示例演示了工作流 2.0如何反复执行步骤，直到满足特定条件，从而确保在进行内容创作之前有足够的调研深度。
适用场景：当您需要迭代优化、质量保证，或单次执行无法保证达到所需输出质量时。尤其适用于研究资料收集、数据采集，或任何以内容分析而非固定迭代次数来判断“足够好”的流程。

```python

from typing import List

from agno.agent import Agent
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.workflow import Loop, Step, Workflow
from agno.workflow.types import StepOutput

# Create agents for research
research_agent = Agent(
    name="Research Agent",
    role="Research specialist",
    tools=[HackerNewsTools(), YFinanceTools()],
    instructions="You are a research specialist. Research the given topic thoroughly.",
    markdown=True,
)

content_agent = Agent(
    name="Content Agent",
    role="Content creator",
    instructions="You are a content creator. Create engaging content based on research.",
    markdown=True,
)

# Create research steps
research_hackernews_step = Step(
    name="Research HackerNews",
    agent=research_agent,
    description="Research trending topics on HackerNews",
)

research_web_step = Step(
    name="Research Web",
    agent=research_agent,
    description="Research additional information from web sources",
)

content_step = Step(
    name="Create Content",
    agent=content_agent,
    description="Create content based on research findings",
)


# End condition function
def research_evaluator(outputs: List[StepOutput]) -> bool:
    """
    Evaluate if research results are sufficient
    Returns True to break the loop, False to continue
    """
    # Check if any outputs are present
    if not outputs:
        return False

    # Check if any output contains substantial content
    for output in outputs:
        if output.content and len(output.content) > 200:
            print(
                f"Research evaluation passed - found substantial content ({len(output.content)} chars)"
            )
            return True

    print("Research evaluation failed - need more substantial research")
    return False


# Create workflow with loop
workflow = Workflow(
    name="Research and Content Workflow",
    description="Research topics in a loop until conditions are met, then create content",
    steps=[
        Loop(
            name="Research Loop",
            steps=[research_hackernews_step, research_web_step],
            end_condition=research_evaluator,
            max_iterations=3,  # Maximum 3 iterations
        ),
        content_step,
    ],
)

if __name__ == "__main__":
    # Test the workflow
    workflow.print_response(
        input="Research the latest trends in AI and machine learning, then create a summary",
    )
```

## 循环迭代累积

演示循环迭代如何将前一次迭代的输出传递下去。

每次迭代都通过接收前一次迭代的输出`step_input.get_last_step_content()`，从而实现累积、细化和收敛等迭代处理模式。
本例每次迭代将数值递增 10，当数值达到 50 或更大时停止。

```python
from agno.workflow import Loop, Step, Workflow
from agno.workflow.types import StepInput, StepOutput


def increment_executor(step_input: StepInput) -> StepOutput:
    """Increment the previous step's numeric content by 10."""
    last_content = step_input.get_last_step_content()
    if last_content and last_content.isdigit():
        new_value = int(last_content) + 10
        return StepOutput(content=str(new_value))
    return StepOutput(content="0")


workflow = Workflow(
    name="Iterative Accumulation Workflow",
    steps=[
        Step(
            name="Initial Value",
            executor=lambda step_input: StepOutput(content=step_input.input),
        ),
        Loop(
            name="Increment Loop",
            steps=[
                Step(
                    name="Increment Step",
                    executor=increment_executor,
                )
            ],
            end_condition=lambda step_outputs: int(step_outputs[-1].content) >= 50,
            max_iterations=10,
            forward_iteration_output=True,  # Default: each iteration gets previous output
        ),
    ],
)

if __name__ == "__main__":
    # Starting from 35: iteration 1 -> 45, iteration 2 -> 55 (>= 50, stops)
    workflow.print_response("35")
```

## 条件分支工作流
此示例演示了Workflows 2.0路由模式，用于智能的、基于内容的流程路由。

此示例演示了Workflows 2.0如何根据输入分析动态选择最佳执行路径，从而实现自适应工作流，为每个主题选择最佳策略。

何时使用：当您需要基于业务逻辑的互斥执行路径时。非常适合特定主题的工作流、专家路由，或不同主题需要完全不同的处理策略的情况。与可以触发多个并行路径的条件不同，路由只会选择一条路径。


```python
from typing import List

from agno.agent.agent import Agent
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.workflow.router import Router
from agno.workflow.step import Step
from agno.workflow.types import StepInput
from agno.workflow.workflow import Workflow

# Define the research agents
hackernews_agent = Agent(
    name="HackerNews Researcher",
    instructions="You are a researcher specializing in finding the latest tech news and discussions from Hacker News. Focus on startup trends, programming topics, and tech industry insights.",
    tools=[HackerNewsTools()],
)

finance_agent = Agent(
    name="Finance Researcher",
    instructions="You are a finance researcher. Search for stock data, market trends, and financial information to gather detailed insights.",
    tools=[YFinanceTools()],
)

content_agent = Agent(
    name="Content Publisher",
    instructions="You are a content creator who takes research data and creates engaging, well-structured articles. Format the content with proper headings, bullet points, and clear conclusions.",
)

# Create the research steps
research_hackernews = Step(
    name="research_hackernews",
    agent=hackernews_agent,
    description="Research latest tech trends from Hacker News",
)

research_finance = Step(
    name="research_finance",
    agent=finance_agent,
    description="Research financial data and market trends",
)

publish_content = Step(
    name="publish_content",
    agent=content_agent,
    description="Create and format final content for publication",
)


# Now returns Step(s) to execute
def research_router(step_input: StepInput) -> List[Step]:
    """
    Decide which research method to use based on the input topic.
    Returns a list containing the step(s) to execute.
    """
    # Use the original workflow message if this is the first step
    topic = step_input.previous_step_content or step_input.input or ""
    topic = topic.lower()

    # Check if the topic is tech/startup related - use HackerNews
    tech_keywords = [
        "startup",
        "programming",
        "ai",
        "machine learning",
        "software",
        "developer",
        "coding",
        "tech",
        "silicon valley",
        "venture capital",
        "cryptocurrency",
        "blockchain",
        "open source",
        "github",
    ]

    if any(keyword in topic for keyword in tech_keywords):
        print(f"Tech topic detected: Using HackerNews research for '{topic}'")
        return [research_hackernews]
    else:
        print(f"Finance topic detected: Using finance research for '{topic}'")
        return [research_finance]


workflow = Workflow(
    name="Intelligent Research Workflow",
    description="Automatically selects the best research method based on topic, then publishes content",
    steps=[
        Router(
            name="research_strategy_router",
            selector=research_router,
            choices=[research_hackernews, research_finance],
            description="Intelligently selects research method based on topic",
        ),
        publish_content,
    ],
)

if __name__ == "__main__":
    workflow.print_response(
        "Latest developments in artificial intelligence and machine learning"
    )

```

## 步骤与功能
本示例演示了如何使用带自定义函数执行器的命名步骤。

本示例演示了如何使用命名步骤对象和自定义函数执行器来构建工作流。这种模式结合了命名步骤的优势和自定义函数的灵活性，从而可以在结构化的工作流步骤中实现复杂的数据处理。

何时使用：当您需要命名步骤组织结构，但又希望实现超出代理/团队功能范围的自定义逻辑时。非常适合复杂的数据处理、多步骤操作，或者需要在单个步骤中协调多个代理的情况。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
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
    instructions="Get stock prices and financial data",
)

# Define research team for complex analysis
research_team = Team(
    name="Research Team",
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

            **Research Integration:** {"Research-based" if previous_step_content else "No research foundation"}

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


# Define steps using different executor types

research_step = Step(
    name="Research Step",
    team=research_team,
)

content_planning_step = Step(
    name="Content Planning Step",
    executor=custom_content_planning_function,
)


# Define and use examples
if __name__ == "__main__":
    content_creation_workflow = Workflow(
        name="Content Creation Workflow",
        description="Automated content creation with custom execution options",
        db=SqliteDb(
            session_table="workflow_session",
            db_file="tmp/workflow.db",
        ),
        # Define the sequence of steps
        # First run the research_step, then the content_planning_step
        # You can mix and match agents, teams, and even regular python functions directly as steps
        steps=[research_step, content_planning_step],
    )
    content_creation_workflow.print_response(
        input="AI trends in 2024",
        markdown=True,
    )

    print("\n" + "=" * 60 + "\n")
```



## 函数代替步骤

本示例演示了如何在工作流程中仅使用单个函数而不是多个步骤。

此示例演示了如何使用单个自定义执行函数而非离散步骤来实现工作流。这种模式使您能够完全控制编排逻辑，同时还能受益于工作流的存储、流式传输和会话管理等功能。

何时使用：当您需要对执行流程进行最大程度的灵活性和控制时，类似于工作流 1.0 的方法，但结构化程度更高。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.workflow.types import WorkflowExecutionInput
from agno.workflow.workflow import Workflow

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


def custom_execution_function(
    workflow: Workflow, execution_input: WorkflowExecutionInput
):
    print(f"Executing workflow: {workflow.name}")

    # Run the research team
    run_response = research_team.run(execution_input.input)
    research_content = run_response.content

    # Create intelligent planning prompt
    planning_prompt = f"""
        STRATEGIC CONTENT PLANNING REQUEST:

        Core Topic: {execution_input.input}

        Research Results: {research_content[:500]}

        Planning Requirements:
        1. Create a comprehensive content strategy based on the research
        2. Leverage the research findings effectively
        3. Identify content formats and channels
        4. Provide timeline and priority recommendations
        5. Include engagement and distribution strategies

        Please create a detailed, actionable content plan.
    """
    content_plan = content_planner.run(planning_prompt)

    # Return the content plan
    return content_plan.content


# Create and use workflow
if __name__ == "__main__":
    content_creation_workflow = Workflow(
        name="Content Creation Workflow",
        description="Automated content creation from blog posts to social media",
        db=SqliteDb(
            session_table="workflow_session",
            db_file="tmp/workflow.db",
        ),
        steps=custom_execution_function,
    )
    content_creation_workflow.print_response(
        input="AI trends in 2024",
    )
```

## 基于类的执行器

本示例演示如何在工作流中使用基于类的执行器。

本示例演示如何在工作流中使用基于类的执行器。


```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
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
    instructions="Get stock prices and financial data",
)

# Define research team for complex analysis
research_team = Team(
    name="Research Team",
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


# Define executor using a class.
# Make sure you define the `__call__` method that receives
# the `step_input`.
class CustomContentPlanning:
    def __call__(self, step_input: StepInput) -> StepOutput:
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
                **Research Integration:** {"Research-based" if previous_step_content else "No research foundation"}
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


# Define steps using different executor types

research_step = Step(
    name="Research Step",
    team=research_team,
)

content_planning_step = Step(
    name="Content Planning Step",
    executor=CustomContentPlanning(),
)


# Define and use examples
if __name__ == "__main__":
    content_creation_workflow = Workflow(
        name="Content Creation Workflow",
        description="Automated content creation with custom execution options",
        db=SqliteDb(
            session_table="workflow_session",
            db_file="tmp/workflow.db",
        ),
        # Define the sequence of steps
        # First run the research_step, then the content_planning_step
        # You can mix and match agents, teams, and even regular python functions directly as steps
        steps=[research_step, content_planning_step],
    )
    content_creation_workflow.print_response(
        input="AI trends in 2024",
        markdown=True,
    )

    print("\n" + "=" * 60 + "\n")
```

## 条件及步骤列表
本示例演示如何使用条件步骤并行执行多个步骤。

此示例演示了Workflows 2.0 的高级条件执行功能，其中条件可以触发多个步骤并并行运行。它展示了如何基于内容分析创建具有复杂多步骤序列的复杂分支逻辑。

适用场景：当不同主题或内容类型需要完全不同的处理流程时。尤其适用于自适应工作流程，其中研究方法应根据主题或复杂性要求而变化。

```python
from agno.agent.agent import Agent
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.workflow.condition import Condition
from agno.workflow.parallel import Parallel
from agno.workflow.step import Step
from agno.workflow.types import StepInput
from agno.workflow.workflow import Workflow

# === AGENTS ===
hackernews_agent = Agent(
    name="HackerNews Researcher",
    instructions="Research tech news and trends from Hacker News",
    tools=[HackerNewsTools()],
)

finance_agent = Agent(
    name="Finance Researcher",
    instructions="Research financial data and market trends",
    tools=[YFinanceTools()],
)

content_agent = Agent(
    name="Content Creator",
    instructions="Create well-structured content from research data",
)

# Additional agents for multi-step condition
trend_analyzer_agent = Agent(
    name="Trend Analyzer",
    instructions="Analyze trends and patterns from research data",
)

fact_checker_agent = Agent(
    name="Fact Checker",
    instructions="Verify facts and cross-reference information",
)

# === RESEARCH STEPS ===
research_hackernews_step = Step(
    name="ResearchHackerNews",
    description="Research tech news from Hacker News",
    agent=hackernews_agent,
)

research_finance_step = Step(
    name="ResearchFinance",
    description="Research financial data",
    agent=finance_agent,
)

# === MULTI-STEP CONDITION STEPS ===
deep_finance_analysis_step = Step(
    name="DeepFinanceAnalysis",
    description="Conduct deep analysis of financial data",
    agent=finance_agent,
)

trend_analysis_step = Step(
    name="TrendAnalysis",
    description="Analyze trends and patterns from the research data",
    agent=trend_analyzer_agent,
)

fact_verification_step = Step(
    name="FactVerification",
    description="Verify facts and cross-reference information",
    agent=fact_checker_agent,
)

# === FINAL STEPS ===
write_step = Step(
    name="WriteContent",
    description="Write the final content based on research",
    agent=content_agent,
)


# === CONDITION EVALUATORS ===
def check_if_we_should_search_hn(step_input: StepInput) -> bool:
    """Check if we should search Hacker News"""
    topic = step_input.input or step_input.previous_step_content or ""
    tech_keywords = [
        "ai",
        "machine learning",
        "programming",
        "software",
        "tech",
        "startup",
        "coding",
    ]
    return any(keyword in topic.lower() for keyword in tech_keywords)


def check_if_comprehensive_research_needed(step_input: StepInput) -> bool:
    """Check if comprehensive multi-step research is needed"""
    topic = step_input.input or step_input.previous_step_content or ""
    comprehensive_keywords = [
        "comprehensive",
        "detailed",
        "thorough",
        "in-depth",
        "complete analysis",
        "full report",
        "extensive research",
    ]
    return any(keyword in topic.lower() for keyword in comprehensive_keywords)


if __name__ == "__main__":
    workflow = Workflow(
        name="Conditional Workflow with Multi-Step Condition",
        steps=[
            Parallel(
                Condition(
                    name="HackerNewsCondition",
                    description="Check if we should search Hacker News for tech topics",
                    evaluator=check_if_we_should_search_hn,
                    steps=[research_hackernews_step],  # Single step
                ),
                Condition(
                    name="ComprehensiveResearchCondition",
                    description="Check if comprehensive multi-step research is needed",
                    evaluator=check_if_comprehensive_research_needed,
                    steps=[  # Multiple steps
                        deep_finance_analysis_step,
                        trend_analysis_step,
                        fact_verification_step,
                    ],
                ),
                name="ConditionalResearch",
                description="Run conditional research steps in parallel",
            ),
            write_step,
        ],
    )

    try:
        workflow.print_response(
            input="Comprehensive analysis of climate change research",
            stream=True,
        )
    except Exception as e:
        print(f"Error: {e}")
    print()
```

## 具有并行步骤的循环工作流程

此示例演示了Workflows 2.0最复杂的模式，该模式将循环执行与并行处理和实时流式传输相结合。

本示例展示了如何创建迭代工作流，在每次迭代中同时执行多个独立任务，从而优化质量和性能。

适用场景：当您需要通过并行任务执行迭代式质量改进时。尤其适用于包含多个独立任务、且这些任务共同影响整体质量的综合性研究工作流程，并且您需要重复此过程直至达到质量阈值。

```python

from typing import List

from agno.agent import Agent
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.workflow import Loop, Parallel, Step, Workflow
from agno.workflow.types import StepOutput

# Create agents for research
research_agent = Agent(
    name="Research Agent",
    role="Research specialist",
    tools=[HackerNewsTools(), YFinanceTools()],
    instructions="You are a research specialist. Research the given topic thoroughly.",
    markdown=True,
)

analysis_agent = Agent(
    name="Analysis Agent",
    role="Data analyst",
    instructions="You are a data analyst. Analyze and summarize research findings.",
    markdown=True,
)

content_agent = Agent(
    name="Content Agent",
    role="Content creator",
    instructions="You are a content creator. Create engaging content based on research.",
    markdown=True,
)

# Create research steps
research_hackernews_step = Step(
    name="Research HackerNews",
    agent=research_agent,
    description="Research trending topics on HackerNews",
)

research_web_step = Step(
    name="Research Web",
    agent=research_agent,
    description="Research additional information from web sources",
)

# Create analysis steps
trend_analysis_step = Step(
    name="Trend Analysis",
    agent=analysis_agent,
    description="Analyze trending patterns in the research",
)

sentiment_analysis_step = Step(
    name="Sentiment Analysis",
    agent=analysis_agent,
    description="Analyze sentiment and opinions from the research",
)

content_step = Step(
    name="Create Content",
    agent=content_agent,
    description="Create content based on research findings",
)


# End condition function
def research_evaluator(outputs: List[StepOutput]) -> bool:
    """
    Evaluate if research results are sufficient
    Returns True to break the loop, False to continue
    """
    # Check if we have good research results
    if not outputs:
        return False

    # Calculate total content length from all outputs
    total_content_length = sum(len(output.content or "") for output in outputs)

    # Check if we have substantial content (more than 500 chars total)
    if total_content_length > 500:
        print(
            f"Research evaluation passed - found substantial content ({total_content_length} chars total)"
        )
        return True

    print(
        f"Research evaluation failed - need more substantial research (current: {total_content_length} chars)"
    )
    return False


# Create workflow with loop containing parallel steps
workflow = Workflow(
    name="Advanced Research and Content Workflow",
    description="Research topics with parallel execution in a loop until conditions are met, then create content",
    steps=[
        Loop(
            name="Research Loop with Parallel Execution",
            steps=[
                Parallel(
                    research_hackernews_step,
                    research_web_step,
                    trend_analysis_step,
                    name="Parallel Research & Analysis",
                    description="Execute research and analysis in parallel for efficiency",
                ),
                sentiment_analysis_step,
            ],
            end_condition=research_evaluator,
            max_iterations=3,  # Maximum 3 iterations
        ),
        content_step,
    ],
)

if __name__ == "__main__":
    workflow.print_response(
        input="Research the latest trends in AI and machine learning, then create a summary",
        stream=True,
    )
```


## 条件和并行步骤工作流程

此示例演示了Workflows 2.0 的高级模式，该模式将条件执行与并行处理相结合。

本示例展示了如何创建复杂的工作流程，其中多个条件同时进行评估，每个条件都可能根据全面的内容分析触发不同的研究策略。
适用场景：当您需要进行全面、多维度的内容分析，且输入数据的不同方面可能同时触发不同的专业研究流程时。非常适合能够根据各种内容特征利用多个数据源的自适应研究工作流程。

```python
from agno.agent import Agent
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.workflow.condition import Condition
from agno.workflow.parallel import Parallel
from agno.workflow.step import Step
from agno.workflow.types import StepInput
from agno.workflow.workflow import Workflow

# === AGENTS ===
hackernews_agent = Agent(
    name="HackerNews Researcher",
    instructions="Research tech news and trends from Hacker News",
    tools=[HackerNewsTools()],
)

finance_agent = Agent(
    name="Finance Researcher",
    instructions="Research financial data and market trends",
    tools=[YFinanceTools()],
)

content_agent = Agent(
    name="Content Creator",
    instructions="Create well-structured content from research data",
)

# === RESEARCH STEPS ===
research_hackernews_step = Step(
    name="ResearchHackerNews",
    description="Research tech news from Hacker News",
    agent=hackernews_agent,
)

research_finance_step = Step(
    name="ResearchFinance",
    description="Research financial data and trends",
    agent=finance_agent,
)

prepare_input_for_write_step = Step(
    name="PrepareInput",
    description="Prepare and organize research data for writing",
    agent=content_agent,
)

write_step = Step(
    name="WriteContent",
    description="Write the final content based on research",
    agent=content_agent,
)


# === CONDITION EVALUATORS ===
def check_if_we_should_search_hn(step_input: StepInput) -> bool:
    """Check if we should search Hacker News"""
    topic = step_input.input or step_input.previous_step_content or ""
    tech_keywords = [
        "ai",
        "machine learning",
        "programming",
        "software",
        "tech",
        "startup",
        "coding",
    ]
    return any(keyword in topic.lower() for keyword in tech_keywords)


def check_if_we_should_search_finance(step_input: StepInput) -> bool:
    """Check if we should search for financial data"""
    topic = step_input.input or step_input.previous_step_content or ""
    finance_keywords = ["stock", "market", "finance", "investment", "trading", "price"]
    return any(keyword in topic.lower() for keyword in finance_keywords)


if __name__ == "__main__":
    workflow = Workflow(
        name="Conditional Workflow",
        steps=[
            Parallel(
                Condition(
                    name="HackerNewsCondition",
                    description="Check if we should search Hacker News for tech topics",
                    evaluator=check_if_we_should_search_hn,
                    steps=[research_hackernews_step],
                ),
                Condition(
                    name="FinanceCondition",
                    description="Check if we should search for financial data",
                    evaluator=check_if_we_should_search_finance,
                    steps=[research_finance_step],
                ),
                name="ConditionalResearch",
                description="Run conditional research steps in parallel",
            ),
            prepare_input_for_write_step,
            write_step,
        ],
    )

    try:
        workflow.print_response(
            input="Latest AI developments in machine learning",
            stream=True,
        )
    except Exception as e:
        print(f"Error: {e}")
    print()
```

## 条件和并行步骤工作流程
此示例演示了Workflows 2.0 的高级模式，该模式将条件执行与并行处理相结合。

本示例展示了如何创建复杂的工作流程，其中多个条件同时进行评估，每个条件都可能根据全面的内容分析触发不同的研究策略。

适用场景：当您需要进行全面、多维度的内容分析，且输入数据的不同方面可能同时触发不同的专业研究流程时。非常适合能够根据各种内容特征利用多个数据源的自适应研究工作流程。

```python
from agno.agent import Agent
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.workflow.condition import Condition
from agno.workflow.parallel import Parallel
from agno.workflow.step import Step
from agno.workflow.types import StepInput
from agno.workflow.workflow import Workflow

# === AGENTS ===
hackernews_agent = Agent(
    name="HackerNews Researcher",
    instructions="Research tech news and trends from Hacker News",
    tools=[HackerNewsTools()],
)

finance_agent = Agent(
    name="Finance Researcher",
    instructions="Research financial data and market trends",
    tools=[YFinanceTools()],
)

content_agent = Agent(
    name="Content Creator",
    instructions="Create well-structured content from research data",
)

# === RESEARCH STEPS ===
research_hackernews_step = Step(
    name="ResearchHackerNews",
    description="Research tech news from Hacker News",
    agent=hackernews_agent,
)

research_finance_step = Step(
    name="ResearchFinance",
    description="Research financial data and trends",
    agent=finance_agent,
)

prepare_input_for_write_step = Step(
    name="PrepareInput",
    description="Prepare and organize research data for writing",
    agent=content_agent,
)

write_step = Step(
    name="WriteContent",
    description="Write the final content based on research",
    agent=content_agent,
)


# === CONDITION EVALUATORS ===
def check_if_we_should_search_hn(step_input: StepInput) -> bool:
    """Check if we should search Hacker News"""
    topic = step_input.input or step_input.previous_step_content or ""
    tech_keywords = [
        "ai",
        "machine learning",
        "programming",
        "software",
        "tech",
        "startup",
        "coding",
    ]
    return any(keyword in topic.lower() for keyword in tech_keywords)


def check_if_we_should_search_finance(step_input: StepInput) -> bool:
    """Check if we should search for financial data"""
    topic = step_input.input or step_input.previous_step_content or ""
    finance_keywords = ["stock", "market", "finance", "investment", "trading", "price"]
    return any(keyword in topic.lower() for keyword in finance_keywords)


if __name__ == "__main__":
    workflow = Workflow(
        name="Conditional Workflow",
        steps=[
            Parallel(
                Condition(
                    name="HackerNewsCondition",
                    description="Check if we should search Hacker News for tech topics",
                    evaluator=check_if_we_should_search_hn,
                    steps=[research_hackernews_step],
                ),
                Condition(
                    name="FinanceCondition",
                    description="Check if we should search for financial data",
                    evaluator=check_if_we_should_search_finance,
                    steps=[research_finance_step],
                ),
                name="ConditionalResearch",
                description="Run conditional research steps in parallel",
            ),
            prepare_input_for_write_step,
            write_step,
        ],
    )

    try:
        workflow.print_response(
            input="Latest AI developments in machine learning",
            stream=True,
        )
    except Exception as e:
        print(f"Error: {e}")
    print()
```

## 带循环步骤的路由器
此示例演示了Workflows 2.0高级模式，该模式将基于路由的智能路径选择与循环执行相结合，以实现迭代质量改进。

这种高级模式结合了智能路径选择和迭代执行，使工作流能够根据任务复杂度动态地在简单路径和深度迭代循环之间进行选择。它尤其适用于处理深度需要与内容复杂度相匹配的自适应工作流。

### 工作原理
- 主题分析：路由器评估输入的复杂性，以确定合适的研究策略。
- 路径选择：简单的主题会被路由到单个研究步骤，而复杂的主题会触发专门的研究步骤Loop。
- 迭代式深度研究：该Loop步骤执行多轮研究，并在每次迭代中检查质量标准。
- 统一输出：无论采用何种路径（单步或循环），结果都会流入最终发布步骤，以实现一致的格式。


### 例子
可在快速金融研究和深度迭代技术研究之间切换的自适应研究工作流程

```python
from typing import List

from agno.agent.agent import Agent
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.workflow.loop import Loop
from agno.workflow.router import Router
from agno.workflow.step import Step
from agno.workflow.types import StepInput, StepOutput
from agno.workflow.workflow import Workflow

# Define the research agents
hackernews_agent = Agent(
    name="HackerNews Researcher",
    instructions="You are a researcher specializing in finding the latest tech news and discussions from Hacker News. Focus on startup trends, programming topics, and tech industry insights.",
    tools=[HackerNewsTools()],
)

finance_agent = Agent(
    name="Finance Researcher",
    instructions="You are a finance researcher. Search for stock data, market trends, and financial information to gather detailed insights.",
    tools=[YFinanceTools()],
)

content_agent = Agent(
    name="Content Publisher",
    instructions="You are a content creator who takes research data and creates engaging, well-structured articles. Format the content with proper headings, bullet points, and clear conclusions.",
)

# Create the research steps
research_hackernews = Step(
    name="research_hackernews",
    agent=hackernews_agent,
    description="Research latest tech trends from Hacker News",
)

research_finance = Step(
    name="research_finance",
    agent=finance_agent,
    description="Research financial data and market trends",
)

publish_content = Step(
    name="publish_content",
    agent=content_agent,
    description="Create and format final content for publication",
)

# End condition function for the loop


def research_quality_check(outputs: List[StepOutput]) -> bool:
    """
    Evaluate if research results are sufficient
    Returns True to break the loop, False to continue
    """
    if not outputs:
        return False

    # Check if any output contains substantial content
    for output in outputs:
        if output.content and len(output.content) > 300:
            print(
                f"Research quality check passed - found substantial content ({len(output.content)} chars)"
            )
            return True

    print("Research quality check failed - need more substantial research")
    return False


# Create a Loop step for deep tech research
deep_tech_research_loop = Loop(
    name="Deep Tech Research Loop",
    steps=[research_hackernews],
    end_condition=research_quality_check,
    max_iterations=3,
    description="Perform iterative deep research on tech topics",
)

# Router function that selects between simple web research or deep tech research loop


def research_strategy_router(step_input: StepInput) -> List[Step]:
    """
    Decide between simple web research or deep tech research loop based on the input topic.
    Returns either a single web research step or a tech research loop.
    """
    # Use the original workflow message if this is the first step
    topic = step_input.previous_step_content or step_input.input or ""
    topic = topic.lower()

    # Check if the topic requires deep tech research
    deep_tech_keywords = [
        "startup trends",
        "ai developments",
        "machine learning research",
        "programming languages",
        "developer tools",
        "silicon valley",
        "venture capital",
        "cryptocurrency analysis",
        "blockchain technology",
        "open source projects",
        "github trends",
        "tech industry",
        "software engineering",
        "writers",
    ]

    # Check if it's a complex tech topic that needs deep research
    if any(keyword in topic for keyword in deep_tech_keywords) or (
        "tech" in topic and len(topic.split()) > 3
    ):
        print(
            f"Deep tech topic detected: Using iterative research loop for '{topic}'"
        )
        return [deep_tech_research_loop]
    else:
        print(f"Finance topic detected: Using finance research for '{topic}'")
        return [research_finance]


workflow = Workflow(
    name="Adaptive Research Workflow",
    description="Intelligently selects between simple web research or deep iterative tech research based on topic complexity",
    steps=[
        Router(
            name="research_strategy_router",
            selector=research_strategy_router,
            choices=[research_finance, deep_tech_research_loop],
            description="Chooses between simple web research or deep tech research loop",
        ),
        publish_content,
    ],
)

if __name__ == "__main__":
    print("=== Testing with deep tech topic ===")
    workflow.print_response(
        "Latest developments in artificial intelligence and machine learning and deep tech research trends"
    )

```

## 带步骤选择的路由器

路由 step_choices 参数和字符串、步骤以及 List[Step] 返回值。

该step_choices参数为动态路由提供了极大的灵活性，允许路由选择器返回字符串、步骤对象或步骤列表以进行链式执行。它支持在运行时检查可用路径而非硬编码的复杂逻辑。

### 主要优势
当您的路由逻辑需要灵活的返回类型或对可用选项的动态访问时，step_choices它是合适的工具。
- 灵活返回值：根据需要返回字符串名称、步骤对象或列表
- 动态选择：用于step_choices在运行时检查可用选项。
- 步骤链：返回List[Step]以按顺序执行多个步骤
- 嵌套容器：在分组顺序执行的选择中使用嵌套列表
- 可重用逻辑：构建能够适应任何选择集的路由函数


### 选择器返回类型汇总
| 返回类型 | 描述 | 例子 |
| :--- | :--- | :--- |
| str | 步骤名称－路由器从选项中解析 | return＂Tech Research＂ |
| Step | 步骤对象直接 | return step＿map［＂writer＂］ |
| List［Step］ | 连锁多个步骤 | return［researcher，writer，reviewer］ |


## 数据和 I/O
访问多个先前步骤的输出

此示例演示了Workflows 2.0 的高级数据流功能。

本示例演示了工作流 2.0如何执行以下操作：
- 访问特定命名步骤的输出（get_step_content()）
- 汇总所有先前的输出（get_all_previous_content()）
- 结合多种研究资料，撰写综合报告。

### 主要特点：
- 步骤输出访问：按名称或集合检索任何先前步骤中的数据。
- 自定义报告：合并和分析并行或顺序步骤的输出结果。
- 流媒体支持：执行过程中实时更新。

```python
from agno.agent.agent import Agent
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.workflow.step import Step
from agno.workflow.types import StepInput, StepOutput
from agno.workflow.workflow import Workflow

# Define the research agents
hackernews_agent = Agent(
    name="HackerNews Researcher",
    instructions="You are a researcher specializing in finding the latest tech news and discussions from Hacker News. Focus on startup trends, programming topics, and tech industry insights.",
    tools=[HackerNewsTools()],
)

finance_agent = Agent(
    name="Finance Researcher",
    instructions="You are a finance researcher. Search for stock data, market trends, and financial information to gather detailed insights.",
    tools=[YFinanceTools()],
)

reasoning_agent = Agent(
    name="Reasoning Agent",
    instructions="You are an expert analyst who creates comprehensive reports by analyzing and synthesizing information from multiple sources. Create well-structured, insightful reports.",
)

# Create the research steps
research_hackernews = Step(
    name="research_hackernews",
    agent=hackernews_agent,
    description="Research latest tech trends from Hacker News",
)

research_finance = Step(
    name="research_finance",
    agent=finance_agent,
    description="Research financial data and market trends",
)

# Custom function step that has access to ALL previous step outputs


def create_comprehensive_report(step_input: StepInput) -> StepOutput:
    """
    Custom function that creates a report using data from multiple previous steps.
    This function has access to ALL previous step outputs and the original workflow message.
    """

    # Access original workflow input
    original_topic = step_input.input or ""

    # Access specific step outputs by name
    hackernews_data = step_input.get_step_content("research_hackernews") or ""
    finance_data = step_input.get_step_content("research_finance") or ""

    # Or access ALL previous content
    _ = step_input.get_all_previous_content()

    # Create a comprehensive report combining all sources
    report = f"""
        # Comprehensive Research Report: {original_topic}

        ## Executive Summary
        Based on research from HackerNews and web sources, here's a comprehensive analysis of {original_topic}.

        ## HackerNews Insights
        {hackernews_data[:500]}...

        ## Finance Research Findings
        {finance_data[:500]}...
    """

    return StepOutput(
        step_name="comprehensive_report", content=report.strip(), success=True
    )


comprehensive_report_step = Step(
    name="comprehensive_report",
    executor=create_comprehensive_report,
    description="Create comprehensive report from all research sources",
)

# Final reasoning step using reasoning agent
reasoning_step = Step(
    name="final_reasoning",
    agent=reasoning_agent,
    description="Apply reasoning to create final insights and recommendations",
)

workflow = Workflow(
    name="Enhanced Research Workflow",
    description="Multi-source research with custom data flow and reasoning",
    steps=[
        research_hackernews,
        research_finance,
        comprehensive_report_step,  # Has access to both previous steps
        reasoning_step,  # Gets the last step output (comprehensive report)
    ],
)

if __name__ == "__main__":
    workflow.print_response(
        "Latest developments in artificial intelligence and machine learning",
        stream=True,
    )
```

## 使用附加数据执行函数步骤
本示例展示了 Workflow 2.0 通过 additional_data 传递元数据和上下文信息到步骤的支持。

这个例子展示了如何通过additional_data传递元数据和上下文信息到步。这使得工作流逻辑与配置分离，实现基于外部上下文的动态行为。

### 主要特点：
- 上下文感知步骤：在自定义函数中访问step_input.additional_data
- 灵活的元数据：传递用户信息、优先级、设置等。
- 清晰分离：保持工作流逻辑的专注，同时用上下文丰富步骤。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
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
    instructions="Get stock prices and financial data",
)

# Define research team for complex analysis
research_team = Team(
    name="Research Team",
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


def custom_content_planning_function(step_input: StepInput) -> StepOutput:
    """
    Custom function that does intelligent content planning with context awareness
    Now also uses additional_data for extra context
    """
    message = step_input.input
    previous_step_content = step_input.previous_step_content

    # Access additional_data that was passed with the workflow
    additional_data = step_input.additional_data or {}
    user_email = additional_data.get("user_email", "No email provided")
    priority = additional_data.get("priority", "normal")
    client_type = additional_data.get("client_type", "standard")

    # Create intelligent planning prompt
    planning_prompt = f"""
        STRATEGIC CONTENT PLANNING REQUEST:

        Core Topic: {message}

        Research Results: {previous_step_content[:500] if previous_step_content else "No research results"}

        Additional Context:
        - Client Type: {client_type}
        - Priority Level: {priority}
        - Contact Email: {user_email}

        Planning Requirements:
        1. Create a comprehensive content strategy based on the research
        2. Leverage the research findings effectively
        3. Identify content formats and channels
        4. Provide timeline and priority recommendations
        5. Include engagement and distribution strategies
        {"6. Mark as HIGH PRIORITY delivery" if priority == "high" else "6. Standard delivery timeline"}

        Please create a detailed, actionable content plan.
    """

    try:
        response = content_planner.run(planning_prompt)

        enhanced_content = f"""
            ## Strategic Content Plan

            **Planning Topic:** {message}

            **Client Details:**
            - Type: {client_type}
            - Priority: {priority.upper()}
            - Contact: {user_email}

            **Research Integration:** {"Research-based" if previous_step_content else "No research foundation"}

            **Content Strategy:**
            {response.content}

            **Custom Planning Enhancements:**
            - Research Integration: {"High" if previous_step_content else "Baseline"}
            - Strategic Alignment: Optimized for multi-channel distribution
            - Execution Ready: Detailed action items included
            - Priority Level: {priority.upper()}
        """.strip()

        return StepOutput(content=enhanced_content, response=response)

    except Exception as e:
        return StepOutput(
            content=f"Custom content planning failed: {str(e)}",
            success=False,
        )


# Define steps using different executor types

research_step = Step(
    name="Research Step",
    team=research_team,
)

content_planning_step = Step(
    name="Content Planning Step",
    executor=custom_content_planning_function,
)


# Define and use examples
if __name__ == "__main__":
    content_creation_workflow = Workflow(
        name="Content Creation Workflow",
        description="Automated content creation with custom execution options",
        db=SqliteDb(
            session_table="workflow_session",
            db_file="tmp/workflow.db",
        ),
        steps=[research_step, content_planning_step],
    )

    # Run workflow with additional_data
    content_creation_workflow.print_response(
        input="AI trends in 2024",
        additional_data={
            "user_email": "michael@dundermifflin.com",
            "priority": "high",
            "client_type": "enterprise",
        },
        markdown=True,
        stream=True,
    )

    print("\n" + "=" * 60 + "\n")
```


## Structured I/O
使用Pydantic模型实现智能体、团队和自定义函数之间的类型安全数据流。

演示了代理/团队/自定义 Python 函数之间的类型安全数据流。每一步：
- 接收结构化输入（pydantic 模型、列表、字典或原始字符串）
- 生成结构化输出（例如ResearchFindings，ContentStrategy）

您还可以使用此模式创建可在任何步骤中使用的自定义函数，并且您可以：
- 检查传入的数据类型（原始字符串或 Pydantic 模型）。
- 分析先前步骤的结构化输出。
- 在保证类型安全的前提下生成报告。

```python
from typing import List

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.workflow.step import Step
from agno.workflow.workflow import Workflow
from pydantic import BaseModel, Field


# Define structured models for each step
class ResearchFindings(BaseModel):
    """Structured research findings with key insights"""

    topic: str = Field(description="The research topic")
    key_insights: List[str] = Field(description="Main insights discovered", min_items=3)
    trending_technologies: List[str] = Field(
        description="Technologies that are trending", min_items=2
    )
    market_impact: str = Field(description="Potential market impact analysis")
    sources_count: int = Field(description="Number of sources researched")
    confidence_score: float = Field(
        description="Confidence in findings (0.0-1.0)", ge=0.0, le=1.0
    )


class ContentStrategy(BaseModel):
    """Structured content strategy based on research"""

    target_audience: str = Field(description="Primary target audience")
    content_pillars: List[str] = Field(description="Main content themes", min_items=3)
    posting_schedule: List[str] = Field(description="Recommended posting schedule")
    key_messages: List[str] = Field(
        description="Core messages to communicate", min_items=3
    )
    hashtags: List[str] = Field(description="Recommended hashtags", min_items=5)
    engagement_tactics: List[str] = Field(
        description="Ways to increase engagement", min_items=2
    )


class FinalContentPlan(BaseModel):
    """Final content plan with specific deliverables"""

    campaign_name: str = Field(description="Name for the content campaign")
    content_calendar: List[str] = Field(
        description="Specific content pieces planned", min_items=6
    )
    success_metrics: List[str] = Field(
        description="How to measure success", min_items=3
    )
    budget_estimate: str = Field(description="Estimated budget range")
    timeline: str = Field(description="Implementation timeline")
    risk_factors: List[str] = Field(
        description="Potential risks and mitigation", min_items=2
    )


# Define agents with response models
research_agent = Agent(
    name="AI Research Specialist",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools(), YFinanceTools()],
    role="Research AI trends and extract structured insights",
    output_schema=ResearchFindings,
    instructions=[
        "Research the given topic thoroughly using available tools",
        "Provide structured findings with confidence scores",
        "Focus on recent developments and market trends",
        "Make sure to structure your response according to the ResearchFindings model",
    ],
)

strategy_agent = Agent(
    name="Content Strategy Expert",
    model=OpenAIResponses(id="gpt-5.2"),
    role="Create content strategies based on research findings",
    output_schema=ContentStrategy,
    instructions=[
        "Analyze the research findings provided from the previous step",
        "Create a comprehensive content strategy based on the structured research data",
        "Focus on audience engagement and brand building",
        "Structure your response according to the ContentStrategy model",
    ],
)

planning_agent = Agent(
    name="Content Planning Specialist",
    model=OpenAIResponses(id="gpt-5.2"),
    role="Create detailed content plans and calendars",
    output_schema=FinalContentPlan,
    instructions=[
        "Use the content strategy from the previous step to create a detailed implementation plan",
        "Include specific timelines and success metrics",
        "Consider budget and resource constraints",
        "Structure your response according to the FinalContentPlan model",
    ],
)

# Define steps
research_step = Step(
    name="research_insights",
    agent=research_agent,
)

strategy_step = Step(
    name="content_strategy",
    agent=strategy_agent,
)

planning_step = Step(
    name="final_planning",
    agent=planning_agent,
)

# Create workflow
structured_workflow = Workflow(
    name="Structured Content Creation Pipeline",
    description="AI-powered content creation with structured data flow",
    steps=[research_step, strategy_step, planning_step],
)

if __name__ == "__main__":
    print("=== Testing Structured Output Flow Between Steps ===")

    # Test with simple string input
    structured_workflow.print_response(
        input="Latest developments in artificial intelligence and machine learning"
    )
```

## 带有输入模式验证的工作流
此示例演示了工作流如何使用 Pydantic 模型进行输入模式验证，以确保工作流入口点的类型安全性和数据完整性。

本示例展示了如何在工作流中使用输入模式验证来强制执行类型安全和数据结构验证。通过input_schema使用 Pydantic 模型定义输入模式验证，您可以确保工作流在执行开始前接收到结构正确且经过验证的数据。


### 主要特点：
- 类型安全：根据 Pydantic 模型自动验证输入数据
- 结构验证：确保所有必填字段均已存在且填写正确。
- 清晰的合同：明确定义您的工作流程需要哪些数据
- 错误预防：在工作流执行开始前捕获无效输入
- 多种输入格式：支持 Pydantic 模型和匹配字典

```python
from typing import List

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.workflow.step import Step
from agno.workflow.workflow import Workflow
from pydantic import BaseModel, Field


class DifferentModel(BaseModel):
    name: str


class ResearchTopic(BaseModel):
    """Structured research topic with specific requirements"""

    topic: str
    focus_areas: List[str] = Field(description="Specific areas to focus on")
    target_audience: str = Field(description="Who this research is for")
    sources_required: int = Field(description="Number of sources needed", default=5)


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
        db=SqliteDb(
            session_table="workflow_session",
            db_file="tmp/workflow.db",
        ),
        steps=[research_step, content_planning_step],
        input_schema=ResearchTopic,  # <-- Define input schema for validation
    )

    print("=== Example 1: Valid Pydantic Model Input ===")
    research_topic = ResearchTopic(
        topic="AI trends in 2024",
        focus_areas=[
            "Machine Learning",
            "Natural Language Processing",
            "Computer Vision",
            "AI Ethics",
        ],
        target_audience="Tech professionals and business leaders",
    )

    # This will work properly - input matches the schema
    content_creation_workflow.print_response(
        input=research_topic,
        markdown=True,
    )

    print("\n=== Example 2: Valid Dictionary Input ===")
    # This will also work - dict matches ResearchTopic structure
    content_creation_workflow.print_response(
        input={
            "topic": "AI trends in 2024",
            "focus_areas": ["Machine Learning", "Computer Vision"],
            "target_audience": "Tech professionals",
            "sources_required": 8
        },
        markdown=True,
    )

    print("\n=== Example 3: Missing Required Fields (Commented - Would Fail) ===")
    # This would fail - missing required 'target_audience' field
    # Uncomment to see validation error:
    # content_creation_workflow.print_response(
    #     input=ResearchTopic(
    #         topic="AI trends in 2024",
    #         focus_areas=[
    #             "Machine Learning",
    #             "Natural Language Processing",
    #             "Computer Vision",
    #             "AI Ethics",
    #         ],
    #         # target_audience missing - will raise ValidationError
    #     ),
    #     markdown=True,
    # )

    print("\n=== Example 4: Wrong Model Type (Commented - Would Fail) ===")
    # This would fail - different Pydantic model provided
    # Uncomment to see validation error:
    # content_creation_workflow.print_response(
    #     input=DifferentModel(name="test"),
    #     markdown=True,
    # )

    print("\n=== Example 5: Type Mismatch (Commented - Would Fail) ===")
    # This would fail - wrong data types
    # Uncomment to see validation error:
    # content_creation_workflow.print_response(
    #     input={
    #         "topic": 123,  # Should be string
    #         "focus_areas": "Machine Learning",  # Should be List[str]
    #         "target_audience": ["audience1", "audience2"],  # Should be string
    #     },
    #     markdown=True,
    # )

```


## 基本对话工作流程
本示例演示了使用 WorkflowAgent 的基本对话工作流程。

本示例展示了如何使用该功能WorkflowAgent创建对话式工作流程。
```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.workflow import WorkflowAgent
from agno.workflow.types import StepInput
from agno.workflow.workflow import Workflow

story_writer = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="You are tasked with writing a 100 word story based on a given topic",
)

story_formatter = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="You are tasked with breaking down a short story in prelogues, body and epilogue",
)


def add_references(step_input: StepInput):
    """Add references to the story"""

    previous_output = step_input.previous_step_content

    if isinstance(previous_output, str):
        return previous_output + "\n\nReferences: https://www.agno.com"


# Create a WorkflowAgent that will decide when to run the workflow
workflow_agent = WorkflowAgent(model=OpenAIResponses(id="gpt-5.2"), num_history_runs=4)

# Create workflow with the WorkflowAgent
workflow = Workflow(
    name="Story Generation Workflow",
    description="A workflow that generates stories, formats them, and adds references",
    agent=workflow_agent,
    steps=[story_writer, story_formatter, add_references],
    db=SqliteDb(db_file="tmp/workflow.db"),
)

def main():
    print("\n\n" + "=" * 80)
    print("STREAMING MODE EXAMPLES")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("FIRST CALL (STREAMING): Tell me a story about a dog named Rocky")
    print("=" * 80)
    workflow.print_response(
        "Tell me a story about a dog named Rocky",
        stream=True,
    )

    print("\n" + "=" * 80)
    print("SECOND CALL (STREAMING): What was Rocky's personality?")
    print("=" * 80)
    workflow.print_response(
        "What was Rocky's personality?", stream=True
    )

    print("\n" + "=" * 80)
    print("THIRD CALL (STREAMING): Now tell me a story about a cat named Luna")
    print("=" * 80)
    workflow.print_response(
        "Now tell me a story about a cat named Luna",
        stream=True,
    )

    print("\n" + "=" * 80)
    print("FOURTH CALL (STREAMING): Compare Rocky and Luna")
    print("=" * 80)
    workflow.print_response(
        "Compare Rocky and Luna", stream=True
    )

# ============================================================================
# STREAMING EXAMPLES
# ============================================================================

if __name__ == "__main__":
    main()

```


## 提前停止工作流程
此示例演示了Workflows 2.0如何提前终止正在运行的工作流。

本示例展示了如何创建工作流，以便在质量条件不满足时优雅地终止，从而防止对无效或不安全的数据进行下游处理。


适用场景：当您需要安全机制、质量门控或验证检查点，以便在条件不满足时阻止后续处理。非常适合数据验证流程、安全检查、质量保证工作流程，或任何继续使用无效输入可能导致问题的流程。

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.workflow import Workflow
from agno.workflow.types import StepInput, StepOutput

# Create agents with more specific validation criteria
data_validator = Agent(
    name="Data Validator",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=[
        "You are a data validator. Analyze the provided data and determine if it's valid.",
        "For data to be VALID, it must meet these criteria:",
        "- user_count: Must be a positive number (> 0)",
        "- revenue: Must be a positive number (> 0)",
        "- date: Must be in a reasonable date format (YYYY-MM-DD)",
        "",
        "Return exactly 'VALID' if all criteria are met.",
        "Return exactly 'INVALID' if any criteria fail.",
        "Also briefly explain your reasoning.",
    ],
)

data_processor = Agent(
    name="Data Processor",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="Process and transform the validated data.",
)

report_generator = Agent(
    name="Report Generator",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="Generate a final report from processed data.",
)


def early_exit_validator(step_input: StepInput) -> StepOutput:
    """
    Custom function that checks data quality and stops workflow early if invalid
    """
    # Get the validation result from previous step
    validation_result = step_input.previous_step_content or ""

    if "INVALID" in validation_result.upper():
        return StepOutput(
            content="Data validation failed. Workflow stopped early to prevent processing invalid data.",
            stop=True,  # Stop the entire workflow here
        )
    else:
        return StepOutput(
            content="Data validation passed. Continuing with processing...",
            stop=False,  # Continue normally
        )


# Create workflow with conditional early termination
workflow = Workflow(
    name="Data Processing with Early Exit",
    description="Process data but stop early if validation fails",
    steps=[
        data_validator,  # Step 1: Validate data
        early_exit_validator,  # Step 2: Check validation and possibly stop early
        data_processor,  # Step 3: Process data (only if validation passed)
        report_generator,  # Step 4: Generate report (only if processing completed)
    ],
)

if __name__ == "__main__":
    print("\n=== Testing with INVALID data ===")
    workflow.print_response(
        input="Process this data: {'user_count': -50, 'revenue': 'invalid_amount', 'date': 'bad_date'}"
    )

    print("=== Testing with VALID data ===")
    workflow.print_response(
        input="Process this data: {'user_count': 1000, 'revenue': 50000, 'date': '2024-01-15'}"
    )

```

### 工作流程取消
此示例演示了Workflows 2.0对取消正在运行的工作流执行的支持，包括基于线程的取消和处理已取消的响应。

本示例演示了如何实时取消正在运行的工作流执行。具体来说，它演示了：
- 基于线程的执行：在单独的线程中运行工作流，以实现非阻塞操作。
- 动态取消：在工作流运行时取消工作流
- 取消事件：处理和应对取消事件
- 状态跟踪：监控工作流程在执行和取消过程中的状态

```python
import threading
import time

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.run.agent import RunEvent
from agno.run.base import RunStatus
from agno.run.workflow import WorkflowRunEvent
from agno.tools.hackernews import HackerNewsTools
from agno.workflow.step import Step
from agno.workflow.workflow import Workflow


def long_running_task(workflow: Workflow, run_id_container: dict):
    """
    Simulate a long-running workflow task that can be cancelled.

    Args:
        workflow: The workflow to run
        run_id_container: Dictionary to store the run_id for cancellation

    Returns:
        Dictionary with run results and status
    """
    try:
        # Start the workflow run - this simulates a long task
        final_response = None
        content_pieces = []

        for chunk in workflow.run(
            "Write a very long story about a dragon who learns to code. "
            "Make it at least 2000 words with detailed descriptions and dialogue. "
            "Take your time and be very thorough.",
            stream=True,
        ):
            if "run_id" not in run_id_container and chunk.run_id:
                print(f"> Workflow run started: {chunk.run_id}")
                run_id_container["run_id"] = chunk.run_id

            if chunk.event in [RunEvent.run_content]:
                print(chunk.content, end="", flush=True)
                content_pieces.append(chunk.content)
            elif chunk.event == RunEvent.run_cancelled:
                print(f"\n> Workflow run was cancelled: {chunk.run_id}")
                run_id_container["result"] = {
                    "status": "cancelled",
                    "run_id": chunk.run_id,
                    "cancelled": True,
                    "content": "".join(content_pieces)[:200] + "..."
                    if content_pieces
                    else "No content before cancellation",
                }
                return
            elif chunk.event == WorkflowRunEvent.workflow_cancelled:
                print(f"\n> Workflow run was cancelled: {chunk.run_id}")
                run_id_container["result"] = {
                    "status": "cancelled",
                    "run_id": chunk.run_id,
                    "cancelled": True,
                    "content": "".join(content_pieces)[:200] + "..."
                    if content_pieces
                    else "No content before cancellation",
                }
                return
            elif hasattr(chunk, "status") and chunk.status == RunStatus.completed:
                final_response = chunk

        # If we get here, the run completed successfully
        if final_response:
            run_id_container["result"] = {
                "status": final_response.status.value
                if final_response.status
                else "completed",
                "run_id": final_response.run_id,
                "cancelled": final_response.status == RunStatus.cancelled,
                "content": ("".join(content_pieces)[:200] + "...")
                if content_pieces
                else "No content",
            }
        else:
            run_id_container["result"] = {
                "status": "unknown",
                "run_id": run_id_container.get("run_id"),
                "cancelled": False,
                "content": ("".join(content_pieces)[:200] + "...")
                if content_pieces
                else "No content",
            }

    except Exception as e:
        print(f"\n> Exception in run: {str(e)}")
        run_id_container["result"] = {
            "status": "error",
            "error": str(e),
            "run_id": run_id_container.get("run_id"),
            "cancelled": True,
            "content": "Error occurred",
        }


def cancel_after_delay(
    workflow: Workflow, run_id_container: dict, delay_seconds: int = 3
):
    """
    Cancel the workflow run after a specified delay.

    Args:
        workflow: The workflow whose run should be cancelled
        run_id_container: Dictionary containing the run_id to cancel
        delay_seconds: How long to wait before cancelling
    """
    print(f"> Will cancel workflow run in {delay_seconds} seconds...")
    time.sleep(delay_seconds)

    run_id = run_id_container.get("run_id")
    if run_id:
        print(f"> Cancelling workflow run: {run_id}")
        success = workflow.cancel_run(run_id)
        if success:
            print(f"> Workflow run {run_id} marked for cancellation")
        else:
            print(
                f"> Failed to cancel workflow run {run_id} (may not exist or already completed)"
            )
    else:
        print(">  No run_id found to cancel")


def main():
    """Main function demonstrating workflow run cancellation."""

    # Create workflow agents
    researcher = Agent(
        name="Research Agent",
        model=OpenAIResponses(id="gpt-5.2"),
        tools=[HackerNewsTools()],
        instructions="Research the given topic and provide key facts and insights.",
    )

    writer = Agent(
        name="Writing Agent",
        model=OpenAIResponses(id="gpt-5.2"),
        instructions="Write a comprehensive article based on the research provided. Make it engaging and well-structured.",
    )
    research_step = Step(
        name="research",
        agent=researcher,
        description="Research the topic and gather information",
    )

    writing_step = Step(
        name="writing",
        agent=writer,
        description="Write an article based on the research",
    )

    # Create a Steps sequence that chains these above steps together
    article_workflow = Workflow(
        description="Automated article creation from research to writing",
        steps=[research_step, writing_step],
        debug_mode=True,
    )

    print("> Starting workflow run cancellation example...")
    print("=" * 50)

    # Container to share run_id between threads
    run_id_container = {}

    # Start the workflow run in a separate thread
    workflow_thread = threading.Thread(
        target=lambda: long_running_task(article_workflow, run_id_container),
        name="WorkflowRunThread",
    )

    # Start the cancellation thread
    cancel_thread = threading.Thread(
        target=cancel_after_delay,
        args=(article_workflow, run_id_container, 8),  # Cancel after 8 seconds
        name="CancelThread",
    )

    # Start both threads
    print("> Starting workflow run thread...")
    workflow_thread.start()

    print("> Starting cancellation thread...")
    cancel_thread.start()

    # Wait for both threads to complete
    print("> Waiting for threads to complete...")
    workflow_thread.join()
    cancel_thread.join()

    # Print the results
    print("\n" + "=" * 50)
    print("> RESULTS:")
    print("=" * 50)

    result = run_id_container.get("result")
    if result:
        print(f"Status: {result['status']}")
        print(f"Run ID: {result['run_id']}")
        print(f"Was Cancelled: {result['cancelled']}")

        if result.get("error"):
            print(f"Error: {result['error']}")
        else:
            print(f"Content Preview: {result['content']}")

        if result["cancelled"]:
            print("\n> SUCCESS: Workflow run was successfully cancelled!")
        else:
            print("\n>  WARNING: Workflow run completed before cancellation")
    else:
        print("> No result obtained - check if cancellation happened during streaming")

    print("\n> Workflow cancellation example completed!")


if __name__ == "__main__":
    # Run the main example
    main()

```



## 异步事件流

此示例演示如何从工作流中传输事件。

```python
import asyncio
from textwrap import dedent
from typing import AsyncIterator

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.run.workflow import WorkflowRunOutputEvent, WorkflowRunEvent
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.workflow.types import StepInput, StepOutput
from agno.workflow.workflow import Workflow

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

writer_agent = Agent(
    name="Writer Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="Write a blog post on the topic",
)


async def prepare_input_for_web_search(
    step_input: StepInput,
) -> AsyncIterator[StepOutput]:
    """Generator function that yields StepOutput"""
    topic = step_input.input

    # Create proper StepOutput content
    content = dedent(f"""\
        I'm writing a blog post on the topic
        <topic>
        {topic}
        </topic>

        Search the web for atleast 10 articles\
        """)

    # Yield a StepOutput as the final result
    yield StepOutput(content=content)


async def prepare_input_for_writer(step_input: StepInput) -> AsyncIterator[StepOutput]:
    """Generator function that yields StepOutput"""
    topic = step_input.input
    research_team_output = step_input.previous_step_content

    # Create proper StepOutput content
    content = dedent(f"""\
        I'm writing a blog post on the topic:
        <topic>
        {topic}
        </topic>

        Here is information from the web:
        <research_results>
        {research_team_output}
        </research_results>\
        """)

    # Yield a StepOutput as the final result
    yield StepOutput(content=content)


# Define research team for complex analysis
research_team = Team(
    name="Research Team",
    members=[hackernews_agent, finance_agent],
    instructions="Research tech topics from Hackernews and the web",
)


async def main():
    content_creation_workflow = Workflow(
        name="Blog Post Workflow",
        description="Automated blog post creation from Hackernews and the web",
        steps=[
            prepare_input_for_web_search,
            research_team,
            prepare_input_for_writer,
            writer_agent,
        ],
        db=SqliteDb(
            session_table="workflow_session",
            db_file="tmp/workflow.db",
        ),
    )

    resp: AsyncIterator[WorkflowRunOutputEvent] = content_creation_workflow.arun(
        input="AI trends in 2024",
        markdown=True,
        stream=True,
        stream_events=True,
    )
    async for event in resp:
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


if __name__ == "__main__":
    asyncio.run(main())
```

## 工作流中存储事件和要跳过的事件
此示例演示了Workflows 2.0 的事件存储功能。

本示例演示了Workflows 2.0 的事件存储功能，展示了如何执行以下操作：
- 存储执行事件以进行调试/审计（store_events=True）
- 过滤掉噪声事件（events_to_skip），以便专注于关键工作流程里程碑
- 执行后通过以下方式访问已存储的事件workflow.run_response.events

#### 主要特点：
- 选择性存储：跳过冗长的事件（例如step_started），同时保留关键里程碑。
- 调试/审计：捕获执行流程以进行分析，无需手动记录日志。
- 性能优化：通过过滤非必要事件来减少存储开销。
```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.run.agent import (
    RunContentEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from agno.run.workflow import WorkflowRunEvent, WorkflowRunOutput
from agno.tools.hackernews import HackerNewsTools
from agno.run.agent import RunEvent
from agno.workflow.parallel import Parallel
from agno.workflow.step import Step
from agno.workflow.workflow import Workflow

# Define agents for different tasks
news_agent = Agent(
    name="News Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
    instructions="You are a news researcher. Get the latest tech news and summarize key points.",
)

search_agent = Agent(
    name="Search Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="You are a search specialist. Find relevant information on given topics.",
)

analysis_agent = Agent(
    name="Analysis Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="You are an analyst. Analyze the provided information and give insights.",
)

summary_agent = Agent(
    name="Summary Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="You are a summarizer. Create concise summaries of the provided content.",
)

research_step = Step(
    name="Research Step",
    agent=news_agent,
)

search_step = Step(
    name="Search Step",
    agent=search_agent,
)


def print_stored_events(run_response: WorkflowRunOutput, example_name):
    """Helper function to print stored events in a nice format"""
    print(f"\n--- {example_name} - Stored Events ---")
    if run_response.events:
        print(f"Total stored events: {len(run_response.events)}")
        for i, event in enumerate(run_response.events, 1):
            print(f"  {i}. {event.event}")
    else:
        print("No events stored")
    print()


print("=== Simple Step Workflow with Event Storage ===")
step_workflow = Workflow(
    name="Simple Step Workflow",
    description="Basic workflow demonstrating step event storage",
    db=SqliteDb(
        session_table="workflow_session",
        db_file="tmp/workflow.db",
    ),
    steps=[research_step, search_step],
    store_events=True,
    events_to_skip=[
        WorkflowRunEvent.step_started,
        WorkflowRunEvent.workflow_completed,
        RunEvent.run_content,
        RunEvent.run_started,
        RunEvent.run_completed,
    ],  # Skip step started events to reduce noise
)

print("Running Step workflow with streaming...")
for event in step_workflow.run(
    input="AI trends in 2024",
    stream=True,
    stream_events=True,
):
    # Filter out RunContentEvent from printing to reduce noise
    if not isinstance(
        event, (RunContentEvent, ToolCallStartedEvent, ToolCallCompletedEvent)
    ):
        print(
            f"Event: {event.event if hasattr(event, 'event') else type(event).__name__}"
        )
run_response = step_workflow.get_last_run_output()

print("\nStep workflow completed!")
print(
    f"Total events stored: {len(run_response.events) if run_response and run_response.events else 0}"
)

# Print stored events in a nice format
print_stored_events(run_response, "Simple Step Workflow")

# ------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #

# Example 2: Parallel Primitive with Event Storage
print("=== 2. Parallel Example ===")
parallel_workflow = Workflow(
    name="Parallel Research Workflow",
    steps=[
        Parallel(
            Step(name="News Research", agent=news_agent),
            Step(name="Web Search", agent=search_agent),
            name="Parallel Research",
        ),
        Step(name="Combine Results", agent=analysis_agent),
    ],
    db=SqliteDb(
        session_table="workflow_parallel",
        db_file="tmp/workflow_parallel.db",
    ),
    store_events=True,
    events_to_skip=[
        WorkflowRunEvent.parallel_execution_started,
        WorkflowRunEvent.parallel_execution_completed,
    ],
)

print("Running Parallel workflow...")
for event in parallel_workflow.run(
    input="Research machine learning developments",
    stream=True,
    stream_events=True,
):
    # Filter out RunContentEvent from printing
    if not isinstance(event, RunContentEvent):
        print(
            f"Event: {event.event if hasattr(event, 'event') else type(event).__name__}"
        )

run_response = parallel_workflow.get_last_run_output()
print(f"Parallel workflow stored {len(run_response.events)} events")
print_stored_events(run_response, "Parallel Workflow")
print()

```
## 在 AgentOS 上使用自定义函数流式传输
本示例演示了如何在 AgentOS 上使用带自定义函数执行器和流式传输的命名步骤。

本示例演示了如何使用带有自定义函数执行器的 Step 对象，以及如何使用AgentOS流式传输它们的响应。

在自定义函数步骤中运行的代理和团队也可以将其结果直接流式传输到 AgentOS。

```python
from typing import AsyncIterator, Union

from agno.agent.agent import Agent
from agno.db.in_memory import InMemoryDb

# Import the workflows
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.os import AgentOS
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.workflow.step import Step, StepInput, StepOutput, WorkflowRunOutputEvent
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
    instructions="Get stock prices and financial data",
)

# Define research team for complex analysis
research_team = Team(
    name="Research Team",
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

            **Research Integration:** {"Research-based" if previous_step_content else "No research foundation"}

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


# Define steps using different executor types

research_step = Step(
    name="Research Step",
    team=research_team,
)

content_planning_step = Step(
    name="Content Planning Step",
    executor=custom_content_planning_function,
)

streaming_content_workflow = Workflow(
    name="Streaming Content Creation Workflow",
    description="Automated content creation with streaming custom execution functions",
    db=SqliteDb(
        session_table="workflow_session",
        db_file="tmp/workflow.db",
    ),
    steps=[
        research_step,
        content_planning_step,
    ],
)


# Initialize the AgentOS with the workflows
agent_os = AgentOS(
    description="Example OS setup",
    workflows=[streaming_content_workflow],
)
app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(app="workflow_with_custom_function_stream:app", reload=True)
```


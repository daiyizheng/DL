# 取消运行
取消正在运行的代理、团队或工作流执行。

运行取消功能允许您停止正在运行的代理、团队或工作流执行。此功能对于管理长时间运行的操作、实现超时设置或允许用户中断进程非常有用。
​
## 工作原理
该cancel_run()函数会将一次运行标记为取消。当前步骤完成后，执行将优雅地停止，确保操作顺利完成，不会使资源处于不一致的状态。

## 取消行为：
- 非流式运行：RunOutput返回的对象状态设置为RunStatus.cancelled。
- 流式运行：取消时会发出RunCancelledEvent。

## 代理运行取消
从另一个线程取消正在运行的代理执行。

此示例演示了如何通过在单独的线程中启动代理运行并从另一个线程取消它来取消正在运行的代理执行。它展示了对已取消响应的正确处理和线程管理。


```python
"""
Example demonstrating how to cancel a running agent execution.

This example shows how to:
1. Start an agent run in a separate thread
2. Cancel the run from another thread
3. Handle the cancelled response
"""

import threading
import time

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.run.agent import RunEvent
from agno.run.base import RunStatus


def long_running_task(agent: Agent, run_id_container: dict):
    """
    Simulate a long-running agent task that can be cancelled.

    Args:
        agent: The agent to run
        run_id_container: Dictionary to store the run_id for cancellation

    Returns:
        Dictionary with run results and status
    """
    # Start the agent run - this simulates a long task
    final_response = None
    content_pieces = []

    for chunk in agent.run(
        "Write a very long story about a dragon who learns to code. "
        "Make it at least 2000 words with detailed descriptions and dialogue. "
        "Take your time and be very thorough.",
        stream=True,
    ):
        if "run_id" not in run_id_container and chunk.run_id:
            run_id_container["run_id"] = chunk.run_id

        if chunk.event == RunEvent.run_content:
            print(chunk.content, end="", flush=True)
            content_pieces.append(chunk.content)
        # When the run is cancelled, a `RunEvent.run_cancelled` event is emitted
        elif chunk.event == RunEvent.run_cancelled:
            print(f"\nRun was cancelled: {chunk.run_id}")
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


def cancel_after_delay(agent: Agent, run_id_container: dict, delay_seconds: int = 3):
    """
    Cancel the agent run after a specified delay.

    Args:
        agent: The agent whose run should be cancelled
        run_id_container: Dictionary containing the run_id to cancel
        delay_seconds: How long to wait before cancelling
    """
    print(f"Will cancel run in {delay_seconds} seconds...")
    time.sleep(delay_seconds)

    run_id = run_id_container.get("run_id")
    if run_id:
        print(f"Cancelling run: {run_id}")
        success = agent.cancel_run(run_id)
        if success:
            print(f"Run {run_id} marked for cancellation")
        else:
            print(
                f"Failed to cancel run {run_id} (may not exist or already completed)"
            )
    else:
        print("No run_id found to cancel")


def main():
    """Main function demonstrating agent run cancellation."""

    # Initialize the agent with a model
    agent = Agent(
        name="StorytellerAgent",
        model=OpenAIResponses(id="gpt-5.2"),  # Use a model that can generate long responses
        description="An agent that writes detailed stories",
    )

    print("Starting agent run cancellation example...")
    print("=" * 50)

    # Container to share run_id between threads
    run_id_container = {}

    # Start the agent run in a separate thread
    agent_thread = threading.Thread(
        target=lambda: long_running_task(agent, run_id_container), name="AgentRunThread"
    )

    # Start the cancellation thread
    cancel_thread = threading.Thread(
        target=cancel_after_delay,
        args=(agent, run_id_container, 8),  # Cancel after 8 seconds
        name="CancelThread",
    )

    # Start both threads
    print("Starting agent run thread...")
    agent_thread.start()

    print("Starting cancellation thread...")
    cancel_thread.start()

    # Wait for both threads to complete
    print("Waiting for threads to complete...")
    agent_thread.join()
    cancel_thread.join()

    # Print the results
    print("\n" + "=" * 50)
    print("RESULTS:")
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
            print("\nSUCCESS: Run was successfully cancelled!")
        else:
            print("\nWARNING: Run completed before cancellation")
    else:
        print("No result obtained - check if cancellation happened during streaming")

    print("\nExample completed!")


if __name__ == "__main__":
    # Run the main example
    main()
```

## 团队跑步活动取消

从另一个线程取消正在运行的团队执行。

此示例演示了如何通过在单独的线程中启动一个团队运行，然后从另一个线程取消该团队运行来取消正在运行的团队执行。它展示了如何正确处理已取消的响应以及线程管理。
​

```python
"""
Example demonstrating how to cancel a running team execution.

This example shows how to:
1. Start a team run in a separate thread
2. Cancel the run from another thread
3. Handle the cancelled response
"""

import threading
import time

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.run.agent import RunEvent
from agno.run.base import RunStatus
from agno.run.team import TeamRunEvent
from agno.team import Team


def long_running_task(team: Team, run_id_container: dict):
    """
    Simulate a long-running team task that can be cancelled.

    Args:
        team: The team to run
        run_id_container: Dictionary to store the run_id for cancellation

    Returns:
        Dictionary with run results and status
    """
    try:
        # Start the team run - this simulates a long task
        final_response = None
        content_pieces = []

        for chunk in team.run(
            "Write a very long story about a dragon who learns to code. "
            "Make it at least 2000 words with detailed descriptions and dialogue. "
            "Take your time and be very thorough.",
            stream=True,
        ):
            if "run_id" not in run_id_container and chunk.run_id:
                print(f"Team run started: {chunk.run_id}")
                run_id_container["run_id"] = chunk.run_id

            if chunk.event in [TeamRunEvent.run_content, RunEvent.run_content]:
                print(chunk.content, end="", flush=True)
                content_pieces.append(chunk.content)
            elif chunk.event == RunEvent.run_cancelled:
                print(f"\nMember run was cancelled: {chunk.run_id}")
                run_id_container["result"] = {
                    "status": "cancelled",
                    "run_id": chunk.run_id,
                    "cancelled": True,
                    "content": "".join(content_pieces)[:200] + "..."
                    if content_pieces
                    else "No content before cancellation",
                }
                return
            elif chunk.event == TeamRunEvent.run_cancelled:
                print(f"\nTeam run was cancelled: {chunk.run_id}")
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
        print(f"\nException in run: {str(e)}")
        run_id_container["result"] = {
            "status": "error",
            "error": str(e),
            "run_id": run_id_container.get("run_id"),
            "cancelled": True,
            "content": "Error occurred",
        }


def cancel_after_delay(team: Team, run_id_container: dict, delay_seconds: int = 3):
    """
    Cancel the team run after a specified delay.

    Args:
        team: The team whose run should be cancelled
        run_id_container: Dictionary containing the run_id to cancel
        delay_seconds: How long to wait before cancelling
    """
    print(f"Will cancel team run in {delay_seconds} seconds...")
    time.sleep(delay_seconds)

    run_id = run_id_container.get("run_id")
    if run_id:
        print(f"Cancelling team run: {run_id}")
        success = team.cancel_run(run_id)
        if success:
            print(f"Team run {run_id} marked for cancellation")
        else:
            print(
                f"Failed to cancel team run {run_id} (may not exist or already completed)"
            )
    else:
        print("No run_id found to cancel")


def main():
    """Main function demonstrating team run cancellation."""

    # Create team members
    storyteller_agent = Agent(
        name="StorytellerAgent",
        model=OpenAIResponses(id="gpt-5.2"),
        description="An agent that writes creative stories",
    )

    editor_agent = Agent(
        name="EditorAgent",
        model=OpenAIResponses(id="gpt-5.2"),
        description="An agent that reviews and improves stories",
    )

    # Initialize the team with agents
    team = Team(
        name="Storytelling Team",
        members=[storyteller_agent, editor_agent],
        model=OpenAIResponses(id="gpt-5.2"),  # Team leader model
        description="A team that collaborates to write detailed stories",
    )

    print("Starting team run cancellation example...")
    print("=" * 50)

    # Container to share run_id between threads
    run_id_container = {}

    # Start the team run in a separate thread
    team_thread = threading.Thread(
        target=lambda: long_running_task(team, run_id_container), name="TeamRunThread"
    )

    # Start the cancellation thread
    cancel_thread = threading.Thread(
        target=cancel_after_delay,
        args=(team, run_id_container, 8),  # Cancel after 8 seconds
        name="CancelThread",
    )

    # Start both threads
    print("Starting team run thread...")
    team_thread.start()

    print("Starting cancellation thread...")
    cancel_thread.start()

    # Wait for both threads to complete
    print("Waiting for threads to complete...")
    team_thread.join()
    cancel_thread.join()

    # Print the results
    print("\n" + "=" * 50)
    print("RESULTS:")
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
            print("\nSUCCESS: Team run was successfully cancelled!")
        else:
            print("\nWARNING: Team run completed before cancellation")
    else:
        print("No result obtained - check if cancellation happened during streaming")

    print("\nTeam cancellation example completed!")


if __name__ == "__main__":
    # Run the main example
    main()
```

## 工作流运行取消

从另一个线程取消正在运行的工作流执行。

本示例演示了如何通过在单独的线程中启动工作流运行并从另一个线程取消该工作流来取消正在运行的工作流执行。它展示了对已取消响应的正确处理和线程管理。

```python
"""
Example demonstrating how to cancel a running workflow execution.

This example shows how to:
1. Start a workflow run in a separate thread
2. Cancel the run from another thread
3. Handle the cancelled response
"""

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
                print(f"Workflow run started: {chunk.run_id}")
                run_id_container["run_id"] = chunk.run_id

            if chunk.event in [RunEvent.run_content]:
                print(chunk.content, end="", flush=True)
                content_pieces.append(chunk.content)
            elif chunk.event == RunEvent.run_cancelled:
                print(f"\nWorkflow run was cancelled: {chunk.run_id}")
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
                print(f"\nWorkflow run was cancelled: {chunk.run_id}")
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
        print(f"\nException in run: {str(e)}")
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
    print(f"Will cancel workflow run in {delay_seconds} seconds...")
    time.sleep(delay_seconds)

    run_id = run_id_container.get("run_id")
    if run_id:
        print(f"Cancelling workflow run: {run_id}")
        success = workflow.cancel_run(run_id)
        if success:
            print(f"Workflow run {run_id} marked for cancellation")
        else:
            print(
                f"Failed to cancel workflow run {run_id} (may not exist or already completed)"
            )
    else:
        print("No run_id found to cancel")


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

    print("Starting workflow run cancellation example...")
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
    print("Starting workflow run thread...")
    workflow_thread.start()

    print("Starting cancellation thread...")
    cancel_thread.start()

    # Wait for both threads to complete
    print("Waiting for threads to complete...")
    workflow_thread.join()
    cancel_thread.join()

    # Print the results
    print("\n" + "=" * 50)
    print("RESULTS:")
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
            print("\nSUCCESS: Workflow run was successfully cancelled!")
        else:
            print("\nWARNING: Workflow run completed before cancellation")
    else:
        print("No result obtained - check if cancellation happened during streaming")

    print("\nWorkflow cancellation example completed!")


if __name__ == "__main__":
    # Run the main example
    main()
```


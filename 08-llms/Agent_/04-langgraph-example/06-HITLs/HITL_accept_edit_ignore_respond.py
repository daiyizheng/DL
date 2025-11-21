from typing import Annotated, Sequence, TypedDict
import uuid
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command


from langchain_core.messages import BaseMessage, AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph.message import MessagesState

from langgraph.prebuilt.interrupt import (
    ActionRequest,
    HumanInterrupt,
    HumanInterruptConfig,
    HumanResponse,
)
# from langchain.agents import HumanInterrupt, ActionRequest, HumanInterruptConfig,HumanResponse ## langchain  v1版本


class ReportState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    topic: str # Report topic  


## 定义输出类型
class ReportStateOutput(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    final_report: str # Final report
    topic: str # Report topic  

## 定义输入类型
class StateInput(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    topic: str # Report topic
    


def human_node_1(state: MessagesState):
    sections_str = '白术很好吃！！'
     # 从中断处获取关于报告计划的反馈
    interrupt_message = f"""请对以下报告提供反馈。
                        \n\n{sections_str}\n
                        \n报告是否满足您的需求？\n若通过，请回复“true”以批准报告。\n或者，提供反馈以重新生成报告："""

    action_request = ActionRequest(
        action="确认报告计划",
        args={"report_plan": interrupt_message},
    ) ## 类似标题

    interrupt_config = HumanInterruptConfig(
        allow_ignore=True,  # Allow the user to `ignore` the interrupt.        ## 忽略
        allow_respond=True,  # Allow the user to `respond` to the interrupt.   ## 发送新的内容
        allow_edit=True,  # Allow the user to `edit` the interrupt's args.     ## 对给定的内容进行编辑
        allow_accept=True,  # Allow the user to `accept` the interrupt's args. ## 接受反馈
    )

    description = (
        "# 确认报告"
        + "请仔细阅读报告，并就其是否满足您的需求提供反馈。 "
        + "如果你接受，它将启动部分写作。 "
        + "如果您编辑并提交，编辑后的报告将用于生成部分。"
        + "如果忽略，则不会生成报告"
        + "如果您做出响应，该响应将用于生成新报告"
    )

    request = HumanInterrupt(
        action_request=action_request, config=interrupt_config, description=description
    )

    human_response: HumanResponse = interrupt([request])[0]

    # print(human_response.get("args")) # {'type': 'accept', 'args': {'action': 'Confirm report plan', 'args': {'report_plan': 'Please provide...'}}}
    if human_response.get("type") == "response":
        # If the user provides feedback, regenerate the report plan
        # return Command(
        #     goto="generate_report_plan",
        #     update={"feedback_on_report_plan": human_response.get("args")},
        # )
        print("response")
    elif human_response.get("type") == "accept":
        print("accept")
    elif  human_response.get("type") == "ignore":
        print("ignore")
    elif human_response.get("type") == "edit":
        print("ignore")
    else:
        raise TypeError(
            f"Interrupt value of type {type(human_response)} is not supported."
        )

    return {"topic": human_response.get("type")}




graph_builder = StateGraph(StateInput)
graph_builder.add_node("human_node_1", human_node_1)
graph_builder.add_edge(START, "human_node_1")

checkpointer = InMemorySaver()
# graph = graph_builder.compile(checkpointer=checkpointer)
graph = graph_builder.compile()
# thread_id = str(uuid.uuid4())
# config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
# result = graph.invoke(
#      {"messages":[{"role":"user", "content": "original text 2"}]}, config=config
# )

# graph.invoke(
#     Command(resume=[{"type": "accept"}]),
#     # Command(resume=[{"type": "edit", "args": {"args": {"hotel_name": "McKittrick Hotel"}}}]),
#     config=config)

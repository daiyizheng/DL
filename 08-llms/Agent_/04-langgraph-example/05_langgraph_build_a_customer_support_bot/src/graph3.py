from typing import Annotated
from datetime import datetime
import uuid
from dotenv import load_dotenv
import sys  # noqa: E402    
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages
from langgraph.types import Send, Command

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import tools_condition
from langchain_core.messages import ToolMessage
from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
sys.path.insert(0, "/Users/a1-6/Documents/projects/DL/08-llms/Agent_/04-langgraph-example/05_langgraph_build_a_customer_support_bot")
from src.tools.policies import lookup_policy
from src.tools.flights import fetch_user_flight_information, search_flights, update_ticket_to_new_flight, cancel_ticket
from src.tools.car_rental import  search_car_rentals, book_car_rental, update_car_rental, cancel_car_rental
from src.tools.hotels import search_hotels, book_hotel, update_hotel, cancel_hotel
from src.tools.excursions import search_trip_recommendations, book_excursion, update_excursion, cancel_excursion
from src.utils.utilities import create_tool_node_with_fallback, _print_event


# 仅“读取”的工具（例如检索器）不需要用户确认即可使用
part_3_safe_tools = [
    TavilySearchResults(max_results=1),
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
    search_car_rentals,
    search_hotels,
    search_trip_recommendations,
]

# 这些工具都会改变用户的保留意见。用户有权控制做出哪些决定
part_3_sensitive_tools = [
    update_ticket_to_new_flight,
    cancel_ticket,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
    book_hotel,
    update_hotel,
    cancel_hotel,
    book_excursion,
    update_excursion,
    cancel_excursion,
]
sensitive_tool_names = {t.name for t in part_3_sensitive_tools}


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


# 俳句速度更快、成本更低，但准确性较低
# llm = ChatAnthropic(model="claude-3-haiku-20240307")
# llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)
# 你可以更新 LLM，但你可能需要更新提示
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4-turbo-preview")

assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)


# 我们的 LLM 无需知道要路由到哪些节点。在它的“意识”下，它只是在调用函数。
part_3_assistant_runnable = assistant_prompt | llm.bind_tools(
    part_3_safe_tools + part_3_sensitive_tools
)

builder = StateGraph(State)

def user_info(state: State):
    return {"user_info": fetch_user_flight_information.invoke({})}


# 新功能：fetch_user_info 节点将首先运行，这意味着我们的助手无需执行任何操作即可查看用户的航班信息
builder.add_node("fetch_user_info", user_info)
builder.add_edge(START, "fetch_user_info")
builder.add_node("assistant", Assistant(part_3_assistant_runnable))
builder.add_node("safe_tools", create_tool_node_with_fallback(part_3_safe_tools))
builder.add_node(
    "sensitive_tools", create_tool_node_with_fallback(part_3_sensitive_tools)
)
# 定义逻辑
builder.add_edge("fetch_user_info", "assistant")


def route_tools(state: State):
    next_node = tools_condition(state)
    # 如果没有调用任何工具，则返回给用户
    if next_node == END:
        return END
    ai_message = state["messages"][-1]
   
    # 这假设单个工具调用。要处理并行工具调用，您需要使用 ANY 条件
    first_tool_call = ai_message.tool_calls[0]

    if first_tool_call["name"] in sensitive_tool_names:
        return "sensitive_tools"
    return "safe_tools"


### 下面是不并行计算
builder.add_conditional_edges(
    "assistant", route_tools, ["safe_tools", "sensitive_tools", END]
)
builder.add_edge("safe_tools", "assistant")
builder.add_edge("sensitive_tools", "assistant")

memory = InMemorySaver()
part_3_graph = builder.compile(
    checkpointer=memory,
    # 
    # 新增：图表在执行“工具”节点之前始终会暂停。
    # 用户可以在助手继续执行之前批准或拒绝（甚至修改请求）。
    interrupt_before=["sensitive_tools"],
)

# from IPython.display import Image, display

# try:
#     display(Image(part_3_graph.get_graph(xray=True).draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass

if __name__ == '__main__':

    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            # The passenger_id is used in our flight tools to
            # fetch the user's flight information
            "passenger_id": "3442 587242",
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,
        }
    }

    tutorial_questions = [
        "Hi there, what time is my flight?",
        "Am i allowed to update my flight to something sooner? I want to leave later today.",
        "Update my flight to sometime next week then",
        "The next available option is great",
        "what about lodging and transportation?",
        "Yeah i think i'd like an affordable hotel for my week-long stay (7 days). And I'll want to rent a car.",
        "OK could you place a reservation for your recommended hotel? It sounds nice.",
        "yes go ahead and book anything that's moderate expense and has availability.",
        "Now for a car, what are my options?",
        "Awesome let's just get the cheapest option. Go ahead and book for 7 days",
        "Cool so now what recommendations do you have on excursions?",
        "Are they available while I'm there?",
        "interesting - i like the museums, what options are there? ",
        "OK great pick one and book it for my second day there.",
    ]


    _printed = set()
    # 我们可以重复使用第 1 部分中的教程问题来查看其效果。
    for question in tutorial_questions:
        events = part_3_graph.stream(
            {"messages": ("user", question)}, config, stream_mode="values"
        )
        for event in events:
            _print_event(event, _printed)
        snapshot = part_3_graph.get_state(config)
        while snapshot.next:
            # 我们遇到中断！代理正在尝试使用一个工具，用户可以批准或拒绝它。
            # 注意：这段代码完全在你的图表之外。通常，你会将输出流式传输到 UI。
            # 然后，当用户提供输入时，你会让前端通过 API 调用触发新的运行。
            try:
                user_input = input(
                    "Do you approve of the above actions? Type 'y' to continue;"
                    " otherwise, explain your requested changed.\n\n"
                )
            except:
                user_input = "y"
            if user_input.strip() == "y":
                # Just continue
                result = part_3_graph.invoke(
                    None,
                    config,
                )
            else:

                # 通过提供有关所请求的更改/改变主意的说明来满足工具调用
                result = part_3_graph.invoke(
                    {
                        "messages": [
                            ToolMessage(
                                tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                                content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                            )
                        ]
                    },
                    config,
                )
            snapshot = part_3_graph.get_state(config)
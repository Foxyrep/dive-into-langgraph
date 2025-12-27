import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

_ = load_dotenv()

# 配置大模型服务
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen3-coder-plus",
    temperature=0.7,
)

# 定义工具函数
@tool
def get_weather(city: str) -> str:
    """简单天气查询工具"""
    return f"It's always sunny in {city}!"

# 创建工具节点
tools = [get_weather]
tool_node = ToolNode(tools)

# 创建助手节点
def assistant(state: MessagesState, config: RunnableConfig):
    system_prompt = "You are a helpful assistant that can check weather."
    all_messages = [SystemMessage(system_prompt)] + state["messages"]
    model = llm.bind_tools(tools)
    return {"messages": [model.invoke(all_messages)]}

# 创建条件边
def should_continue(state: MessagesState, config: RunnableConfig):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "continue"
    return "end"

# 构建状态图
def build_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tool", tool_node)
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        should_continue,
        {
            "continue": "tool",
            "end": END,
        },
    )
    builder.add_edge("tool", "assistant")
    return builder.compile(name="weather-graph")

# 运行状态图
def demo(query: str = "上海天气怎么样？"):
    graph = build_graph()

    # 运行状态图
    response = graph.invoke({"messages": [HumanMessage(content=query)]})
    print(response["messages"])


if __name__ == "__main__":
    demo()
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from operator import add
import json

_ = load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    temperature=0.3,
)

class OrderState(TypedDict):
    messages: Annotated[List[BaseMessage], add]
    extracted_fields: Dict[str, Any]
    missing_fields: List[str]

_order_counter = 1000

@tool
def create_sales_order(product_code: str, color: str, quantity: int, customer: str) -> str:
    """创建销售单
    
    Args:
        product_code: 款号
        color: 颜色
        quantity: 条数
        customer: 客户名称
    
    Returns:
        订单创建结果
    """
    global _order_counter
    _order_counter += 1
    order_id = f"SO202512{_order_counter:06d}"
    result = {
        "success": True,
        "order_id": order_id,
        "product_code": product_code,
        "color": color,
        "quantity": quantity,
        "customer": customer,
        "message": "订单创建成功"
    }
    return json.dumps(result, ensure_ascii=False)

@tool
def track_sales_order(order_id: str) -> str:
    """查询销售单状态
    
    Args:
        order_id: 订单编号
    
    Returns:
        订单状态信息
    """
    result = {
        "order_id": order_id,
        "status": "已确认",
        "message": "订单状态查询成功"
    }
    return json.dumps(result, ensure_ascii=False)

@tool
def get_product_info(product_code: str) -> str:
    """根据款号获取产品信息
    
    Args:
        product_code: 款号
    
    Returns:
        产品信息
    """
    products = {
        "A001": {"product_name": "经典款T恤", "price": 99.00, "available_colors": ["红色", "蓝色", "黑色", "白色"], "stock": 500},
        "A002": {"product_name": "休闲款卫衣", "price": 159.00, "available_colors": ["红色", "蓝色", "灰色"], "stock": 300},
        "A003": {"product_name": "运动款外套", "price": 299.00, "available_colors": ["黑色", "白色", "绿色"], "stock": 200},
    }
    product = products.get(product_code, {"product_name": "未知产品", "price": 0, "available_colors": [], "stock": 0})
    result = {
        "product_code": product_code,
        "product_name": product["product_name"],
        "price": product["price"],
        "available_colors": product["available_colors"],
        "stock": product["stock"]
    }
    return json.dumps(result, ensure_ascii=False)

@tool
def get_color_info(product_code: str, color: str) -> str:
    """获取产品颜色信息
    
    Args:
        product_code: 款号
        color: 颜色
    
    Returns:
        颜色信息
    """
    color_codes = {"红色": "#FF0000", "蓝色": "#0000FF", "黑色": "#000000", "白色": "#FFFFFF", "灰色": "#808080", "绿色": "#008000"}
    color_code = color_codes.get(color, "#000000")
    result = {
        "product_code": product_code,
        "color": color,
        "color_code": color_code,
        "stock": 100,
        "available": True
    }
    return json.dumps(result, ensure_ascii=False)

order_tools = [create_sales_order, track_sales_order, get_product_info, get_color_info]
tool_node = ToolNode(order_tools)

def extract_fields(state: OrderState, config: RunnableConfig):
    system_prompt = """你是一个订单助手，负责从用户的订单文本中提取订单字段。
需要提取的字段包括：
- 款号 (product_code): 产品唯一标识，如 A001, A002
- 颜色 (color): 产品颜色，如 红色, 蓝色
- 条数 (quantity): 订购数量，整数
- 客户 (customer): 客户名称

请仔细分析用户输入，提取所有可识别的字段。
如果某个字段无法从文本中识别，请将其标记为缺失。
返回格式为JSON，包含 extracted_fields 和 missing_fields 两个字段。
只返回JSON，不要包含其他文字说明。"""
    
    all_messages = [SystemMessage(system_prompt)] + state["messages"]
    response = llm.invoke(all_messages)
    
    content = response.content if hasattr(response, 'content') else str(response)
    
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        extracted_data = json.loads(content)
        extracted_fields = extracted_data.get("extracted_fields", {})
        missing_fields = extracted_data.get("missing_fields", [])
    except json.JSONDecodeError:
        extracted_fields = {}
        missing_fields = ["款号", "颜色", "条数", "客户"]
    
    state["extracted_fields"] = extracted_fields
    state["missing_fields"] = missing_fields
    
    return {"messages": [response], "extracted_fields": extracted_fields, "missing_fields": missing_fields}

def confirm_fields(state: OrderState, config: RunnableConfig):
    extracted_fields = state.get("extracted_fields", {})
    missing_fields = state.get("missing_fields", [])
    
    field_names = {"product_code": "款号", "color": "颜色", "quantity": "条数", "customer": "客户"}
    
    confirm_message = "我已提取到以下订单信息：\n"
    for key, display_name in field_names.items():
        value = extracted_fields.get(key)
        if value is None or value == "" or value == "[缺失]" or str(value) == "[缺失]":
            value = "[缺失]"
        confirm_message += f"{display_name}: {value}\n"
    
    if missing_fields:
        confirm_message += f"\n缺失字段: {', '.join(missing_fields)}\n"
    
    confirm_message += "\n请确认以上信息是否正确？\n"
    confirm_message += '输入 "确认" 继续，"修改" 修改字段，"取消" 取消订单'
    
    return {"messages": [AIMessage(content=confirm_message)]}

def handle_user_action(state: OrderState, config: RunnableConfig):
    last_message = state["messages"][-1]
    content = last_message.content.lower() if isinstance(last_message, HumanMessage) else ""
    
    if "确认" in content or "confirm" in content:
        return {"user_action": "confirm"}
    elif "修改" in content or "modify" in content:
        return {"user_action": "modify"}
    elif "取消" in content or "cancel" in content:
        return {"user_action": "cancel"}
    else:
        return {"user_action": "unknown"}

def process_confirm(state: OrderState, config: RunnableConfig):
    extracted_fields = state.get("extracted_fields", {})
    
    product_code = extracted_fields.get("product_code", "")
    color = extracted_fields.get("color", "")
    quantity = extracted_fields.get("quantity", 0)
    customer = extracted_fields.get("customer", "")
    
    result = create_sales_order.invoke({"product_code": product_code, "color": color, "quantity": quantity, "customer": customer})
    order_data = json.loads(result)
    
    success_message = f"订单创建成功！\n"
    success_message += f"订单编号: {order_data['order_id']}\n"
    success_message += f"款号: {product_code}\n"
    success_message += f"颜色: {color}\n"
    success_message += f"条数: {quantity}\n"
    success_message += f"客户: {customer}"
    
    return {"messages": [AIMessage(content=success_message)]}

def process_modify(state: OrderState, config: RunnableConfig):
    last_message = state["messages"][-1]
    content = last_message.content if isinstance(last_message, HumanMessage) else ""
    
    system_prompt = f"""你是一个订单助手，用户想要修改订单字段。
用户输入: {content}

请分析用户的修改意图，提取更新后的订单字段。
返回格式为JSON，包含 extracted_fields 和 missing_fields 两个字段。"""
    
    all_messages = [SystemMessage(system_prompt)] + state["messages"][-2:]
    model = llm.bind_tools(order_tools)
    response = model.invoke(all_messages)
    
    response_content = response.content if hasattr(response, 'content') else str(response)
    
    try:
        if "```json" in response_content:
            response_content = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            response_content = response_content.split("```")[1].split("```")[0].strip()
        
        extracted_data = json.loads(response_content)
        extracted_fields = extracted_data.get("extracted_fields", {})
        missing_fields = extracted_data.get("missing_fields", [])
    except json.JSONDecodeError:
        extracted_fields = state.get("extracted_fields", {})
        missing_fields = state.get("missing_fields", [])
    
    field_names = {"product_code": "款号", "color": "颜色", "quantity": "条数", "customer": "客户"}
    
    confirm_message = "我已更新订单信息：\n"
    for key, display_name in field_names.items():
        value = extracted_fields.get(key, "[缺失]")
        confirm_message += f"{display_name}: {value}\n"
    
    if missing_fields:
        confirm_message += f"\n缺失字段: {', '.join(missing_fields)}\n"
    
    confirm_message += "\n请确认以上信息是否正确？\n"
    confirm_message += '输入 "确认" 继续，"修改" 修改字段，"取消" 取消订单'
    
    return {"messages": [AIMessage(content=confirm_message)], "extracted_fields": extracted_fields, "missing_fields": missing_fields}

def process_cancel(state: OrderState, config: RunnableConfig):
    cancel_message = "订单已取消。"
    return {"messages": [AIMessage(content=cancel_message)]}

def build_order_graph():
    builder = StateGraph(OrderState)
    
    builder.add_node("extract_fields", extract_fields)
    builder.add_node("confirm_fields", confirm_fields)
    builder.add_node("process_confirm", process_confirm)
    builder.add_node("process_modify", process_modify)
    builder.add_node("process_cancel", process_cancel)
    
    builder.add_edge(START, "extract_fields")
    builder.add_edge("extract_fields", "confirm_fields")
    builder.add_edge("confirm_fields", END)
    builder.add_edge("process_confirm", END)
    builder.add_edge("process_modify", END)
    builder.add_edge("process_cancel", END)
    
    return builder.compile(name="order-graph")

def demo(query: str = "客户张三要10条红色A001款"):
    graph = build_order_graph()
    
    print(f"用户: {query}\n")
    
    state = {"messages": [HumanMessage(content=query)], "extracted_fields": {}, "missing_fields": []}
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"Agent: {msg.content}\n")
    
    extracted_fields = result.get("extracted_fields", {})
    missing_fields = result.get("missing_fields", [])
    
    max_turns = 10
    turn = 0
    
    while turn < max_turns:
        if "订单已取消。" in result.get("messages", [])[-1].content:
            break
        
        if "订单创建成功" in result.get("messages", [])[-1].content:
            break
        
        user_input = input("用户: ")
        if user_input.lower() in ["退出", "exit", "quit"]:
            break
        
        if "确认" in user_input or "confirm" in user_input.lower():
            result = process_confirm({"messages": [], "extracted_fields": extracted_fields}, {})
            for msg in result["messages"]:
                if isinstance(msg, AIMessage):
                    print(f"Agent: {msg.content}\n")
            break
        elif "修改" in user_input or "modify" in user_input.lower():
            state["messages"] = result["messages"] + [HumanMessage(content=user_input)]
            result = process_modify(state, {})
            for msg in result["messages"]:
                if isinstance(msg, AIMessage):
                    print(f"Agent: {msg.content}\n")
            extracted_fields = result.get("extracted_fields", extracted_fields)
            missing_fields = result.get("missing_fields", missing_fields)
        elif "取消" in user_input or "cancel" in user_input.lower():
            result = process_cancel({}, {})
            for msg in result["messages"]:
                if isinstance(msg, AIMessage):
                    print(f"Agent: {msg.content}\n")
            break
        else:
            print("Agent: 请输入 '确认'、'修改' 或 '取消'\n")
        
        turn += 1

if __name__ == "__main__":
    print("=" * 50)
    print("订单助手 - LangGraph Agent")
    print("=" * 50)
    print("\n示例输入:")
    print("- 客户张三要10条红色A001款")
    print("- 客户李四要5条A002款")
    print("\n输入 '退出' 结束对话\n")
    print("=" * 50)
    
    query = input("请输入订单信息: ")
    if query:
        demo(query)

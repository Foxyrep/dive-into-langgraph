import os
import uuid
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

@tool
def extract_order_info(user_input: str) -> str:
    """提取订单信息，包括产品款号、产品颜色、客户名称、产品条数"""
    # 这里使用LLM来提取结构化信息
    llm = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        model="qwen3-max",
    )
    
    prompt = f"""
    从以下用户输入中提取订单信息，返回JSON格式：
    用户输入：{user_input}
    
    需要提取的信息：
    - product_code: 产品款号
    - product_color: 产品颜色  
    - customer_name: 客户名称
    - quantity: 产品条数（数字）
    
    如果某些信息缺失，返回空字符串或0。
    """
    
    response = llm.invoke(prompt)
    return response.content

@tool  
def create_order(order_info: Dict[str, Any]) -> str:
    """创建订单"""
    return f"订单已创建：{order_info}"

@tool
def confirm_order(order_info: str) -> str:
    """确认订单信息，等待用户审批"""
    return f"订单信息待确认：{order_info}"

def create_approval_agent():
    """创建带审批功能的订单智能体"""
    
    # 配置大模型
    llm = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        model="qwen3-coder-plus",
    )
    
    # 创建审批智能体
    agent = create_agent(
        model=llm,
        tools=[extract_order_info, confirm_order, create_order],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    # 确认订单需要审批（此时 extract_order_info 已经执行完成）
                    "confirm_order": {
                        "allowed_decisions": ["approve", "edit", "reject"]  
                    }
                },
                description_prefix="订单处理需要您的审批",
            ),
        ],
        checkpointer=InMemorySaver(),
        system_prompt="""你是一个订单处理助手。你的工作流程是：
        1. 接收用户的订单信息输入
        2. 使用extract_order_info工具提取结构化订单信息
        3. 使用confirm_order工具展示提取的订单信息并等待用户审批
        4. 如果用户确认，使用create_order工具创建订单
        5. 如果用户取消，则结束流程
        
        请礼貌地与用户交流，确保订单信息准确无误。"""
    )
    
    return agent

def main():
    """主函数 - 交互式订单审批流程"""
    try:
        # 创建审批智能体
        agent = create_approval_agent()
        
        # 配置线程
        config = {'configurable': {'thread_id': str(uuid.uuid4())}}
        
        print("=== 订单审批智能体 ===")
        print("请输入订单信息（输入 'quit' 退出）：")
        
        while True:
            user_input = input("\n请输入订单信息: ").strip()
            
            if user_input.lower() == 'quit':
                print("感谢使用订单审批智能体！")
                break
                
            if not user_input:
                print("请输入有效的订单信息")
                continue
                
            print(f"\n正在处理：{user_input}")
            
            # 第一次调用 - 提取订单信息并触发审批
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config
            )
            
            # 检查是否触发中断
            interrupt_info = result.get('__interrupt__')
            if interrupt_info:
                print("\n[智能体] 已提取订单信息，需要您的审批：")
                
                # 从消息历史中获取 extract_order_info 的返回结果
                messages = result.get('messages', [])
                extracted_info = {}
                
                for msg in messages:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            if tool_call.get('name') == 'extract_order_info':
                                # 查找对应的工具响应消息
                                tool_call_id = tool_call.get('id')
                                for response_msg in messages:
                                    if hasattr(response_msg, 'tool_call_id') and response_msg.tool_call_id == tool_call_id:
                                        tool_output = response_msg.content
                                        print(f"调试 - extract_order_info 返回: {tool_output}")
                                        import json
                                        try:
                                            extracted_info = json.loads(tool_output)
                                        except:
                                            pass
                                        break
                                break
                        if extracted_info:
                            break
                
                # 显示提取结果
                print("\n[已提取的订单信息]")
                print(f"款号：{extracted_info.get('product_code', '-')}")
                print(f"颜色：{extracted_info.get('product_color', '-')}")
                print(f"条数：{extracted_info.get('quantity', '-')}")
                print(f"客户：{extracted_info.get('customer_name', '-')}")
                
                print("\n请选择操作：")
                print("1. 确认 - 批准此订单")
                print("2. 修改 - 修改订单信息") 
                print("3. 取消 - 取消此订单")
                
                while True:
                    choice = input("\n请输入选择 (1/2/3): ").strip()
                    if choice == "1":
                        decision = "approve"
                        break
                    elif choice == "2":
                        decision = "edit"
                        break
                    elif choice == "3":
                        decision = "reject"
                        break
                    else:
                        print("无效选择，请输入 1、2 或 3")
                
                # 恢复执行
                print(f"\n用户选择：{choice}")
                
                # 如果选择修改，需要提供修改后的订单信息
                if decision == "edit":
                    print("\n请输入修改后的订单信息：")
                    modified_input = input("修改后的订单信息: ").strip()
                    if modified_input:
                        result = agent.invoke(
                            Command(resume={"decisions": [{"type": decision, "edited_action": {"args": {"user_input": modified_input}}}]}),
                            config=config
                        )
                    else:
                        print("未输入修改内容，取消修改")
                        result = agent.invoke(
                            Command(resume={"decisions": [{"type": "reject"}]}),
                            config=config
                        )
                else:
                    result = agent.invoke(
                        Command(resume={"decisions": [{"type": decision}]}),
                        config=config
                    )
                
                # 显示最终结果
                final_message = result['messages'][-1].content
                print(f"\n最终结果：{final_message}")
                
                # 如果是确认，重置线程用于下一次交互
                if decision == "approve":
                    config = {'configurable': {'thread_id': str(uuid.uuid4())}}
            else:
                print("未触发审批流程，请重试")
                
    except Exception as e:
        print(f"执行过程中出现错误：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
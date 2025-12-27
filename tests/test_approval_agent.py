#!/usr/bin/env python3
"""
测试审批智能体的简单验证脚本
"""

import subprocess
import sys
import time

def test_approval_agent():
    """测试审批智能体的基本功能"""
    print("🧪 开始测试审批智能体...")
    
    # 启动审批智能体进程
    process = subprocess.Popen(
        [sys.executable, "approval_agent.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # 测试输入
        test_inputs = [
            "我想订购100件红色的产品款号ABC123，客户是张三",
            "1",  # 选择确认
            "quit"  # 退出
        ]
        
        # 发送输入并获取输出
        stdout, stderr = process.communicate(
            input="\n".join(test_inputs) + "\n",
            timeout=30
        )
        
        print("✅ 智能体输出：")
        print(stdout)
        
        if stderr:
            print("⚠️ 错误输出：")
            print(stderr)
            
        # 检查关键输出
        success_indicators = [
            "订单审批智能体",
            "请输入订单信息",
            "🤖 智能体已提取订单信息",
            "需要您的审批",
            "请选择操作"
        ]
        
        found_indicators = sum(1 for indicator in success_indicators if indicator in stdout)
        
        if found_indicators >= 4:
            print("✅ 测试通过！审批智能体正常工作")
            return True
        else:
            print(f"❌ 测试失败！只找到 {found_indicators}/5 个关键指示符")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 测试超时")
        process.kill()
        return False
    except Exception as e:
        print(f"❌ 测试出错：{e}")
        return False
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()

if __name__ == "__main__":
    success = test_approval_agent()
    sys.exit(0 if success else 1)
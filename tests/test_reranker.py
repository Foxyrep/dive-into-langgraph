"""
测试阿里云Reranker Client
"""
import asyncio
import os
from dotenv import load_dotenv
from aliyun_reranker_client import AliyunRerankerClient

# 加载环境变量
load_dotenv()

async def test_reranker():
    # 初始化阿里云Reranker Client
    reranker = AliyunRerankerClient(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen3-rerank"
    )
    
    # 测试数据
    query = "什么是人工智能？"
    passages = [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "苹果是一种水果，富含维生素和纤维，有助于健康。",
        "机器学习是人工智能的一个重要子领域，它使计算机能够从数据中学习并做出预测或决策。",
        "今天天气很好，适合户外活动。",
        "深度学习是机器学习的一个分支，它模仿人脑神经网络的工作方式。"
    ]
    
    print("查询:", query)
    print("\n原始文档:")
    for i, passage in enumerate(passages):
        print(f"{i+1}. {passage}")
    
    # 执行重排序
    ranked_results = await reranker.rank(query, passages)
    
    print("\n重排序后的结果:")
    for i, (passage, score) in enumerate(ranked_results):
        print(f"{i+1}. [得分: {score:.4f}] {passage}")

if __name__ == "__main__":
    asyncio.run(test_reranker())
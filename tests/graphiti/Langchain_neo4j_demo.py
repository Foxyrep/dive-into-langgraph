# 将这两个都从 langchain_neo4j 导入
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI

# =================配置区域=================

# 1. Neo4j 数据库配置 (请修改为你真实的数据库信息)
NEO4J_URI = "bolt://120.-.-.-:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "neo4j@-"  # 替换为你的密码

# 2. 硅基流动 API 配置
SILICONFLOW_API_KEY = 'sk-kodzewuwqkxlypmg-'
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"  # 推荐使用 DeepSeek V3 或 Qwen 2.5-72B


# ==========================================

def query_existing_graph():
    # 1. 连接现有的 Neo4j 数据库
    print(f"正在连接数据库: {NEO4J_URI} ...")
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        enhanced_schema=False
    )

    # 【重要】刷新 Schema
    # 这步会读取你数据库里现有的节点和关系类型，告诉 LLM 数据库里有哪些字段
    graph.refresh_schema()
    print("数据库 Schema 加载完成。")
    # 如果你想看 LLM 此时看到了什么结构，可以取消下面这行的注释
    # print(graph.schema)

    # 2. 初始化 LLM
    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=SILICONFLOW_API_KEY,
        openai_api_base=SILICONFLOW_BASE_URL,
        temperature=0
    )

    # 3. 创建问答链
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,  # 设为 True 可以看到生成的 Cypher 语句，方便调试
        allow_dangerous_requests=True
    )

    # 4. 交互式循环提问
    print("\n=== 系统已就绪，请输入问题 (输入 'exit' 退出) ===")
    while True:
        user_input = input("\n请输入问题: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        try:
            # 执行查询
            response = chain.invoke(user_input)
            print(f">>> 回答: {response['result']}")
        except Exception as e:
            print(f"查询出错: {e}")


if __name__ == "__main__":
    query_existing_graph()

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool

# 加载模型配置
_ = load_dotenv()

# 配置大模型
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen3-coder-plus",
    temperature=0,
)

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)

class DashScopeEmbeddings(Embeddings):
    """DashScope 兼容的 Embeddings 封装。"""
    def __init__(self, model: str = "text-embedding-v4", dimensions: int = 1024):
        self.model = model
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        # 批处理避免超时
        for i in range(0, len(texts), 5):
            chunk = texts[i : i + 5]
            try:
                response = client.embeddings.create(
                    model=self.model,
                    input=chunk,
                    dimensions=self.dimensions,
                )
                vectors.extend([item.embedding for item in response.data])
            except Exception as e:
                print(f"Embedding error: {e}")
                # 简单容错，实际生产需重试机制
                vectors.extend([[0.0]*self.dimensions] * len(chunk))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = client.embeddings.create(
            model=self.model,
            input=[text],
            dimensions=self.dimensions,
        )
        return response.data[0].embedding


def load_txt_documents(data_dir: Path) -> list[Document]:
    """读取目录下的 txt 文件，提取元数据，清洗正文。"""

    def split_on_blank(text: str) -> Iterable[str]:
        # 使用正则按空行分割（兼容 \r\n）
        for block in re.split(r"\n\s*\n", text):
            cleaned = block.strip()
            if cleaned:
                yield cleaned

    def parse_block(text_block: str) -> tuple[str, dict]:
        """
        核心修改：
        1. 提取 权限、关键词 到 metadata。
        2. 返回的正文剔除这些元数据行，减少 Embedding 噪音。
        """
        lines = text_block.split('\n')
        content_lines = []
        metadata = {}
        
        for line in lines:
            line = line.strip()
            if not line: continue

            # 提取权限 (支持中文冒号和英文冒号)
            if line.startswith(("权限:", "权限：")):
                # 统一分隔符，将顿号、逗号都转为列表
                raw_perm = line.split(':', 1)[1].strip()
                # 使用正则分割：顿号、逗号、空格
                perm_list = re.split(r'[、,，\s]+', raw_perm)
                # 去除空字符串
                metadata["permissions"] = [p for p in perm_list if p]
            
            # 提取关键词
            elif line.startswith(("关键词:", "关键词：")):
                metadata["keywords"] = line.split(':', 1)[1].strip()
            
            # 保留正文 (问题和答案)
            else:
                content_lines.append(line)
        
        return "\n".join(content_lines), metadata

    documents: list[Document] = []

    # 确保目录存在
    if not data_dir.exists():
        print(f"警告：目录 {data_dir} 不存在，跳过加载。")
        return []

    for path in sorted(data_dir.glob("*.txt")):
        print(f"正在读取文件: {path.absolute()}") 
        content = path.read_text(encoding="utf-8")
        
        for idx, part in enumerate(split_on_blank(content)):
            clean_content, extracted_meta = parse_block(part)
            
            final_metadata = {
                "source": path.name, 
                "chunk_id": idx,
                **extracted_meta 
            }
            
            documents.append(
                Document(
                    page_content=clean_content, # 这里只包含 问题和答案
                    metadata=final_metadata,
                )
            )
            
    return documents


def build_vector_store(data_dir: Path | None = None) -> InMemoryVectorStore:
    target_dir = data_dir or Path(__file__).parent.parent / "files"
    documents = load_txt_documents(target_dir)

    if not documents:
        print("未加载到任何文档，请检查 files 目录。")
        return InMemoryVectorStore(embedding=DashScopeEmbeddings())

    print(f"成功加载 {len(documents)} 个文档片段")
    
    embeddings = DashScopeEmbeddings()
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(documents)
    
    return vector_store


def create_react_agent(vector_store: InMemoryVectorStore, user_permission: str):

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """检索知识库。"""
        
        # === 核心修改：修复过滤逻辑 ===
        def filter_func(doc: Document) -> bool:
            # 获取文档的权限列表 (在 parse_block 中我们已经将其转为了 list)
            doc_perms = doc.metadata.get("permissions", [])
            
            # 1. 如果文档没有设置权限，默认为公开，返回 True
            if not doc_perms: 
                return True 
            
            # 2. 检查用户权限是否在文档允许的列表中
            # 例如：user="番禺大货仓" in ["番禺大货仓", "色卡组"...] -> True
            return user_permission in doc_perms

        print(f"\n[检索中] 用户权限: {user_permission}, 查询: {query}")
        
        # 执行检索
        retrieved = vector_store.similarity_search(
            query, 
            k=3, 
            filter=filter_func # 传入过滤函数
        )
        
        if not retrieved:
            return "没有找到相关且您有权限查看的文档。", []

        # 格式化上下文
        serialized = "\n\n".join(
            f"---片段 {i+1}---\n{doc.page_content}"
            for i, doc in enumerate(retrieved)
        )
        return serialized, retrieved

    return create_agent(
        llm,
        tools=[retrieve_context],
        system_prompt=(
            "你是一个企业知识问答助手。"
            "必须优先根据检索到的【参考资料】回答用户问题。"
            "只输出用户权限组内的文档内容，不输出其他权限组的文档。"
            "不清晰或有多个相似回答的，需要咨询用户进行确认。"
        ),
    )


def run_demo():
    # 1. 构建向量库
    # 假设当前脚本同级目录下有 files 文件夹
    vector_store = build_vector_store()
    print('\n=== 向量库准备就绪 ===\n')

    query = "怎么考勤？"

    # 场景测试
    print(f"--- 场景测试: 用户权限 = [IT组] ---")
    
    # 传入用户权限
    agent = create_react_agent(vector_store, user_permission="IT组")
    
    # 执行对话
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    
    print("\n=== 最终回答 ===")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    run_demo()
import asyncio
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

# 测试 Graphiti 核心功能
# 注意：新版是graphiti_core，旧版是graphiti
from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

# 测试环境配置
OPENAI_API_KEY = "sk-kodzewuwqkxlypmgegdjdgvhwntqfegmcamipvcoylribmss"
OPENAI_API_BASE = "https://api.siliconflow.cn/v1"
OPENAI_API_MODEL = "zai-org/GLM-4.6"
OPENAI_API_SMALL_MODEL = "zai-org/GLM-4.6"
OPENAI_API_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
OPENAI_API_EMBEDDING_DIM = 4096

# 初始化 LLM 客户端
llm_config = LLMConfig(
    api_key=OPENAI_API_KEY,
    model=OPENAI_API_MODEL,
    small_model=OPENAI_API_SMALL_MODEL,
    base_url=OPENAI_API_BASE,
)
llm_client = OpenAIGenericClient(config=llm_config)

# 初始化 Embedding 客户端
embed_config = OpenAIEmbedderConfig(
    api_key=OPENAI_API_KEY,
    embedding_model=OPENAI_API_EMBEDDING_MODEL,
    embedding_dim=OPENAI_API_EMBEDDING_DIM,
    base_url=OPENAI_API_BASE,
)
embedder = OpenAIEmbedder(embed_config)

# 初始化 Cross-Encoder 客户端
cross_encoder = OpenAIRerankerClient(client=llm_client.client, config=llm_config)

# 图数据库
# 图数据库
from graphiti_core.driver.neo4j_driver import Neo4jDriver
# driver = Neo4jDriver(
#     uri="bolt://localhost:7687",
#     user="neo4j",
#     password="12345678",
#     database="neo4j_test"  # Custom database name
# )

# 初始化 Graphiti 客户端
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="12345678",
    # graph_driver=driver,
    llm_client=llm_client,
    embedder=embedder,
    cross_encoder=cross_encoder
)

# 定义人实体模型
class Person(BaseModel):
    """A person entity with biographical information."""
    age: Optional[int] = Field(None, description="Age of the person")
    occupation: Optional[str] = Field(None, description="Current occupation")
    location: Optional[str] = Field(None, description="Current location")
    birth_date: Optional[datetime] = Field(None, description="Date of birth")

# 定义公司实体模型
class Company(BaseModel):
    """A business organization."""
    industry: Optional[str] = Field(None, description="Primary industry")
    founded_year: Optional[int] = Field(None, description="Year company was founded")
    headquarters: Optional[str] = Field(None, description="Location of headquarters")
    employee_count: Optional[int] = Field(None, description="Number of employees")

# 定义产品实体模型
class Product(BaseModel):
    """A product or service."""
    category: Optional[str] = Field(None, description="Product category")
    price: Optional[float] = Field(None, description="Price in USD")
    release_date: Optional[datetime] = Field(None, description="Product release date")

# 定义就业关系实体模型
class Employment(BaseModel):
    """Employment relationship between a person and company."""
    position: Optional[str] = Field(None, description="Job title or position")
    start_date: Optional[datetime] = Field(None, description="Employment start date")
    end_date: Optional[datetime] = Field(None, description="Employment end date")
    salary: Optional[float] = Field(None, description="Annual salary in USD")
    is_current: Optional[bool] = Field(None, description="Whether employment is current")

# 定义投资关系实体模型
class Investment(BaseModel):
    """Investment relationship between entities."""
    amount: Optional[float] = Field(None, description="Investment amount in USD")
    investment_type: Optional[str] = Field(
        None, description="Type of investment (equity, debt, etc.)"
    )
    stake_percentage: Optional[float] = Field(
        None, description="Percentage ownership"
    )
    investment_date: Optional[datetime] = Field(
        None, description="Date of investment"
    )

# 定义合作关系实体模型
class Partnership(BaseModel):
    """Partnership relationship between companies."""
    partnership_type: Optional[str] = Field(None, description="Type of partnership")
    duration: Optional[str] = Field(None, description="Expected duration")
    deal_value: Optional[float] = Field(None, description="Financial value of partnership")


async def main():
    try:
        # 定义实体类型映射
        entity_types = {
            "Person": Person,
            "Company": Company,
            "Product": Product,
        }
        # 定义关系类型映射
        edge_types = {
            "Employment": Employment,
            "Investment": Investment,
            "Partnership": Partnership,
        }
        # 定义关系类型映射
        edge_type_map = {
            ("Person", "Company"): ["Employment"],
            ("Company", "Company"): ["Partnership", "Investment"],
            ("Person", "Person"): ["Partnership"],
            ("Entity", "Entity"): ["Investment"],
        }
        # 插入知识图谱
        await graphiti.add_episode(
            name="Business Update",
            episode_body=(
                "Sarah joined TechCorp as CTO in January 2023 with a $200K salary. "
                "TechCorp partnered with DataCorp in a $5M deal."
            ),
            source_description="Business news",
            reference_time=datetime.now(),
            entity_types=entity_types,
            edge_types=edge_types,
            edge_type_map=edge_type_map,
        )

    finally:
        await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main())

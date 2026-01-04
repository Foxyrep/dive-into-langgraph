import asyncio
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

OPENAI_API_KEY = "sk-kodzewuwqkxlypmgegdjdgvhwntqf-"
OPENAI_API_BASE = "https://api.siliconflow.cn/v1"
OPENAI_API_MODEL = "zai-org/GLM-4.6"
OPENAI_API_SMALL_MODEL = "zai-org/GLM-4.6"
OPENAI_API_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
OPENAI_API_EMBEDDING_DIM = 4096

llm_config = LLMConfig(
    api_key=OPENAI_API_KEY,
    model=OPENAI_API_MODEL,
    small_model=OPENAI_API_SMALL_MODEL,
    base_url=OPENAI_API_BASE,
)
llm_client = OpenAIGenericClient(config=llm_config)

embed_config = OpenAIEmbedderConfig(
    api_key=OPENAI_API_KEY,
    embedding_model=OPENAI_API_EMBEDDING_MODEL,
    embedding_dim=OPENAI_API_EMBEDDING_DIM,
    base_url=OPENAI_API_BASE,
)
embedder = OpenAIEmbedder(embed_config)

cross_encoder = OpenAIRerankerClient(client=llm_client.client, config=llm_config)

graphiti = Graphiti(
    uri="bolt://120.-.-.-:7687",
    user="neo4j",
    password="neo4j@-",
    llm_client=llm_client,
    embedder=embedder,
    cross_encoder=cross_encoder,
)
class Person(BaseModel):
    """A person entity with biographical information."""
    age: Optional[int] = Field(None, description="Age of the person")
    occupation: Optional[str] = Field(None, description="Current occupation")
    location: Optional[str] = Field(None, description="Current location")
    birth_date: Optional[datetime] = Field(None, description="Date of birth")


class Company(BaseModel):
    """A business organization."""
    industry: Optional[str] = Field(None, description="Primary industry")
    founded_year: Optional[int] = Field(None, description="Year company was founded")
    headquarters: Optional[str] = Field(None, description="Location of headquarters")
    employee_count: Optional[int] = Field(None, description="Number of employees")


class Product(BaseModel):
    """A product or service."""
    category: Optional[str] = Field(None, description="Product category")
    price: Optional[float] = Field(None, description="Price in USD")
    release_date: Optional[datetime] = Field(None, description="Product release date")


class Employment(BaseModel):
    """Employment relationship between a person and company."""
    position: Optional[str] = Field(None, description="Job title or position")
    start_date: Optional[datetime] = Field(None, description="Employment start date")
    end_date: Optional[datetime] = Field(None, description="Employment end date")
    salary: Optional[float] = Field(None, description="Annual salary in USD")
    is_current: Optional[bool] = Field(None, description="Whether employment is current")


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


class Partnership(BaseModel):
    """Partnership relationship between companies."""
    partnership_type: Optional[str] = Field(None, description="Type of partnership")
    duration: Optional[str] = Field(None, description="Expected duration")
    deal_value: Optional[float] = Field(None, description="Financial value of partnership")


async def main():
    try:
        entity_types = {
            "Person": Person,
            "Company": Company,
            "Product": Product,
        }

        edge_types = {
            "Employment": Employment,
            "Investment": Investment,
            "Partnership": Partnership,
        }

        edge_type_map = {
            ("Person", "Company"): ["Employment"],
            ("Company", "Company"): ["Partnership", "Investment"],
            ("Person", "Person"): ["Partnership"],
            ("Entity", "Entity"): ["Investment"],
        }

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

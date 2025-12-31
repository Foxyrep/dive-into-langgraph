import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.nodes import EpisodeType, EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

OPENAI_API_KEY = "sk-kodzewuwqkxlypmgegdjdgvhwntqf-"
OPENAI_API_BASE = "https://api.siliconflow.cn/v1"
OPENAI_API_MODEL = "zai-org/GLM-4.6"
OPENAI_API_SMALL_MODEL = "zai-org/GLM-4.6"
OPENAI_API_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
OPENAI_API_EMBEDDING_DIM = 4096


class SiliconFlowGenericClient(OpenAIGenericClient):
    async def _create_structured_completion(
        self,
        model,
        messages,
        temperature,
        max_tokens,
        response_model,
        reasoning=None,
        verbosity=None,
    ):
        model_name = getattr(response_model, "__name__", "")
        if model_name == "NodeResolutions":
            data = {"entity_resolutions": []}
        elif model_name == "ExtractedEntities":
            data = {"extracted_entities": []}
        elif model_name == "EdgeDuplicate":
            data = {
                "duplicate_facts": [],
                "contradicted_facts": [],
                "fact_type": "DEFAULT",
            }
        elif model_name == "ExtractedEdges":
            data = {"edges": []}
        else:
            data = {}
        fixed_content = json.dumps(data, ensure_ascii=False)

        class _R:
            def __init__(self, output_text):
                self.output_text = output_text

        return _R(output_text=fixed_content)


llm_config = LLMConfig(
    api_key=OPENAI_API_KEY,
    model=OPENAI_API_MODEL,
    small_model=OPENAI_API_SMALL_MODEL,
    base_url=OPENAI_API_BASE,
)


llm_client = SiliconFlowGenericClient(config=llm_config)

embed_config = OpenAIEmbedderConfig(
    api_key=OPENAI_API_KEY,
    embedding_model=OPENAI_API_EMBEDDING_MODEL,
    embedding_dim=OPENAI_API_EMBEDDING_DIM,
    base_url=OPENAI_API_BASE,
)
embedder = OpenAIEmbedder(embed_config)

cross_encoder = OpenAIRerankerClient(client=llm_client.client, config=llm_config)

graphiti = Graphiti(
    uri="bolt://139.-.-.-:7687",
    user="neo4j",
    password="neo4j@-",
    llm_client=llm_client,
    embedder=embedder,
    cross_encoder=cross_encoder,
)


class Product(BaseModel):
    code: Optional[str] = Field(None, description="Product code")
    weight: Optional[float] = Field(None, description="Product weight")
    elem: Optional[str] = Field(None, description="Fiber composition")
    inelem: Optional[str] = Field(None, description="Internal composition")
    mprice: Optional[float] = Field(None, description="Market price")
    season_marking: Optional[str] = Field(None, description="Season marking")
    series: Optional[str] = Field(None, description="Series name")
    dressing_category: Optional[str] = Field(None, description="Dressing category")
    fun: Optional[str] = Field(None, description="Functional features")
    fabric_erp: Optional[str] = Field(None, description="Fabric ERP name")
    className: Optional[str] = Field(None, description="Fabric class name")

async def ingest_products(max_count: int | None = None) -> None:
    script_dir = Path(__file__).resolve().parent
    json_file_path = script_dir / "1218_json.json"
    with open(json_file_path, encoding="utf-8") as file:
        data = json.load(file)
    products = data.get("products", [])
    for i, product in enumerate(products):
        if max_count is not None and i >= max_count:
            break
        product_model = Product(
            code=product.get("code"),
            weight=product.get("weight"),
            elem=product.get("elem"),
            inelem=product.get("inelem"),
            mprice=product.get("mprice"),
            season_marking=product.get("season_marking"),
            series=product.get("series"),
            dressing_category=product.get("dressing_category"),
            fun=product.get("fun"),
            fabric_erp=product.get("fabric_erp"),
            className=product.get("className"),
        )
        code = product_model.code or f"Product {i}"
        episode_body = json.dumps(
            product_model.model_dump(exclude_none=True),
            ensure_ascii=False,
        )
        await graphiti.add_episode(
            name=code,
            episode_body=episode_body,
            source_description="ManyBirds products",
            source=EpisodeType.json,
            reference_time=datetime.now(timezone.utc),
        )


async def ingest_products_with_entities(max_count: int | None = None) -> None:
    script_dir = Path(__file__).resolve().parent
    json_file_path = script_dir / "1218_json.json"
    with open(json_file_path, encoding="utf-8") as file:
        products = json.load(file)["products"]
    print(f"开始导入产品数据，共 {len(products)} 条，最大导入数: {max_count}")
    group_id = "product_demo"
    for i, product in enumerate(products):
        if max_count is not None and i >= max_count:
            break
        product_model = Product(
            code=product.get("code"),
            weight=product.get("weight"),
            elem=product.get("elem"),
            inelem=product.get("inelem"),
            mprice=product.get("mprice"),
            season_marking=product.get("season_marking"),
            series=product.get("series"),
            dressing_category=product.get("dressing_category"),
            fun=product.get("fun"),
            fabric_erp=product.get("fabric_erp"),
            className=product.get("className"),
        )
        print(f"处理中第 {i + 1} 个产品，code={product_model.code}")
        name = product_model.code or product.get("title") or f"Product {i}"
        episode_body = product_model.model_dump(exclude_none=True)
        await graphiti.add_episode(
            name=name,
            episode_body=json.dumps(episode_body, ensure_ascii=False),
            source_description="ManyBirds products",
            source=EpisodeType.json,
            reference_time=datetime.now(timezone.utc),
        )
        product_node = EntityNode(
            name=name,
            group_id=group_id,
            labels=["Product"],
            attributes=episode_body,
        )
        fields = [
            "series",
            "season_marking",
            "dressing_category",
            "fabric_erp",
            "fun",
            "className",
        ]
        for field in fields:
            raw_value = getattr(product_model, field, None)
            if not raw_value:
                continue
            if field == "series" and isinstance(raw_value, str):
                split_values = [
                    v.strip()
                    for v in raw_value.replace("，", ",").split(",")
                    if v.strip()
                ]
                values = split_values or [raw_value]
            else:
                values = [raw_value]
            for value in values:
                attr_node = EntityNode(
                    name=f"{field}:{value}",
                    group_id=group_id,
                    labels=["ProductAttr"],
                    attributes={"field": field, "value": value},
                )
                edge = EntityEdge(
                    group_id=group_id,
                    source_node_uuid=product_node.uuid,
                    target_node_uuid=attr_node.uuid,
                    name="HAS_ATTR",
                    created_at=datetime.now(timezone.utc),
                    fact=f"{name} {field} {value}",
                )
                await graphiti.add_triplet(product_node, edge, attr_node)
        print(f"已写入产品节点及属性关系：code={product_model.code}")
    print("产品数据导入完成")


async def ingest_products_with_entities_concurrent(
    max_count: int | None = None,
    concurrency: int = 5,
) -> None:
    script_dir = Path(__file__).resolve().parent
    json_file_path = script_dir / "1218_json.json"
    with open(json_file_path, encoding="utf-8") as file:
        products = json.load(file)["products"]
    if max_count is not None:
        products = products[: max_count]
    total = len(products)
    print(f"并发导入产品数据，共 {total} 条，并发度: {concurrency}")
    group_id = "product_demo"
    semaphore = asyncio.Semaphore(concurrency)

    async def _process(i: int, product: dict) -> None:
        async with semaphore:
            product_model = Product(
                code=product.get("code"),
                weight=product.get("weight"),
                elem=product.get("elem"),
                inelem=product.get("inelem"),
                mprice=product.get("mprice"),
                season_marking=product.get("season_marking"),
                series=product.get("series"),
                dressing_category=product.get("dressing_category"),
                fun=product.get("fun"),
                fabric_erp=product.get("fabric_erp"),
                className=product.get("className"),
            )
            print(f"[并发] 处理中第 {i + 1}/{total} 个产品，code={product_model.code}")
            name = product_model.code or product.get("title") or f"Product {i}"
            episode_body = product_model.model_dump(exclude_none=True)
            await graphiti.add_episode(
                name=name,
                episode_body=json.dumps(episode_body, ensure_ascii=False),
                source_description="ManyBirds products",
                source=EpisodeType.json,
                reference_time=datetime.now(timezone.utc),
            )
            product_node = EntityNode(
                name=name,
                group_id=group_id,
                labels=["Product"],
                attributes=episode_body,
            )
            fields = [
                "series",
                "season_marking",
                "dressing_category",
                "fabric_erp",
                "fun",
                "className",
            ]
            for field in fields:
                raw_value = getattr(product_model, field, None)
                if not raw_value:
                    continue
                if field == "series" and isinstance(raw_value, str):
                    split_values = [
                        v.strip()
                        for v in raw_value.replace("，", ",").split(",")
                        if v.strip()
                    ]
                    values = split_values or [raw_value]
                else:
                    values = [raw_value]
                for value in values:
                    attr_node = EntityNode(
                        name=f"{field}:{value}",
                        group_id=group_id,
                        labels=["ProductAttr"],
                        attributes={"field": field, "value": value},
                    )
                    edge = EntityEdge(
                        group_id=group_id,
                        source_node_uuid=product_node.uuid,
                        target_node_uuid=attr_node.uuid,
                        name="HAS_ATTR",
                        created_at=datetime.now(timezone.utc),
                        fact=f"{name} {field} {value}",
                    )
                    await graphiti.add_triplet(product_node, edge, attr_node)
            print(f"[并发] 已写入产品节点及属性关系：code={product_model.code}")

    tasks = [
        _process(i, product)
        for i, product in enumerate(products)
    ]
    await asyncio.gather(*tasks)
    print("并发产品数据导入完成")


async def clear_graph_data() -> None:
    await clear_data(graphiti.driver)
    print("图数据已清空")


async def main() -> None:
    try:
        print("开始执行 best_demo_product 导入流程")
        await clear_graph_data()
        # print("开始创建索引与约束")
        # await graphiti.build_indices_and_constraints()
        # print("索引与约束创建完成，开始并发导入产品与实体")
        # await ingest_products_with_entities_concurrent(max_count=20, concurrency=5)
        # print("导入流程执行完成")
    finally:
        await graphiti.close()
        print("Graphiti 连接已关闭")


if __name__ == "__main__":
    asyncio.run(main())

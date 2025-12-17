import asyncio
from graphiti_core import Graphiti
import os

# 硅基流动
OPENAI_API_KEY="sk-kodzewuwqkxlypmgegdjdgvhwntqfegmcamipvcoylribmss"
OPENAI_API_BASE="https://api.siliconflow.cn/v1"
OPENAI_API_MODEL="zai-org/GLM-4.6"
OPENAI_API_SMALL_MODEL="zai-org/GLM-4.6"
OPENAI_API_EMBEDDING_MODEL="BAAI/bge-m3"
# 嵌入模型维度
OPENAI_API_EMBEDDING_DIM=1024

# 开启日志，查看卡在哪一步
import logging
import sys
# ================= 1. 开启调试日志 =================
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# 这一行会让你看到是否卡在 API 请求上
logging.getLogger("httpx").setLevel(logging.INFO)


# 图数据库
from graphiti_core.driver.neo4j_driver import Neo4jDriver

# LLM模型
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
llm_config = LLMConfig(
    api_key=OPENAI_API_KEY,  # OpenAI API key
    model=OPENAI_API_MODEL,
    small_model=OPENAI_API_SMALL_MODEL,
    base_url=OPENAI_API_BASE,  # OpenAI's OpenAI-compatible endpoint

)
llm_client = OpenAIGenericClient(config=llm_config)

# 嵌入模型
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
config=OpenAIEmbedderConfig(
    api_key=OPENAI_API_KEY,  # Placeholder API key
    embedding_model=OPENAI_API_EMBEDDING_MODEL,
    embedding_dim=OPENAI_API_EMBEDDING_DIM,
    base_url=OPENAI_API_BASE,
)
embedder=OpenAIEmbedder(config)

# 重排模型
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config)

# # 图数据库
driver = Neo4jDriver(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="12345678",
    database="neo4j"  # Custom database name
)

# 初始化Graphiti
graphiti = Graphiti(
    # uri="bolt://localhost:7687",
    # user="neo4j",
    # password="12345678",
    # database="neo4j",  # Custom database name
    graph_driver=driver, 
    llm_client=llm_client, 
    embedder=embedder, 
    cross_encoder=cross_encoder
)

# 定义边类型
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType

# 清理数据
from graphiti_core.utils.maintenance.graph_data_operations import clear_data


# 导入产品数据
import json
from datetime import datetime, timezone
from pathlib import Path
async def ingest_products_data(client: Graphiti):
    # script_dir = Path.cwd()
    # json_file_path = script_dir / 'data' / 'products.json'

    # with open(json_file_path) as file:
    #     products = json.load(file)['products']

    products = [
    {
        "elem": "73%棉 27%聚酯纤维",
        "className": "单面",
        "fabric_structure_two": "单面 棉盖丝平纹",
        "production_process": "染定、食毛",
        "fiber_type": "",
        "yarn_type": "",
        "series": "棉涤",
        "quality_level": "合格品",
        "dressing_category": "T恤、户外服、功能夹克、运动上衣",
        "devtype": "未知开发类型",
        "introduce": "",
        "customizable_grade": "",
        "mprice": 13.60,
        "price": 49.00,
        "yprice": 12.50,
        "kgprice": 55.40,
        "taxmprice": 14.80,
        "taxyprice": 13.60,
        "taxkgprice": 60.20,
        "sale_num_year": 0,
        "dnumber": "40",
        "weight": 145,
        "width": "170",
        "inelem": "40S精棉紧密纺 73%+50D/36F涤纶半光低弹丝轻网DTY 27%",
        "code": "6125",
        "name": "韩国精棉",
        "swzoomin": "-6～+2",
        "shzoomin": "-6～+2"
    },
    {
        "elem": "48.9%棉 47%聚酯纤维 4.1%氨纶",
        "className": "提花",
        "fabric_structure_two": "提花 拉架双面小 肌理",
        "production_process": "胚定、染定、手感",
        "fiber_type": "",
        "yarn_type": "",
        "series": "棉涤",
        "quality_level": "合格品",
        "dressing_category": "T恤、户外服、功能夹克、运动上衣、内衣、袜子、高端运动装、医护布",
        "devtype": "未知开发类型",
        "introduce": "",
        "customizable_grade": "",
        "mprice": 23.10,
        "price": 45.00,
        "yprice": 21.20,
        "kgprice": 50.90,
        "taxmprice": 25.10,
        "taxyprice": 23.00,
        "taxkgprice": 55.30,
        "sale_num_year": 0,
        "dnumber": "40",
        "weight": 260,
        "width": "175",
        "inelem": "40STC65/35精梳环锭纺 45.4%+40S精棉环锭纺 33%+75D/36F涤纶半光低弹丝DTY 17.5%+30D氨纶 4.1%",
        "code": "9207",
        "name": "艾美纹-2",
        "swzoomin": "-6～+2",
        "shzoomin": "-6～+2"
    },
    {
        "elem": "41.5%莱赛尔 32.3%粘纤 18.4%绵羊毛 7.8%氨纶",
        "className": "罗纹",
        "fabric_structure_two": "罗纹 拉架8X5",
        "production_process": "胚定、染定",
        "fiber_type": "",
        "yarn_type": "",
        "series": "羊毛,混纺",
        "quality_level": "合格品",
        "dressing_category": "打底",
        "devtype": "未知开发类型",
        "introduce": "",
        "customizable_grade": "",
        "mprice": 30.00,
        "price": 108.00,
        "yprice": 27.40,
        "kgprice": 112.70,
        "taxmprice": 32.60,
        "taxyprice": 29.80,
        "taxkgprice": 122.50,
        "sale_num_year": 0,
        "dnumber": "40",
        "weight": 190,
        "width": "140",
        "inelem": "40S兰精天丝A100 人棉羊毛45/35/20 紧赛纺 92.2%+30D氨纶 7.8%",
        "code": "9217",
        "name": "澳羊毛坑条",
        "swzoomin": "-6～+2",
        "shzoomin": "-6～+2"
    },
    {
        "elem": "94.5%粘纤 5.5%氨纶",
        "className": "罗纹",
        "fabric_structure_two": "罗纹 拉架2X2",
        "production_process": "胚定、染定",
        "fiber_type": "",
        "yarn_type": "",
        "series": "纯纺",
        "quality_level": "非合格品",
        "dressing_category": "打底",
        "devtype": "未知开发类型",
        "introduce": "该产品最大特点是裸感亲肤，不粘身，带来极致的穿着体验。\r\n选用了XG粘胶材料，布面呈现哑光雾面的高级感。几乎接近莫代尔效果，但较莫代尔有很高的性价比。还是一款可再生环保材料，符合当下舒适、环保、健康趋势。\r\n",
        "customizable_grade": "",
        "mprice": 18.80,
        "price": 50.00,
        "yprice": 17.20,
        "kgprice": 56.50,
        "taxmprice": 20.50,
        "taxyprice": 18.70,
        "taxkgprice": 61.40,
        "sale_num_year": 2136,
        "dnumber": "40",
        "weight": 230,
        "width": "145",
        "inelem": "40S人棉XG紧赛纺 94.5%+20D氨纶 5.5%",
        "code": "6219",
        "name": "元气棉",
        "swzoomin": "-7.5～+2",
        "shzoomin": "-7.5～+2"
    },
    {
        "elem": "100%棉",
        "className": "单面",
        "fabric_structure_two": "单面 平纹布",
        "production_process": "改性拉抛、碱缩、浆边、染定、食毛、手感、双面烧毛、无尘、压光",
        "fiber_type": "",
        "yarn_type": "",
        "series": "纯纺",
        "quality_level": "合格品",
        "dressing_category": "T恤、户外服、功能夹克、运动上衣",
        "devtype": "未知开发类型",
        "introduce": "",
        "customizable_grade": "",
        "mprice": 18.80,
        "price": 58.00,
        "yprice": 18.10,
        "kgprice": 63.00,
        "taxmprice": 21.50,
        "taxyprice": 19.70,
        "taxkgprice": 68.50,
        "sale_num_year": 7,
        "dnumber": "26",
        "weight": 170,
        "width": "185",
        "inelem": "26S精棉紧密纺 100%",
        "code": "6250",
        "name": "茧棉",
        "swzoomin": "-6.0~+2.0",
        "shzoomin": "-6.0~+2.0"
    },
    {
        "elem": "46.2%粘纤 46.2%棉 7.6%氨纶",
        "className": "罗纹",
        "fabric_structure_two": "罗纹 拉架2X2",
        "production_process": "胚定、染定、手感",
        "fiber_type": "短纤",
        "yarn_type": "混纺",
        "series": "RC",
        "quality_level": "合格品",
        "dressing_category": "内衣、袜子、T恤、高端运动装、医护布",
        "devtype": "未知开发类型",
        "introduce": "优选优可丝粘胶与精梳棉混纺\r\n柔软亲肤，耐洗舒适，触感持续在线\r\n\r\n棉增强耐磨与抗起球性能\r\n叠加抗静电、透气性佳等基础功能，实穿稳定可靠\r\n\r\n多项物理指标达一等品等级\r\n白色异纤控制在10个以内，高于行业常规标准\r\n入选“中国功能性针织产品流行趋势——舒适性培育推荐产品”\r\n\r\n一款兼具柔感、功能与环保价值的畅销面料",
        "customizable_grade": "可订一等品",
        "mprice": 18.20,
        "price": 48.00,
        "yprice": 16.70,
        "kgprice": 54.30,
        "taxmprice": 18.80,
        "taxyprice": 18.10,
        "taxkgprice": 59.00,
        "sale_num_year": 9766,
        "dnumber": "40",
        "weight": 240,
        "width": "140",
        "inelem": "40SRC50/50紧赛纺 92.4%+30D氨纶 7.6%",
        "code": "6257",
        "name": "优可丝罗纹",
        "swzoomin": "-6～+2",
        "shzoomin": "-6～+2"
    }
    ]

    for i, product in enumerate(products):
        await client.add_episode(
            name=product.get('title', f'Product {i}'),
            episode_body=str({k: v for k, v in product.items() if k != 'images'}),
            source_description='ManyBirds products',
            source=EpisodeType.json,
            reference_time=datetime.now(timezone.utc),
        )
        print(f"✅ 第 {i+1} 条完成！")




async def main():

    # 初始化清理数据
    await clear_data(graphiti.driver)
    await graphiti.build_indices_and_constraints()
    print("✅ 数据清理和索引构建完成！")

    # 导入产品数据
    await ingest_products_data(graphiti)

if __name__ == "__main__":
    asyncio.run(main())

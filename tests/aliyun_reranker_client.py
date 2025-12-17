"""
阿里云百炼平台Reranker Client实现
"""
import asyncio
from typing import List, Tuple
from graphiti_core.cross_encoder.client import CrossEncoderClient

try:
    from dashscope import TextReRank
except ImportError:
    raise ImportError(
        'dashscope is required for AliyunRerankerClient. '
        'Install it with: pip install dashscope'
    ) from None


class AliyunRerankerClient(CrossEncoderClient):
    """
    基于阿里云百炼平台的Reranker Client实现
    使用阿里云百炼平台的gte-rerank或qwen3-rerank模型进行文本重排序
    """
    
    def __init__(self, api_key: str, model: str = "gte-rerank"):
        """
        初始化阿里云Reranker Client
        
        Args:
            api_key (str): 阿里云百炼平台的API密钥
            model (str): 使用的rerank模型名称，默认为gte-rerank
        """
        self.api_key = api_key
        self.model = model
        
    async def rank(self, query: str, passages: List[str]) -> List[Tuple[str, float]]:
        """
        使用阿里云百炼平台的Rerank模型对文档进行重排序
        
        Args:
            query (str): 查询文本
            passages (List[str]): 待排序的文档列表
            
        Returns:
            List[Tuple[str, float]]: 按相关性得分排序的文档列表，每个元素为(文档, 得分)
        """
        if not passages:
            return []
            
        try:
            # 使用DashScope的TextReRank进行重排序
            response = TextReRank.call(
                model=self.model,
                query=query,
                documents=passages,
                api_key=self.api_key
            )
            
            # 提取排序结果
            ranked_results = []
            for item in response.output.results:
                doc_index = item.index
                score = item.relevance_score
                if doc_index < len(passages):
                    ranked_results.append((passages[doc_index], score))
            
            # 按得分降序排序
            ranked_results.sort(key=lambda x: x[1], reverse=True)
            
            return ranked_results
            
        except Exception as e:
            # 如果出现错误，返回默认排序（所有得分设为0）
            return [(passage, 0.0) for passage in passages]
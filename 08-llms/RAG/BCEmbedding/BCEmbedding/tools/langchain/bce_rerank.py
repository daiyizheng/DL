'''
@Description: 
@Author: shenlei
@Date: 2024-01-15 18:56:59
@LastEditTime: 2024-02-24 23:19:02
@LastEditors: shenlei
'''
from __future__ import annotations

from typing import Dict, Optional, Sequence, Any

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra, root_validator

from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

from pydantic.v1 import PrivateAttr


class BCERerank(BaseDocumentCompressor):
    """Document compressor that uses `BCEmbedding RerankerModel API`.
    用于文档压缩或重排的组件
    """

    client: str = 'BCEmbedding' # 客户端名称
    top_n: int = 3 # 返回文档的数量
    """Number of documents to return."""
    model: str = "maidalun1020/bce-reranker-base_v1" # 重排的模型
    """Model to use for reranking."""
    _model: Any = PrivateAttr() # 使用PrivateAttr定义了一个私有属性_model，它将在实例化时存储模型的实例。

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid # 表明不允许传递未在模型中定义的字段，增加了数据安全性。
        arbitrary_types_allowed = True # 允许模型使用任意类型的字段，提高了灵活性。
    
    def __init__(
        self,
        top_n: int = 3,
        model: str = "maidalun1020/bce-reranker-base_v1",
        device: Optional[str] = None,
        **kwargs
    ):
        try:
            from BCEmbedding.models import RerankerModel
        except ImportError:
            raise ImportError(
                "Cannot import `BCEmbedding` package,",
                "please `pip install BCEmbedding>=0.1.2`",
            )
        # 创建RerankerModel的实例，并将其赋值给私有属性_model。
        self._model = RerankerModel(model_name_or_path=model, device=device, **kwargs)
        # 调用父类的构造函数，传递top_n和model参数。
        super().__init__(top_n=top_n, model=model)

    # 定义了一个名为validate_environment的根验证器，用于在实例化前验证环境设置。
    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """验证器确保了必要的配置项（如API密钥和Python包）存在于环境中。
           将client属性的值设置为"BCEmbedding.models.RerankerModel"，确保了环境的一致性。"""
        values["client"] = "BCEmbedding.models.RerankerModel"
        return values

    def compress_documents(
        self,
        documents: Sequence[Document], # 要处理的文档序列
        query: str, # 用于重排文档的查询字符串
        callbacks: Optional[Callbacks] = None, # 在文档压缩过程中运行的回调函数，可选参数。
    ) -> Sequence[Document]:
        """
        Compress documents using `BCEmbedding RerankerModel API`.
        用BCEmbedding RerankerModel API根据给定的查询对一系列文档进行压缩（或更准确地说，重排）
        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        # 如果输入的文档序列为空，则立即返回空列表，避免进行无用的API调用。
        if len(documents) == 0:  # to avoid empty api call
            return []
        # 将文档序列转换为列表
        doc_list = list(documents)
        # 有效文档的文本内容
        passages = []
        # 存储有效和无效的文档对象
        valid_doc_list = []
        invalid_doc_list = []
        # 遍历文档列表
        for d in doc_list:
            # 从每个文档对象中提取文本内容
            passage = d.page_content
            # 判断提取的文本内容是否有效（即是否为非空字符串）
            if isinstance(passage, str) and len(passage) > 0:
                # 如果有效，则将文本内容中的换行符替换为空格
                passages.append(passage.replace('\n', ' '))
                valid_doc_list.append(d)
            else:
                # 如果无效，则将当前文档对象添加到invalid_doc_list。
                invalid_doc_list.append(d)
        # 调用方法进行重排，rerank方法返回一个包含重排后的文段、对应得分和文段索引的字典。
        rerank_result = self._model.rerank(query, passages)
        # 存储最终的文档对象，这些文档将根据重排得分进行更新。
        final_results = []
        # 遍历重排得分(rerank_scores)和文段索引(rerank_ids)
        for score, doc_id in zip(rerank_result['rerank_scores'], rerank_result['rerank_ids']):
            # 获取对应的文档对象
            doc = valid_doc_list[doc_id]
            # 将文档的重排得分score添加到文档的元数据metadata中，键名为"relevance_score"。
            doc.metadata["relevance_score"] = score
            # 将更新后的文档对象添加到final_results列表中
            final_results.append(doc)
        # 处理无效文档
        for doc in invalid_doc_list:
            # 将无效文档的相关性得分设置为0
            doc.metadata["relevance_score"] = 0
            # 确保了即使文档因为内容无效而未参与重排，也能被包含在最终结果中，但其得分为最低。
            final_results.append(doc)
        # 保留得分最高的前n个文档
        final_results = final_results[:self.top_n]
        return final_results

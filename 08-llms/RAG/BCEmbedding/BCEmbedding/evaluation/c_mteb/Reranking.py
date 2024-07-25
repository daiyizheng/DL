import numpy as np
from mteb import RerankingEvaluator, AbsTaskReranking
from tqdm import tqdm

from BCEmbedding.utils import logger_wrapper
logger = logger_wrapper('evaluation.c_mteb.Reranking')


class ModChineseRerankingEvaluator(RerankingEvaluator):
    """
    This class evaluates a SentenceTransformer model for the task of re-ranking.
    Given a query and a list of documents, it computes the score [query, doc_i] for all possible
    documents and sorts them in decreasing order. Then, MRR@10 and MAP is compute to measure the quality of the ranking.
    :param samples: Must be a list and each element is of the form:
        - {'query': '', 'positive': [], 'negative': []}. Query is the search query, positive is a list of positive
        (relevant) documents, negative is a list of negative (irrelevant) documents.
        - {'query': [], 'positive': [], 'negative': []}. Where query is a list of strings, which embeddings we average
        to get the query embedding.
    重排序任务是指在给定查询和一组文档的情况下，计算每个文档相对于查询的得分，并按得分降序排序文档的过程。
    然后，通过计算MRR@10（平均倒数排名在前10）和MAP（平均精确率）来衡量排序的质量。

    输入是一个样本列表，每个样本是一个字典，包含一个查询（query）、一个正面（相关）文档列表（positive）和
    一个负面（不相关）文档列表（negative）。查询可以是一个字符串或一个字符串列表，如果是字符串列表，
    那么这些字符串的嵌入会被平均，以获得查询嵌入。
    """

    def compute_metrics_batched(self, model):
        """
        Computes the metrices in a batched way, by batching all queries and
        all documents together
        用于批量计算评估指标，通过将所有查询和所有文档一起批量处理。
        """
        # 检查传入的模型对象是否有一个名为compute_score的方法，这是一种多态性的体现，根据模型的具体实现选择不同的计算路径。
        if hasattr(model, 'compute_score'):
            # 直接计算查询和文档之间的相似度得分
            return self.compute_metrics_batched_from_crossencoder(model)
        else:
            # bi-encoder模型首先独立计算查询和文档的嵌入，然后通过计算嵌入之间的相似度来评估它们的关系。
            return self.compute_metrics_batched_from_biencoder(model)

    def compute_metrics_batched_from_crossencoder(self, model):
        '''从cross-encoder模型批量计算重排评估指标'''
        # 存储所有的平均倒数排名（MRR）和平均精度（AP）分数
        all_mrr_scores = []
        all_ap_scores = []
        # 存储将要评估的查询和文档对
        pairs = []
        # 遍历self.samples中的每个样本，每个样本都是一个字典，包含查询（query）、正向文档（positive）和负向文档（negative）
        for sample in tqdm(self.samples, desc="Evaluating"):
            if isinstance(sample['query'], str):
                # 对于每个正向（positive）和负向（negative）文档，将查询和文档对添加到pairs列表中。
                for p in sample['positive']:
                    pairs.append([sample['query'], p])
                for n in sample['negative']:
                    pairs.append([sample['query'], n])
            elif isinstance(sample['query'], list):
                # 对于每个正向和负向文档，将每个查询字符串与文档组合，生成多个查询和文档对，并添加到pairs列表中。
                for p in sample['positive']:
                    for q in sample['query']:
                        pairs.append([q, p])
                for n in sample['negative']:
                    for q in sample['query']:
                        pairs.append([q, n])
            else:
                # 如果都不是，抛出异常，表示不支持的查询类型。
                raise NotImplementedError
        # 计算pairs列表中所有查询和文档对的分数
        all_scores = model.compute_score(pairs)
        # 转换为NumPy数组
        all_scores = np.array(all_scores)

        # Fix: sample['query'] is a list.
        # start_inx为0，这个变量用来跟踪在批量计算的分数数组all_scores中的起始索引位置。
        start_inx = 0
        for sample in tqdm(self.samples, desc="Evaluating"):
            # 子查询的数量
            num_subqueries = len(sample["query"]) if isinstance(sample["query"], list) else 1
            # 正向文档（positive）标记为True，负向文档（negative）标记为False，以表示每个文档是否与查询相关。
            is_relevant = [True] * len(sample['positive']) + [False] * len(sample['negative'])
            # 从all_scores数组中切片出当前样本对应的预测分数
            pred_scores = all_scores[start_inx:start_inx + len(is_relevant)*num_subqueries]
            # 更新start_inx索引
            start_inx += len(is_relevant)*num_subqueries
            # 如果有多个子查询
            if num_subqueries > 1:
                # 将pred_scores数组重塑为每个文档对每个查询的分数矩阵
                pred_scores = pred_scores.reshape(len(is_relevant), num_subqueries)
                # 使用np.max沿着子查询维度（axis=-1）取最大值，以便每个文档只保留最高的分数。
                pred_scores = np.max(pred_scores, axis=-1)
            # 使用np.argsort对预测分数pred_scores进行降序排序，得到排序后的索引数组
            pred_scores_argsort = np.argsort(-pred_scores)  # Sort in decreasing order
            # 计算当前样本的MRR分数，传入参数包括文档的相关性标记is_relevant、
            # 排序后的索引pred_scores_argsort和MRR计算时考虑的最高位置self.mrr_at_k。
            mrr = self.mrr_at_k_score(is_relevant, pred_scores_argsort, self.mrr_at_k)
            # 计算当前样本的AP分数
            ap = self.ap_score(is_relevant, pred_scores)

            all_mrr_scores.append(mrr)
            all_ap_scores.append(ap)
        # 计算所有样本的MRR和AP分数的平均值
        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)
        # 返回一个字典，包含平均精度（MAP）和平均倒数排名（MRR）的值。
        return {"map": mean_ap, "mrr": mean_mrr}

    def compute_metrics_batched_from_biencoder(self, model):
        '''在使用双编码器（bi-encoder）模型进行批量计算重排评估指标时，首先对查询进行编码的过程。'''
        # 储计算得到的所有平均倒数排名（MRR）和平均精度（AP）分数
        all_mrr_scores = []
        all_ap_scores = []
        logger.info("Encoding queries...")
        # 检查self.samples列表中第一个样本的query字段是否为字符串类型
        if isinstance(self.samples[0]["query"], str):
            # 检查模型（model）是否具有encode_queries方法
            if hasattr(model, 'encode_queries'):
                # 对所有查询进行编码
                all_query_embs = model.encode_queries(
                    [sample["query"] for sample in self.samples], # 取所有样本的查询文本
                    convert_to_tensor=True, # 将编码后的查询转换为张量
                    batch_size=self.batch_size, # 批处理大小
                )
            else:
                # 如果没有encode_queries方法，则回退使用模型的encode方法进行查询编码。
                all_query_embs = model.encode(
                    [sample["query"] for sample in self.samples],
                    convert_to_tensor=True,
                    batch_size=self.batch_size,
                )
        elif isinstance(self.samples[0]["query"], list):
            # In case the query is a list of strings, we get the most similar embedding to any of the queries
            all_query_flattened = [q for sample in self.samples for q in sample["query"]]
            if hasattr(model, 'encode_queries'):
                all_query_embs = model.encode_queries(all_query_flattened, convert_to_tensor=True,
                                                      batch_size=self.batch_size)
            else:
                all_query_embs = model.encode(all_query_flattened, convert_to_tensor=True, batch_size=self.batch_size)
        else:
            raise ValueError(f"Query must be a string or a list of strings but is {type(self.samples[0]['query'])}")

        logger.info("Encoding candidates...")
        all_docs = []
        for sample in self.samples:
            all_docs.extend(sample["positive"])
            all_docs.extend(sample["negative"])
        # 使用模型的encode_corpus方法对所有文档（正样本和负样本）进行编码
        all_docs_embs = model.encode_corpus(all_docs, convert_to_tensor=True, batch_size=self.batch_size)

        # Compute scores
        logger.info("Evaluating...")
        # 跟踪当前处理到的查询和文档编码的位置
        query_idx, docs_idx = 0, 0
        for instance in self.samples:
            # 计算每个样本中查询的数量
            num_subqueries = len(instance["query"]) if isinstance(instance["query"], list) else 1
            # 从之前编码的所有查询嵌入all_query_embs中提取当前样本的查询嵌入
            query_emb = all_query_embs[query_idx: query_idx + num_subqueries]
            #  更新query_idx
            query_idx += num_subqueries
            # 计算正样本和负样本的数量
            num_pos = len(instance["positive"])
            num_neg = len(instance["negative"])
            # 从所有文档编码中提取当前样本对应的文档编码，包括正样本和负样本的编码。
            docs_emb = all_docs_embs[docs_idx: docs_idx + num_pos + num_neg]
            # 更新docs_idx
            docs_idx += num_pos + num_neg
            # 如果正样本或负样本数量为0，则跳过当前样本。这是因为没有足够的数据来计算相关性得分。
            if num_pos == 0 or num_neg == 0:
                continue
            # 标记哪些文档是相关的（正样本）哪些是不相关的（负样本）
            is_relevant = [True] * num_pos + [False] * num_neg
            # 计算当前样本的评估指标，如MRR和AP。
            scores = self._compute_metrics_instance(query_emb, docs_emb, is_relevant)
            # 将计算得到的MRR和AP分数添加到列表中
            all_mrr_scores.append(scores["mrr"])
            all_ap_scores.append(scores["ap"])
        # 计算所有样本的AP和MRR分数的平均值
        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)

        return {"map": mean_ap, "mrr": mean_mrr}


def evaluate(self, model, split="test", **kwargs):
    '''用于评估模型在指定数据集分割（如测试集）上的表现'''
    logger.info('##BCEmbedding##: using ModChineseRerankingEvaluator')
    if not self.data_loaded:
        # 判断检查数据是否已经被加载到内存中
        self.load_data()
    # 获取相应的数据分割，通常是"train"、"test"或"valid"中的一个。
    data_split = self.dataset[split]
    # 创建一个ModChineseRerankingEvaluator实例
    evaluator = ModChineseRerankingEvaluator(data_split, **kwargs)
    # 调用评估器的实例，传入要评估的模型model，执行评估过程。
    scores = evaluator(model)
    # 将评估得到的分数转换为字典后返回
    return dict(scores)


AbsTaskReranking.evaluate = evaluate

####################################################################################
# C_MTEB: https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB
class T2RerankingZh2En(AbsTaskReranking):
    @property
    def description(self):
        return {
            'name': 'T2RerankingZh2En',
            'hf_hub_name': "C-MTEB/T2Reranking_zh2en",
            'description': 'T2Ranking: A large-scale Chinese Benchmark for Passage Ranking',
            "reference": "https://arxiv.org/abs/2304.03679",
            'type': 'Reranking',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh-en'],
            'main_score': 'map',
        }


class T2RerankingEn2Zh(AbsTaskReranking):
    @property
    def description(self):
        return {
            'name': 'T2RerankingEn2Zh',
            'hf_hub_name': "C-MTEB/T2Reranking_en2zh",
            'description': 'T2Ranking: A large-scale Chinese Benchmark for Passage Ranking',
            "reference": "https://arxiv.org/abs/2304.03679",
            'type': 'Reranking',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['en-zh'],
            'main_score': 'map',
        }

####################################################################################
class MMarcoRerankingZh2En(AbsTaskReranking):
    @property
    def description(self):
        return {
            'name': 'MMarcoRerankingZh2En',
            'hf_hub_name': "maidalun1020/MMarcoRerankingZh2En",
            'description': '',
            "reference": "",
            'type': 'Reranking',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh-en'],
            'main_score': 'map',
        }

class MMarcoRerankingEn2Zh(AbsTaskReranking):
    @property
    def description(self):
        return {
            'name': 'MMarcoRerankingEn2Zh',
            'hf_hub_name': "maidalun1020/MMarcoRerankingEn2Zh",
            'description': '',
            "reference": "",
            'type': 'Reranking',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['en-zh'],
            'main_score': 'map',
        }
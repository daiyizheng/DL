'''
@Description: 
@Author: shenlei
@Date: 2023-12-31 01:04:18
@LastEditTime: 2024-02-06 10:33:56
@LastEditors: shenlei
'''
import os, json, sys
import os.path as osp
import argparse
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm

from BCEmbedding.utils import logger_wrapper
logger = logger_wrapper('tools.eval_rag.summarize_eval_results')
from IPython import embed

def read_results(results_dir):
    '''从指定目录中读取CSV文件，汇总和处理这些CSV文件中的结果数据，并计算出加权平均的命中率和平均倒数排名（MRR）。'''
    # 使用列表推导式生成一个包含目录中所有CSV文件路径的列表
    csv_files = [osp.join(results_dir, i) for i in os.listdir(results_dir) if i.endswith('.csv')]
    logger.info('find {} csv files'.format(len(csv_files)))
    # 用于累加所有查询的数量
    tot_num = 0
    # 循环遍历所有CSV文件，读取每个文件并累加第一行最后一列的值到tot_num，这代表了每个CSV文件中包含的查询数量。
    for csv_file in csv_files:
        csv_df = pd.read_csv(csv_file)
        tot_num += csv_df.values[0,-1]
    logger.info('total number of queries: {}'.format(tot_num))
    # 用于存储合并后的结果。OrderedDict是一个字典，但它会保持元素被添加的顺序。
    merged_results = OrderedDict()
    for csv_file in tqdm(csv_files):
        csv_df = pd.read_csv(csv_file)
        # 遍历DataFrame的每一行
        for _, it in csv_df.iterrows():
            queries_num = it['nums']
            embedding = it['Embedding']
            # 检查是否已经为当前的嵌入（Embedding）创建了结果字典，如果没有，则为其创建一个OrderedDict。
            if embedding not in merged_results:
                merged_results[embedding] = OrderedDict()
            # 检查当前嵌入下是否已经为当前的重排序器（Reranker）创建了结果字典
            reranker = it['Reranker']
            if reranker not in merged_results[embedding]:
                # 如果没有，则初始化一个包含hit_rate和mrr的字典，并将它们的初始值设为0。
                merged_results[embedding][reranker] = {'hit_rate': 0, 'mrr': 0}
            # 对于每一行数据，根据查询数量queries_num和总查询数量tot_num的比例，
            # 计算加权的hit_rate和mrr，然后累加到相应的嵌入和重排序器的结果中。
            merged_results[embedding][reranker]['hit_rate'] += it['hit_rate'] * (queries_num/tot_num)
            merged_results[embedding][reranker]['mrr'] += it['mrr'] * (queries_num/tot_num)
    # 返回处理并合并后的结果字典
    return merged_results


def output_markdown(merged_results, # 合并后的结果字典
                    save_file, # 保存Markdown文件的路径
                    eval_more_domains=True # 指示是否评估了更多领域
                    ):
    '''用于将合并后的结果输出为Markdown格式，并保存到指定的文件中。'''
    with open(save_file, 'w') as f:
        f.write(f"# RAG Evaluations in LlamaIndex  \n")

        if eval_more_domains:
            f.write(f'\n## Multiple Domains Scenarios  \n')
        else:
            f.write(f'\n## Reproduce [LlamaIndex Blog](https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83)  \n')

        first_line = "| Embedding Models |"
        second_line = "|:-------------------------------|"
        # 从merged_results字典中提取所有嵌入模型的名称，并按字典序进行排序。
        embeddings = sorted(list(merged_results.keys()))
        # 存储所有出现过的重排序器的名称
        rerankers = []
        # 遍历embeddings列表中的每个嵌入模型名称
        for embedding in embeddings:
            # 遍历该嵌入模型对应的所有重排序器名称
            for reranker in list(merged_results[embedding].keys()):
                # 如果重排序器的名称尚未出现在rerankers列表中，则将其添加到该列表中。
                if reranker not in rerankers:
                    rerankers.append(reranker)

        # 向Markdown文件中添加表格内容，包括重排序器的名称、对应的命中率（hit rate）和平均倒数排名（MRR）的值。
        for reranker in rerankers:
            # 在first_line字符串中追加每个重排序器的名称和一个换行<br>标签，后面跟着[*hit_rate/mrr*]文字，然后是一个分隔符|。
            first_line += f" {reranker} <br> [*hit_rate/mrr*] |"
            # 用于设置Markdown表格中该列的对齐方式为居中对齐
            second_line += ":--------:|"
        f.write(first_line + ' \n')
        f.write(second_line + ' \n')
        # 遍历之前按字典序排序的嵌入模型名称列表embeddings
        for embedding in embeddings:
            # 获取当前嵌入模型对应的结果字典
            embedding_results = merged_results[embedding]
            # 构建Markdown表格的一行，行的开头是当前嵌入模型的名称。
            write_line = f"| {embedding} |"
            for reranker in rerankers:
                # 如果当前嵌入模型有对应的重排序器结果，则将命中率和MRR的值格式化为两位小数并追加到write_line字符串中；
                # 如果没有对应结果，则在该位置显示" -- "。
                if reranker in embedding_results:
                    write_line += f" {embedding_results[reranker]['hit_rate']:.2f}/{embedding_results[reranker]['mrr']:.2f} |"
                else:
                    write_line += " -- |"
            # 将构建好的write_line字符串写入到文件中，并在末尾添加一个换行符\n。
            f.write(write_line + '  \n')

def get_args():
    parser = argparse.ArgumentParser()
    # 指定包含评估结果CSV文件的目录路径
    parser.add_argument('--results_dir', default=osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), "results/rag_results"), type=str, help="pdfs to eval")
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数，并获取args对象。
    args = get_args()
    # 通过检查args.results_dir目录下CSV文件的数量来决定评估是否涵盖了多个领域。如果CSV文件数量大于1，则认为评估了多个领域。
    eval_more_domains = len([i for i in os.listdir(args.results_dir) if i.endswith('.csv')]) > 1
    # 确定输出文件的名称
    save_name = 'rag_eval_multiple_domains_summary' if eval_more_domains else 'rag_eval_reproduced_summary'
    # 构建了完整的保存路径
    save_path = os.path.join(args.results_dir, save_name + ".md")
    logger.info('Summary {}. Save as {}'.format('on more domains' if eval_more_domains else 'reproduce from BLOG', save_path))
    # 从指定目录读取并处理CSV文件，得到合并后的评估结果
    merged_results = read_results(args.results_dir)
    # 生成Markdown格式的汇总报告，并保存到指定的文件路径。
    output_markdown(
        merged_results, 
        save_path,
        eval_more_domains
        )
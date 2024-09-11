# 差异表达基因分析
差异分析流程:
1. 初始数据收集
2. 标准化(normalization)：DESeq、TMM等
3. 根据模型检验求p value：泊松分布(poisson distribution)、负二项式分布(NB)等
4. 多重假设得FDR值：
    - 检验方法：Wald、LRT
    - 多重检验：BH
5. 差异基因筛选：pvalue、padj

1)为什么要标准化？
  消除文库大小不同，测序深度对差异分析结果的影响
2)怎样标准化？
  找到一个能反映文库大小的因子，利用这个因子对rawdata进行标准化

- [基于omicverse基因差异分析（python）](./omicverse.ipynb)
- [基于DESeq2基因差异分析（python）](./DESeq2.ipynb)




# 学习笔记

> 注意环境准备
> 1. 依赖 ： 
> 配置项：cp example.env .env 

## 01 Python

## 02 数据分析


## 03 numpy 


## 04 基础算法
### 欧几里德变换
- []()

### 统计推断
#### 采样
- [重要性采样](04-Base-Algorithm/Statistical-Inference/Samples/Importance-Sampling.ipynb)
- [逆变换采样](04-Base-Algorithm/Statistical-Inference/Samples/Inverse-Transform-Sampling.ipynb)
- [拒绝抽样](04-Base-Algorithm/Statistical-Inference/Samples/Rejection-Sampling.ipynb)
- [吉布斯采样](04-Base-Algorithm/Statistical-Inference/Samples/Gibbs-Sampling.ipynb)
- [蒙特卡洛](04-Base-Algorithm/Statistical-Inference/Samples/MCMC.ipynb)
- [朗格万-蒙特卡洛](004-Base-Algorithm/Statistical-Inference/Samples/Langevin-Monte-Carlo.ipynb)
- [Metropolis Hastings](04-Base-Algorithm/Statistical-Inference/Samples/Metropolis-Hastings.ipynb)

## 05 Pytorch 学习
- BatchNorm 


## 06 scikit-learn 机器学习
- LinerRegression
- Logistics Regression
- Tree
- RandomForest
- GBDT
- XGBoost

## 07 rdkit 分子表征
- []()
- []()

## 08 LLM 大语言模型
### 深度学习基础知识
- [Postion编码]()

### 注意力机制
- 非参数注意力
- 参数注意力
- 内积注意力
- 加性注意力
- 多头注意力机制
- self-Attention


### RAG
- [相关向量数据库](08-llms/RAG/DB)
- [评估指标](08-llms/RAG/metric)




### Agent 智能体
- [Agent系列教程](08-llms/Agent_)
- [functioncall系列教程](08-llms/Functioncall)
- [模型上下文协议MCP系列教程](08-llms/MCP_)


##### MCP 框架源码分析


## 09 强化学习



## 10 web server



## 11 langchain



## 12  slurm 集群


## 97 生信分析
### R 语言
- [R 语言基础](97-bioinformatics/R_project/R_base)
- [R 语言统计推断](97-bioinformatics/R_project/R_statistical_inference)
- [R 语言可视化](97-bioinformatics/R_project/R_plot)
- [R语言机器学习](97-bioinformatics/R_project/R_ml)

### 生信相关包
- [Omicverse](97-bioinformatics/bio_package/Omicverse)
- [pydeseq2](97-bioinformatics/bio_package/pydeseq2)
- [PyWGCNA](97-bioinformatics/bio_package/PyWGCNA)
- [rnanorm](97-bioinformatics/bio_package/rnanorm)
- [ScanPy](97-bioinformatics/bio_package/ScanPy)

### 生信流程
#### bulk 
##### bulk上游分析
- [Aspera——利用SRR号批量高效下载FASTQ或SRA数据](97-bioinformatics/gene_process_for_python/bulk_RNA-seq_process/_00_Aspera——利用SRR号批量高效下载FASTQ或SRA数据.ipynb)
- [Aspera——利用SRR号批量高效下载FASTQ或SRA数据](97-bioinformatics/gene_process_for_python/bulk_RNA-seq_process/11_Aspera——利用SRR号批量高效下载FASTQ或SRA数据.ipynb)
- [上游数据下载、格式转化和质控清洗](97-bioinformatics/gene_process_for_python/bulk_RNA-seq_process/00_上游数据下载、格式转化和质控清洗.ipynb)
- [上游数据的比对计数——Hisat2+ featureCounts 与 Salmon](97-bioinformatics/gene_process_for_python/bulk_RNA-seq_process/01_上游数据的比对计数——Hisat2+ featureCounts 与 Salmon.ipynb)
- [ensembl_id转换与gene symbol基因名去重复的两种方法](97-bioinformatics/gene_process_for_python/bulk_RNA-seq_process/01-01_ensembl_id转换与gene symbol基因名去重复的两种方法.ipynb)
- [从featureCounts与Salmon输出文件获取counts矩阵](97-bioinformatics/gene_process_for_python/bulk_RNA-seq_process/02_从featureCounts与Salmon输出文件获取counts矩阵.ipynb)
- [Counts FPKM RPKM TPM CPM 的转](97-bioinformatics/gene_process_for_python/bulk_RNA-seq_process/02-02_Counts FPKM RPKM TPM CPM 的转化.ipynb)

#### bulk下游分析
- [差异分析前的准备——数据检查](97-bioinformatics/gene_process_for_python/bulk_RNA-seq_process/03_差异分析前的准备——数据检查.ipynb)
- [差异分析——DESeq2 edgeR limma的使用与比较](97-bioinformatics/gene_process_for_python/bulk_RNA-seq_process/04_差异分析——DESeq2 edgeR limma的使用与比较.ipynb)
- [GO、KEGG富集分析与enrichplot超全可视化](97-bioinformatics/gene_process_for_python/bulk_RNA-seq_process/05_GO、KEGG富集分析与enrichplot超全可视化攻略.ipynb)
- [GSEA——基因集富集分析](97-bioinformatics/gene_process_for_python/bulk_RNA-seq_process/06_GSEA——基因集富集分析.ipynb)
- [PPI蛋白互作网络构建（上）](97-bioinformatics/gene_process_for_python/bulk_RNA-seq_process/08_PPI蛋白互作网络构建（上）——STRING数据库的使用.ipynb)
- [PPI蛋白互作网络构建（下）——Cytoscape软件的使用](97-bioinformatics/gene_process_for_python/bulk_RNA-seq_process/09_PPI蛋白互作网络构建（下）——Cytoscape软件的使用.ipynb)
- [WGCNA加权基因共表达网络分析——关联基因模块与表型](97-bioinformatics/gene_process_for_python/bulk_RNA-seq_process/10_WGCNA加权基因共表达网络分析——关联基因模块与表型.ipynb)


#### 单细胞组学
##### 单细胞上游分析

##### 单细胞下游分析
- [单细胞完整工作流说明](97-bioinformatics/gene_process_for_python/single-cell-rna/01-单细胞数据说明.ipynb)
- [SCTransfom去批次效应](97-bioinformatics/gene_process_for_python/single-cell-rna/02-SCTransfom去批次效应.ipynb)
- [数据整合锚定法](97-bioinformatics/gene_process_for_python/single-cell-rna/03-锚定法对项目数据整合.ipynb)
- [数据整合harmony](97-bioinformatics/gene_process_for_python/single-cell-rna/04-harmony对项目数据着整合.ipynb)
- [mark基因展示](97-bioinformatics/gene_process_for_python/single-cell-rna/05-mark基于展示.ipynb)

#### 空间组学


## 98 开源框架学习





# RNA-seq 分析流程及所需软件如下
## 上游流程
- 创建单独的分析环境：miniconda3
- 数据下载和格式转化：sra-tools (sratoolkit)
- 质控清洗: fastqc multiqc trim-galore         | trimmomatic cutadapt fastp
- 比对: hisat2 subread samtools=1.6 salmon     | star bwa bowtie2 tophat
- 计数: featureCounts (已经整合在subread中)     |htseq bedtools deeptools

## 下游分析
- DEG差异分析：DESeq2 edgeR limma
- 差异基因注释：KEGG基因通路分析与 GO基因功能分析
- GSEA 基因集富集分析
- GSVA 基因集变异分析
- PPI 蛋白互作网络
- WGCNA 加权基因共表达网络分析

## 软件说明
- SRA Toolkit 是由美国国家生物技术信息中心（NCBI）提供的一组工具，专门用于处理 Sequence Read Archive（SRA）中存储的高通量测序数据。这个工具包包含了一系列命令行工具，用于检索、转换、处理和分析来自 SRA 的数据。
    - prefetch 是SRA Toolkit中的工具包中提供的一个专门用来下载SRR数据的工具, spera工具可以加速下载SRR fastq文件
    - fasterq-dump 从 SRA 下载数据并将其转换为 FASTQ 格式的工具，比 fastq-dump 速度更快
- FastQC的结果只针对一个测序文件，如果想将多个测序文件的结果整合一块来看，可以尝试用MultiQC软件
- MultiQC是一种基于Python的工具，用于整合和查看多种类型高通量测序数据的质量控制结果。
- Trim Galore是对FastQC和Cutadapt的包装。适用于所有高通量测序，包括RRBS(Reduced Representation Bisulfite-Seq ), Illumina、Nextera 和smallRNA测序平台的双端和单端数据。主要功能包括两步：
    - 首先去除低质量碱基，然后去除3' 末端的adapter, 如果没有指定具体的adapter，程序会自动检测前1million的序列，然后对比前12-13bp的序列是否符合以下  类型的adapter:
        - ● Illumina:   AGATCGGAAGAGC
        - ● Small RNA:  TGGAATTCTCGG
        - ● Nextera:    CTGTCTCTTATA
- Cutadapt 的核心功能是识别并移除读取序列（reads）两端的特定序列，例如 PCR 接头、adapters 或其他不需要的部分。它的操作基于简单的命令行接口，可以轻松地定制化参数以适应各种场景。该工具支持多种匹配模式，包括精确匹配、模糊匹配以及允许一定错误率的匹配。

> Cutadapt:仅仅进行adapter修剪,使用此工具。 TrimGalore:工具 TrimGalore 是围绕 cutadapt 和 FastQC 构建的包装器,具有一些附加功能
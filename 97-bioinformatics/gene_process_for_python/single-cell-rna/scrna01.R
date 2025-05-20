 ##系统报错改为英文
Sys.setenv(LANGUAGE = "en")
##禁止转化为因子
options(stringsAsFactors = FALSE)
##清空环境
rm(list=ls())

## 安装依赖包
### github 安装
install.packages('remotes')
remotes::install_version("Seurat", version = "3.2.0")
### 包管理安装
install.packages('Seurat')
getwd()
 
setwd("/Users/a1-6/Documents/projects/DL/97-bioinformatics/R_project/genomics_projects/02-single-cell-rna/")
###加载所需要的包
library(Seurat)
library(tidyverse)
library(dplyr)
library(patchwork)
####清除环境，就是把环境清空了
rm(list=ls())

##读取10x的数据
scRNA.counts=Read10X("/Users/a1-6/Documents/projects/DL/97-bioinformatics/R_project/genomics_projects/02-single-cell-rna/data/GSE152048_BC21.matrix/BC21")
class(scRNA.counts)

###创建Seurat对象
?CreateSeuratObject
# counts：输入对应的矩阵信息
# project：Seurat对象的项目名
# min.cells：规定基因的表达范围，即单个基因至少在多少个细胞中表达方可保留
# min.features：规定细胞表达的基因数范围，即单个细胞中至少表达多少个基因方可保留
scRNA = CreateSeuratObject(scRNA.counts , 
                           min.cells = 3, 
                           project="os", 
                           min.features = 300)
view(scRNA)
View(scRNA)

###Seurat对象
#### RNA 层
#### counts： 保存的是未经处理的原始数据，适合存放稀疏矩阵
#### data： 原始数据经过标准化后，会存放在@data中，和counts 一样也是一个特殊的 Matrix 对象
#### scale.data： 当数据进行scale后，存放在名为scale.data中,标准化了，很明显有正负
#### var.features： 是一个普通的向量，里面存放的是高表达变异的基因名。可以用函数VaribleFeatures来获得这个向量。
### key： 每个active对象都有一个key值，可以用fetch函数来获取
### meta.features： 对每个 features 做的注释。 如果要对 features 的功能进行注释、打分、筛选都需要用到meta.features。对于不同的assay来说，每个features的含义是不同的。
## meta.data 用来对所有细胞做注释的数据框。
## active.ident 细胞的类型，在Seurat对象中，细胞可能有好几种不同方法注释的类型，但是在某一时刻，只有一种细胞类型是默认激活的。
## reduction 是 assay 对象进行某种降维分析后得到的结果。降维也就是PCA 、tsne 、umap 三种
## Version：创建这个对象时，所使用的Seurat版本
## Commands：一个列表，里面保存的是workflow中每个步骤所使用的命令和参数,还有命令执行的日期和时间
### s3 data.frame  list  character matirx 
###s4 层级结构

##第一种s4对象的提取方法  点击白框
phe=scRNA@meta.data
count=scRNA@assays[["RNA"]]@counts
z=scRNA@assays[["RNA"]]@counts@Dimnames[[2]]

#####第二提取s4对象的方法 @ $交替使用
count=scRNA@assays$RNA@counts
scRNA@meta.data


#查看样本的细胞数量
table(scRNA@meta.data$orig.ident)        
##计算质控指标
#计算细胞中线粒体基因比例
?PercentageFeatureSet
scRNA[[]]# 取的是数据框
scRNA[["percent.mt"]] <- PercentageFeatureSet(scRNA, pattern = "^MT-") # 计算每个细胞计算线粒体基因相关的reads的百分比
head(scRNA@meta.data)
#计算红细胞比例
HB.genes <- c("HBA1","HBA2","HBB","HBD","HBE1","HBG1","HBG2","HBM","HBQ1","HBZ")
HB_m <- match(HB.genes, rownames(scRNA@assays$RNA)) 
HB.genes <- rownames(scRNA@assays$RNA)[HB_m] 
HB.genes <- HB.genes[!is.na(HB.genes)] 
scRNA[["percent.HB"]]<-PercentageFeatureSet(scRNA, features=HB.genes)  # 使用特征基因去计算红细胞相关基因在细胞中的reads的百分比
head(scRNA@meta.data)
col.num <- length(levels(scRNA@active.ident))
####Feature、count、线粒体基因、红细胞基因占比可视化。
violin <- VlnPlot(scRNA,
                  features = c("nFeature_RNA", "nCount_RNA", "percent.mt","percent.HB"), 
                  cols =rainbow(col.num), 
                  pt.size = 0.01, #不需要显示点，可以设置pt.size = 0
                  ncol = 4) + 
  theme(axis.title.x=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank()) 
###把图片画到画板上面
violin
#####以后保存图片都手动保存 不要用代码保存了。
#####你做到保存图片的时候喊师傅  我们语音聊天 我教你怎么保存
ggsave("vlnplot_before_qc.pdf", plot = violin, width = 12, height = 6) 
ggsave("vlnplot_before_qc.png", plot = violin, width = 12, height = 6)  
###这几个指标之间的相关性。 把图画到画板上，然后手动保存
plot1=FeatureScatter(scRNA, feature1 = "nCount_RNA", feature2 = "percent.mt")
plot2=FeatureScatter(scRNA, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")
plot3=FeatureScatter(scRNA, feature1 = "nCount_RNA", feature2 = "percent.HB")
pearplot <- CombinePlots(plots = list(plot1, plot2, plot3), nrow=1, legend="none") 
plot1
####看画板
plot2
####看画板
plot3
####看画板
pearplot
####看画板
#自己选择性保存图片，但是 pearplot的图片必须要看，因为这个是做质控用的
##我们可以看到，nFeature_RNA的范围在0到8000之内，percent.mt代表线粒体含量
###我们默认线粒体含量至少要小于`20%`，这是根据生物学知识得出的默认阈值。红细胞的数目要至少小于`5%``
###至于nFeature_RNA和nCount_RNA的阈值怎么确定，这个要结合 pearplot的图来判断。我们质控的目标就是删除离异值。而且注意阈值尽可能取的宽松一下，防止后面分析想要的细胞得不到。
###接下来从pearplot的图片来做质控---剔除离异值
##nFeature_RNA选择大于200 小于7500的 nFeature_RNA选择小于100000，percent.mt小于20，percent.HB小于5
scRNA1 <- subset(scRNA, 
                 subset = nFeature_RNA > 300& nFeature_RNA < 7000 & percent.mt < 10 & percent.HB < 3 & nCount_RNA < 100000)
scRNA
scRNA1
###在控制台中我们可以看到有500多细胞过滤了
####过滤完之后 我们就要对数据进行均一化，使用NormalizeData这个函数。
###注意均一化是用NormalizeData，标准化是用ScaleData

# 过滤后，小提琴图可视化QC指标
VlnPlot(scRNA1, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)
# FeatureScatter可视化QC指标之间的关系
plot1 <- FeatureScatter(scRNA1, feature1 = "nCount_RNA", feature2 = "percent.mt")
plot2 <- FeatureScatter(scRNA1, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")
plot1 + plot2

?NormalizeData
scRNA1 <- NormalizeData(scRNA1, normalization.method = "LogNormalize", scale.factor = 10000)
###好了，这一节数据加载、质控的内容就算是做完了。
###在我们关闭rstudio之前 先把环境中运行好的数据保存一下
###数据将保存在之前设定好的路径中。还有保存的scRNA1，不是scRNA，因为scRNA1才是过滤好的数据。\
## 对你结果
head(scRNA1@assays$RNA@layers$counts[1:6, 1:25])
head(scRNA1@assays$RNA@layers$data[1:6, 1:25])


save(scRNA1,file='scRNA1.Rdata')


###官方推荐是2000个高变基因，很多文章也有设置30000的，这个因自己的实验项目决定
scRNA1 <- FindVariableFeatures(scRNA1, selection.method = "vst", nfeatures = 3000) 
# Identify the 10 most highly variable genes，把top10的高变基因挑选出来，目的是为了作图
top10 <- head(VariableFeatures(scRNA1), 10) 
# plot variable features with and without labels  画出来不带标签的高变基因图
plot1 <- VariableFeaturePlot(scRNA1) 
###把top10的基因加到图中
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE, size=2.5) 
plot <- CombinePlots(plots = list(plot1, plot2),legend="bottom") 
###画图
plot 
####去画板看看 想保存就自己手动保存 



##如果内存足够最好对所有基因进行中心化 
## scale的作用是使数据具有一个统一的尺度，这并不会影响每个点的相对位置，只是使他们的表达量尺度统一起来。
scale.genes <-  rownames(scRNA1)
scRNA1 <- ScaleData(scRNA1, features = scale.genes)
##如果内存不够，可以只对高变基因进行标准化
#scale.genes <-  VariableFeatures(scRNA)
#scRNA <- ScaleData(scRNA, features = scale.genes)


#细胞周期回归：上一步找到的高变基因，常常会包含一些细胞周期相关基因。
#它们会导致细胞聚类发生一定的偏移，即相同类型的细胞在聚类时会因为细胞周期的不同而分开。
?CaseMatch
cc.genes
CaseMatch(c(cc.genes$s.genes,cc.genes$g2m.genes),VariableFeatures(scRNA1))
#细胞周期评分
g2m_genes = cc.genes$g2m.genes
g2m_genes = CaseMatch(search = g2m_genes, match = rownames(scRNA1))
s_genes = cc.genes$s.genes
s_genes = CaseMatch(search = s_genes, match = rownames(scRNA1))
scRNA1 <- CellCycleScoring(object=scRNA1,  g2m.features=g2m_genes,  s.features=s_genes)
#查看细胞周期基因对细胞聚类的影响
scRNAa <- RunPCA(scRNA1, features = c(s_genes, g2m_genes))
p <- DimPlot(scRNAa, reduction = "pca", group.by = "Phase")
p

VlnPlot(scRNAa, features = c("nFeature_RNA", "nCount_RNA", "percent.mt","percent.HB","G2M.Score","S.Score"), ncol = 6)
ggsave("cellcycle_pca.png", p, width = 8, height = 6)
 

## vars.to.regress
### 该参数允许你指定一个或多个变量（如技术协变量或生物协变量），通过线性回归的方式将这些变量的影响从数据中移除。
#scRNAb <- ScaleData(scRNA1, vars.to.regress = c("S.Score", "G2M.Score"), features = rownames(scRNA1))

#### 降低维度
scRNA1 <- RunPCA(scRNA1, features = VariableFeatures(scRNA1)) 
plot1 <- DimPlot(scRNA1, reduction = "pca", group.by="orig.ident") 
plot1

### 降纬后的可视化， Seurat提供了几种方法来对PCA结果的细胞和基因进行可视化，包括VizDimReduce()、 DimPlot()和DimHeatmap()。
ElbowPlot(scRNA1, ndims=20, reduction="pca") 

### 首先基于PCA空间中的欧几里得度量构建一个KNN图，
##  然后基于局部邻域中的共享重叠(Jaccard 相似性)来精确任意两个细胞之间的边缘权重。
pc.num=1:20
scRNA1 <- FindNeighbors(scRNA1, dims = pc.num) 

# 为了将细胞聚类，我们接下来应用模块化优化技术，如Louvain算法(默认值)或SLM，
## 迭代地将细胞聚类在一起。该步骤通过FindClusters()函数实现，其包含一个解析参数resolution，
### 可以理解为就是分辨率，分辨率数值越高，对细胞分群就越细，分到的群就可能会越多；反之则越少。
###官网这个3K的细胞集，基本上分辨率会选0.4-1.2.然后还跟我们说，对于较大的数据集，最佳分辨率往往会提高。
scRNA1 <- FindClusters(scRNA1, resolution = 1.0)

scRNA1<-BuildClusterTree(scRNA1)
PlotClusterTree(scRNA1)

### 执行非线性降维（UMAP/tSNE）
# Seurat 提供了一些非线性降维技术，如 tSNE 和 UMAP，来可视化和探索这些数据集。
scRNA1 = RunTSNE(scRNA1, dims = pc.num)
embed_tsne <- Embeddings(scRNA1, 'tsne')
write.csv(embed_tsne,'embed_tsne.csv')
plot1 = DimPlot(scRNA1, reduction = "tsne") 
##画图
plot1
###label = TRUE把注释展示在图中
DimPlot(scRNA1, reduction = "tsne",label = TRUE) 
###你会发现cluster都标了图中
ggsave("tSNE.pdf", plot = plot1, width = 8, height = 7)
##把图片保存一下

#UMAP---第二种可视化降维
scRNA1 <- RunUMAP(scRNA1, dims = pc.num)
embed_umap <- Embeddings(scRNA1, 'umap')
write.csv(embed_umap,'embed_umap.csv') 
plot2 = DimPlot(scRNA1, reduction = "umap") 
plot2






















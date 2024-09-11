setwd("/slurm/home/admin/nlp/DL/97-bioinformatics/R_project/gene_process/去除批次效应")
library(sva)
library(limma)
mm <- read.csv("GSE122340.csv", row.names = 1) # 读取GSE122340
mm2 <- read.csv("GSE69948.csv", row.names = 1) # 读取GSE69948
# las:标签是否平等于或垂直于坐标轴las=0:行；las=2:列
tcga_exp <- mm
boxplot(tcga_exp[, 1:10], las="2")## 每一列为坐标
tcga_group <- data.frame(row.names = colnames(tcga_exp), Sample=colnames(tcga_exp), DataSet="TCGA")

###
#columns Sample DataSet
#  -       -      -
###
gtex_exp <- mm2
boxplot(gtex_exp[, 1:10], las="2") #
gtex_group <- data.frame(row.names = colnames(gtex_exp), Sample = colnames(gtex_exp), DataSet = "GTEx")

dat_group <- rbind(tcga_group,  gtex_group) # 按行合并

com_ensg<-intersect(rownames(gtex_exp), rownames(tcga_exp))

dat_exp_before<-cbind(tcga_exp[com_ensg,], gtex_exp[com_ensg,])

boxplot(dat_exp_before[,170:185], las="2")  ##

library(factoextra)
library(FactoMineR)
before_exp_pca <- PCA(t(dat_exp_before[, rownames(dat_group)]), scale.unit=T, ncp=5, graph=F)
before_exp_pca.plot <- fviz_pca_ind(before_exp_pca, axes=c(1,2), label="none", addEllipses = T, ellipse.level=0.9, habillage = factor(dat_group$DataSet),
                                    palette = "aaas",
                                    mean.point=F,
                                    title="")
before_exp_pca.plot
before_exp_pca.plot <- before_exp_pca.plot+theme_bw()+
  theme(legend.direction = "horizontal", legend.position = "top")+
  xlim(-150,150)+ylim(-150,150)+
  xlab("Dim1")+ylab("Dim2")
before_exp_pca.plot


library(limma)
dat_exp<-removeBatchEffect(dat_exp_before[,rownames(dat_group)],
                           batch = dat_group$DataSet)
dim(dat_exp)
boxplot(dat_exp[,170:185],las="2")
dat_exp<-normalizeBetweenArrays(dat_exp)
boxplot(dat_exp[,170:185],las="2")
write.csv(dat_exp,"dat_exp.csv")


afterexp_pca<-PCA(t(dat_exp[,rownames(dat_group)]),
                    scale.unit=T,ncp=5,graph=F)
afterexp_pca.plot<-fviz_pca_ind(afterexp_pca,
                                  axes=c(1,2),
                                  label="none",
                                  addEllipses = T,
                                  ellipse.level=0.9,
                                  habillage = factor(dat_group$DataSet),
                                  palette = "aaas",
                                  mean.point=F,
                                  title="")
afterexp_pca.plot
afterexp_pca.plot<-afterexp_pca.plot+theme_bw()+
  theme(legend.direction = "horizontal",legend.position = "top")+
  xlim(-150,150)+ylim(-150,150)+
  xlab("Dim1")+ylab("Dim2")
afterexp_pca.plot

dat_PCA<-cowplot::plot_grid(before_exp_pca.plot,
                            afterexp_pca.plot,
                            ncol=2,nrow=1,
                            labels=toupper(letters)[1:2],
                            align = "hv")
dat_PCA
ggsave(plot=dat_PCA,
       "dat_PCA.pdf",
       width = 10,heigh=5,device=cairo_pdf)

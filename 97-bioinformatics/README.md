# 生物信息学
## 数据集格式
- 表达矩阵（GCT，TXT）
- 表型文件（CLS）
- 功能数据集文件（GMT）

## Bioconductor 下载
```shell
install.packages("BiocManager") 

###设置好清华镜像
rm(list = ls())   
options()$repos 
options()$BioC_mirror
#options(BioC_mirror="https://mirrors.ustc.edu.cn/bioc/") 
options(BioC_mirror="http://mirrors.tuna.tsinghua.edu.cn/bioconductor/")
options("repos" = c(CRAN="https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))
options()$repos 
options()$BioC_mirror

###安装需要的包
BiocManager::install(c("GSEABase","GSVA","msigdbr","clusterProfiler" ),ask = F,update = F)
BiocManager::install(c("GEOquery","limma","impute" ),ask = F,update = F)
BiocManager::install(c("org.Hs.eg.db","org.Mm.eg.db"),ask = F,update = F)
BiocManager::install(c("DESeq2","edgeR" ),ask = F,update = F)
BiocManager::install("enrichplot",ask = F,update = F)
BiocManager::install("devtools",ask = F,update = F)
BiocManager::install("WGCNA",ask = F,update = F) 
BiocManager::install("data.table",ask = F,update = F)
BiocManager::install("tximport",ask = F,update = F)
BiocManager::install("tidyverse",ask = F,update = F)
BiocManager::install("DOSE",ask = F,update = F)
BiocManager::install("patchwork",ask = F,update = F)
BiocManager::install("RBGL",ask = F,update = F)  #Vennerable依赖包
BiocManager::install("pathview",ask = F,update = F)
BiocManager::install(c("STRINGdb","ggraph","igraph"),ask = F,update = F)
install.packages("Vennerable", repos="http://R-Forge.R-project.org") #安装Vennerable包
install.packages("statmod")

#其他一些基础包安装
options()$repos 
install.packages(c("FactoMineR", "factoextra")) 
install.packages(c("ggplot2", "pheatmap","ggpubr","ggthemes","ggstatsplot","ggsci","ggsignif")) 
install.packages("rvcheck")
(.packages())  #查看当前加载运行包

#更新所有包
rvcheck::update_all(check_R = F,
                    which = c("CRAN", "BioC", "github"))
```






数据结构: Adata
数据处理: scanpy, rapids-singlecell
bulk RNA-seq: 

## Gene ID 转换






## R语言常用的包汇总【基础包+数据预处理+可视化】
1、数据导入导出包
readr：实现表格数据的快速导入，支持多种数据格式。
readxl：专门用于读取Excel电子表格数据。
openxlsx：用于读取Microsoft Excel电子表格数据。
googlesheets：用于读取Google电子表格数据。
rio:用于读取和写入各种数据格式的包，包括 CSV、Excel、JSON 等。
haven：用于读取SAS、SPSS和Stata统计软件格式的数据。
httr：从网站开放的API中读取数据，也是由“Hadley大神”的作品之一。
XML：用于读取XML格式的数据。
RJSONIO：用于读取和写入JSON格式的数据。
RODBC：用于连接ODBC数据库。
RMySQL：用于连接MySQL数据库。
RPostgreSQL：用于连接PostgreSQL数据库。
SQLite：用于连接SQLite数据库。
rJava：用于连接Java数据库。
export：用于将 R 图形导出为多种格式，如 PNG、JPEG、SVG 等。
flextable：用于将数据导出为 HTML 表格的包。
write.csv：将数据导出为CSV格式。
png、jpeg、bmp：用于将数据导出为各种图片格式。
2、数据预处理
dplyr：数据操作和处理，如数据筛选、聚合、排序等功能。
tidyr：数据整理和转换，如数据格式转换、列合并、行分组等功能。
stringr：字符串操作和处理，如字符串函数，如字符串匹配、替换、截取等。
reshape2：数据重塑，数据转换函数，如长格式转换为宽格式、宽格式转换为长格式等。
lubridate：日期和时间数据的处理，如日期格式化、日期计算等。
3、数据可视化
ggplot2：非常流行的数据可视化包，基于图形语法，提供了强大的绘图功能。
lattice：基于网格系统的数据可视化包，适用于绘制各种类型的图形。
ggvis：基于浏览器的交互式数据可视化包，提供了丰富的交互功能。
rCharts：生成各种类型的图表，包括折线图、柱状图、饼图等。
rggobi：专门用于探索和分析高维数据的可视化包。
rms：提供了基于模型的统计图形，适用于绘制预测和诊断图形。
visNetwork：用于绘制交互式的网络图。
plotly：提供了基于Web的交互式图形，用户可以自定义图形的外观和交互功能。
ggmap：专门用于在地图上绘制数据的可视化包。
gganimate：用于创建动画图形的包，可以定制动画的外观和速度。
Highcharter：绘制交互式Highcharts图。
DiagrammeR：绘制交互式图表。
MetricsGraphics：绘制交互式MetricsGraphics图。
rgl：用于创建三维立体图形的包。
rms：除了提供基于模型的统计图形外，还提供了生存分析图形的绘制功能。
rJavaPlotly：基于Java和Plotly库的可视化包，可以绘制各种类型的图形。
leaflet：专门用于在地图上绘制数据的可视化包，支持交互式地图。
dygraphs：用于创建动态时间序列图的包。



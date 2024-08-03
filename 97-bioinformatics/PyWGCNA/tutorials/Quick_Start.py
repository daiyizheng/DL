# -*- encoding: utf-8 -*-
'''
Filename         :Quick_Start.py
Description      :
Time             :2024/07/30 15:00:46
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

import sys
sys.path.insert(0, "/slurm/home/admin/nlp/DL/97-bioinformatics/PyWGCNA")
import PyWGCNA
geneExp = 'PyWGCNA/tutorials/5xFAD_paper/expressionList.csv'
pyWGCNA_5xFAD = PyWGCNA.WGCNA(name='5xFAD', 
                              species='mus musculus', 
                              geneExpPath=geneExp, 
                              outputPath='',
                              save=True)
pyWGCNA_5xFAD.geneExpr.to_df().head(5)
pyWGCNA_5xFAD.preprocess()
pyWGCNA_5xFAD.findModules()
pyWGCNA_5xFAD.top_n_hub_genes(moduleName="coral", n=10)
mm <- read.csv("/slurm/home/admin/nlp/DL/97-bioinformatics/R_project/plot_R_python/datasets/custom.csv",row.names = 1)


dim(mm)

boxplot(mm, las="2", xla="text1", yla="text2")

data("mtcars", package = "datasets")
data <- mtcars[, 1:3]
library(mlr3)

task_mtcars <- as_task_regr(data, target = "mpg", id = "cars" ) # id是随便起一个名字
library("mlr3viz") # 使用此包可视化数据
autoplot(task_mtcars, type = "pairs") # 基于GGally，我之前介绍过
boxplot(1:5)

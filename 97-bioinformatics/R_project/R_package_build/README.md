# R 包构建
下面是创建一个简单的R包的步骤：
设置R包：
首先，使用usethis包来创建一个新的R包骨架。
在此之前，先安装devtools包
```shell
install.packages(c("devtools", "usethis"))
usethis::create_package(path = "path")
```
这将在主目录下创建一个名为examplepkg的新文件夹，并初始化一些必要的文件和文件夹结构。并且会打开一个新的R界面，
```shell
ctrl+shift+N
##创建一个新的Rscript
```
在这个新的界面进行代码的操作

添加函数：
然后，我们添加一个简单的函数到我们的包中。
假设我们要创建一个函数add_numbers，它将两个数字相加。
首先，创建一个名为R的文件夹（如果尚未存在），正常应该存在这个R文件夹，

然后在该文件夹内创建一个新的R脚本文件add_numbers.R，并添加以下内容，

```R
add_numbers <- function(x, y) {
  return(x + y)
}
```

使用document()生成文档：
要使得你的函数可以在包中被用户发现和使用，你需要用devtools来生成文档。
```shell
devtools::document("E:/R/Rscripts/_R_packages/examplepkg")
```
配置包描述文件
DESCRIPTION文件位于包的根目录下，提供了包的元数据。打开这个文件，确保至少填写了以下字段：
```R
file.edit("DESCRIPTION")
```

会打开一个文件，让你编辑，可以修改一下名称、版本、标题、作者信息、描述、许可证、编码和数据加载策略等。

## 添加数据集
为了添加数据集到你的R包中，首先将数据文件（假设为data/example_data.rda）保存到包的data目录下。
你可以使用usethis::use_data()函数来帮助完成这个步骤：
假设example_data已经在你的R环境中存在

```R
usethis::use_data(example_data, overwrite = TRUE)
```
## 为数据集添加文档
为了确保你的数据集可以被用户理解和使用，你需要为它添加文档。这通常通过创建一个与数据集同名的.R文件来完成，
并在R/目录下使用roxygen2风格的注释来描述数据集。
在R/目录下创建一个新的R脚本文件（假设命名为data.R），并添加如下内容来描述example_data数据集：
要添加的内容如下：
```R
#' Example Dataset
#'
#' A brief description of the dataset. Here you can include details about
#' the dataset's source, its structure, and any other relevant information.
#'
#' @format A data frame with 5 rows and 2 columns:
#' \describe{
#'   \item{column1}{Numeric, simple sequence from 1 to 5.}
#'   \item{column2}{Factor, letters from a to e.}
#' }
#' @source Describe the source of your data here
"example_data"

```

## 重新生成文档和安装包
添加数据和数据的文档后，你需要重新生成包的文档并安装包，使得更改生效。
```R
devtools::document()
devtools::install()

```
## 编写单元测试
单元测试是确保你的函数按照预期工作的好方法。首先，使用usethis::use_testthat()初始化测试框架：
```shell
usethis::use_testthat()
```
然后，创建一个测试脚本在tests/testthat目录下。例如，为add_numbers函数创建一个测试文件test-add_numbers.R：
```shell
usethis::use_test("add_numbers")

```

把test-add_numbers.R中的内容给删除，然后把下面的内容添加到test-add_numbers.R中

```R
devtools::test()

```

## 构建和检查包
在提交包到CRAN之前，确保它能成功地构建并且通过R CMD check的检查。这可以使用devtools来完成：
```R
devtools::build()
devtools::check()

```
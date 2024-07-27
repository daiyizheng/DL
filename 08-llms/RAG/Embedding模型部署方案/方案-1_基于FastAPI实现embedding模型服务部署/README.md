# 案例运行指南

## 情形一：简单启动

**1. 启动embedding服务**

```shell
# 当前项目路径下，执行命令：
python embedding_app.py
```

**2. 测试API接口**

```shell
python embedding_test.py
```

或

```shell
# 并发测试
python embedding_concurrent_test.py
```

## 情形二：服务容器化部署【推荐】

**1. 创建Dockerfile文件**
具体代码，请查看项目下面的 `Dockerfile`

**2. 构建镜像**

```shell
# 构建一个名为embedding_app的Docker镜像
docker build -t embedding_app:v0.1 .

# 构建完之后，可以查看本机的镜像
docker images

# 删除镜像命令
docker rmi IMAGE_ID
```

**3. 运行容器**

```shell
# 基于embedding_app镜像启动一个名为embedding_service的容器，
# 并将容器的8000端口映射到宿主机的8000端口，允许通过http://localhost:8000访问服务。

docker run -d --name embedding_service -p 8000:8000 embedding_app:v0.1
```

```shell
# 查看运行的容器
docker ps

# 停止容器
docker stop 容器名称/ID

# 删除容器
docker rm 容器名称/ID
```

```shell
# 进入容器
docker exec  -it embedding_service bash

# 退出容器
exit
```
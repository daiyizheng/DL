import gc # 用于手动控制垃圾回收
import glob
import os
import sys

import torch
import tqdm # 用于显示进度条


def main() -> None:
    """Compare two llama checkpoint directories"""
    # 使用 glob.glob 加载两个目录中的所有检查点文件, sys.argv[1] 和 sys.argv[2] 是从命令行传入的两个文件夹路径。
    # 搜索模式 "consolidated.*.pth" 用于找到所有与之匹配的PyTorch模型文件。
    one_files = sorted(glob.glob(os.path.join(sys.argv[1], "consolidated.*.pth")))
    two_files = sorted(glob.glob(os.path.join(sys.argv[2], "consolidated.*.pth")))
    # 检查两个文件夹中文件数量是否相同
    assert len(one_files) == len(
        two_files
    ), "One directory has {} files while another has {} files.".format(
        len(one_files), len(two_files)
    )
    # 存储差异数据
    deltas = []
    # 遍历所有文件对，进行比较
    # tqdm.trange 用于显示进度条
    # torch.load 加载每个检查点文件
    for i in tqdm.trange(len(one_files), desc="Comparing shards"):
        one = torch.load(one_files[i])
        two = torch.load(two_files[i])
        assert len(one) == len(
            two
        ), "shard should have the same length: {} != {}".format(len(one), len(two))
        # 对每对文件中的项进行比较
        for _, (v, w) in enumerate(zip(one.items(), two.items())):
            assert v[0] == w[0], "{} != {}".format(v[0], w[0]) # 检查键（模型参数名）是否匹配
            assert v[1].shape == w[1].shape, "tensor {} shape {} != {}".format(
                v[0], v[1].shape, w[1].shape # 检查张量形状是否一致
            )

            delta = (v[1] - w[1]).abs().max().item() # 计算两个张量之间的最大绝对差异，并将其添加到 deltas 列表中。
            deltas.append((i, v[0], delta))
        # 删除加载的模型并调用 gc.collect() 来回收内存。
        del one
        del two
        gc.collect()
    # 将 deltas 列表按差异值降序排序
    deltas = sorted(deltas, key=lambda x: x[-1], reverse=True)
    print("Top 10 largest deltas:")
    # 打印前10个最大的差异项
    for i, k, v in deltas[:10]:
        print(f"  shard {i} {k}: {v}")


if __name__ == "__main__":
    main()

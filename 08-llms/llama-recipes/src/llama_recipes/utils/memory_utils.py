# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
# 导入Python的垃圾回收（garbage collection）模块。这个模块提供了接口来手动控制垃圾回收。
import gc
# psutil模块，一个跨平台的库，用于获取进程和系统利用率（如内存、CPU）的信息。
import psutil
# threading模块，它允许创建和管理线程。
import threading

import torch

def byte2gb(x):
    '''将字节单位的数值转换成GB。这是通过将输入的字节值除以2^30(1GB等于2^30字节)来实现'''
    return int(x / 2**30)
# This context manager is used to track the peak memory usage of the process
class MemoryTrace:
    def __enter__(self):
        '''这是上下文管理器的一部分。当进入with MemoryTrace()块时，会执行这个方法。'''
        # 在开始之前强制进行垃圾回收
        gc.collect()
        # 清空PyTorch CUDA缓存
        torch.cuda.empty_cache()
        # 重置CUDA的最大内存分配计数器
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        # 记录当前CUDA分配的内存
        self.begin = byte2gb(torch.cuda.memory_allocated())
        # 创建一个psutil.Process对象，代表当前进程。
        self.process = psutil.Process()
        # 记录当前进程的CPU内存使用情况
        self.cpu_begin = byte2gb(self.cpu_mem_used())
        self.peak_monitoring = True
        # 创建一个新线程，目标函数是self.peak_monitor_func。
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        # 将该线程设置为守护线程，这意味着当主线程结束时，守护线程也会自动结束。
        peak_monitor_thread.daemon = True
        # 启动监控线程
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process
        用于获取当前进程的内存使用情况"""
        # 返回当前进程的RSS（Resident Set Size）内存使用量
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        # 初始化CPU内存使用的峰值为-1，表示尚未记录任何峰值。
        self.cpu_peak = -1
        # 开始一个无限循环，用于不断监控内存使用情况。
        while True:
            # 更新cpu_peak值，保持其为迄今为止观察到的最大内存使用量。
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # 注释部分说明了为什么没有使用time.sleep：因为使用睡眠可能会错过内存使用的峰值。
            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            # 如果self.peak_monitoring变为False，则跳出循环。
            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        # 将self.peak_monitoring设置为False，这会导致监控线程停止。
        self.peak_monitoring = False
        # 在退出上下文管理器时，进行垃圾回收和清空CUDA缓存。
        gc.collect()
        torch.cuda.empty_cache()
        # 记录退出时CUDA分配的内存量
        self.end = byte2gb(torch.cuda.memory_allocated())
        # 记录CUDA内存分配的峰值
        self.peak = byte2gb(torch.cuda.max_memory_allocated())
        # 获取CUDA内存的详细统计信息
        cuda_info = torch.cuda.memory_stats()
        # 记录活跃内存的峰值
        self.peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
        # 记录CUDA内存分配重试的次数
        self.cuda_malloc_retires = cuda_info.get("num_alloc_retries", 0)
        self.peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
        # 记录CUDA内存不足（Out of Memory，OOM）的次数
        self.m_cuda_ooms = cuda_info.get("num_ooms", 0)
        # 计算使用的内存总量
        self.used = byte2gb(self.end - self.begin)
        # 计算内存峰值的增量
        self.peaked = byte2gb(self.peak - self.begin)
        # 记录CUDA预留的最大内存量
        self.max_reserved = byte2gb(torch.cuda.max_memory_reserved())
        # 记录退出时的CPU内存使用量
        self.cpu_end = self.cpu_mem_used()
        # 计算CPU内存的使用量
        self.cpu_used = byte2gb(self.cpu_end - self.cpu_begin)
        # 计算CPU内存的峰值增量
        self.cpu_peaked = byte2gb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")
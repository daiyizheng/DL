'''
@Description: 
@Author: shenlei
@Date: 2023-11-28 17:29:01
@LastEditTime: 2023-12-28 18:09:24
@LastEditors: shenlei
'''
import logging

def logger_wrapper(name='BCEmbedding'):
    '''创建一个配置好的日志记录器'''
    # 定义日志的格式：%(asctime)s代表日志事件的时间，%(levelname)s代表日志级别的名称（如INFO、WARNING等），
    # %(name)s代表日志记录器的名称，%(message)s代表日志消息本身。
    logging.basicConfig(format='%(asctime)s - [%(levelname)s] -%(name)s->>>    %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    # 创建一个新的日志记录器
    logger = logging.getLogger(name)
    return logger

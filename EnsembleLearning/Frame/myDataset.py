#coding=utf-8
import csv
import os
import re
import time
import sys
sys.setrecursionlimit(10000) # 增加快排递归的次数




# @Parameter    讀取路徑
# @Parameter    限制下標
def getFiles(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files if file.endswith(suffix)]

# @Parameter    創建路徑


def mkdir(path):
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        os.makedirs(path)
        print(path+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')
        return False



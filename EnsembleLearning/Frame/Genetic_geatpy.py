 # -*- coding: utf-8 -*
import geatpy as ea  # import geatpy
import pandas as pd
import numpy as np
from Frame import EnsembleProblem
from sklearn.preprocessing import StandardScaler
np.set_printoptions(suppress=True)




class geneticOptimizer():
    def __init__(self,NIND = 50,MAXGEN = 1000):
        self.NIND   = NIND
        self.MAXGEN = MAXGEN

    def run(self,X_train, X_test, y_train, y_test, save_path, PoolType='Process',ensemble_model = "catboost"):
        self.ensemble_model = ensemble_model
        print("save_path", save_path)
        """===============================实例化问题对象==========================="""
        # 设置采用多线程，若修改为: PoolType = 'Process'，则表示用多进程
        # 生成问题对象
        if (self.ensemble_model == "catboost"):
            problem = EnsembleProblem.CBProblem(PoolType, save_path, X_train, X_test, y_train, y_test)
        elif (self.ensemble_model == "xgboost"):
            problem = EnsembleProblem.XGBProblem(PoolType, save_path, X_train, X_test, y_train, y_test)
        elif(self.ensemble_model == "lightgbm"):
            problem = EnsembleProblem.LGBProblem(PoolType, save_path, X_train, X_test, y_train, y_test)
        elif (self.ensemble_model == "Adaboost"):
            problem = EnsembleProblem.AdaboostProblem(PoolType, save_path, X_train, X_test, y_train, y_test)
        elif (self.ensemble_model == "SVR"):
            problem = EnsembleProblem.SVRProblem(PoolType, save_path, X_train, X_test, y_train, y_test)
        elif (self.ensemble_model == "gbdt"):
            problem = EnsembleProblem.gbdtProblem(PoolType, save_path, X_train, X_test, y_train, y_test)
        elif (self.ensemble_model == "bagging"):
            problem = EnsembleProblem.baggingProblem(PoolType, save_path, X_train, X_test, y_train, y_test)
        elif (self.ensemble_model == "RandomForest"):
            problem = EnsembleProblem.RandomForestProblem(PoolType, save_path, X_train, X_test, y_train, y_test)
        elif (self.ensemble_model == "ExtraTrees"):
            problem = EnsembleProblem.ExtraTreesProblem(PoolType, save_path, X_train, X_test, y_train, y_test)
        else:
            print("model name have error")
            return 0
        # problem = MyProblem(PoolType, save_path, X_train, X_test, y_train, y_test)  # 生成问题对象
        """=================================种群设置=============================="""
        Encoding = 'RI'  # 编码方式
        NIND = self.NIND  # 种群规模
        Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
        population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
        """===============================算法参数设置============================="""
        myAlgorithm = ea.soea_DE_rand_1_bin_templet(problem, population)  # 实例化一个算法模板对象
        myAlgorithm.MAXGEN = self.MAXGEN  # 最大进化代数
        myAlgorithm.trappedValue = 1e-6  # “进化停滞”判断阈值
        myAlgorithm.maxTrappedCount = 10  # 进化停滞计数器最大上限值，如果连续maxTrappedCount代被判定进化陷入停滞，则终止进化
        myAlgorithm.logTras = 1  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
        myAlgorithm.verbose = True  # 设置是否打印输出日志信息
        myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
        """==========================调用算法模板进行种群进化========================"""
        [BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
        BestIndi.save()  # 把最优个体的信息保存到文件中
        """=================================输出结果=============================="""
        print('评价次数：%s' % myAlgorithm.evalsNum)
        print('时间已过 %s 秒' % myAlgorithm.passTime)
        if BestIndi.sizes != 0:
            print('最优的目标函数值为：%s' % (BestIndi.ObjV[0][0]))
            print('最优的控制变量值为：')
            for i in range(BestIndi.Phen.shape[1]):
                print(BestIndi.Phen[0, i])
            """=================================检验结果==============================="""
            # problem.test(learning_rate=BestIndi.Phen[0, 0], depth=BestIndi.Phen[0, 1],
            #              loss_function_index=BestIndi.Phen[0, 2])
        else:
            print('没找到可行解。')






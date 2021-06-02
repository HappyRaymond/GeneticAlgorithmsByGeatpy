from sklearn import datasets  # 导入库
from sklearn.tree import DecisionTreeRegressor
from multiprocessing import  Manager
from sklearn import metrics
import numpy as np
import geatpy as ea
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,ExtraTreesRegressor,AdaBoostRegressor
import geatpy as ea  # import geatpy
import pandas as pd
import numpy as np
from Frame import myDataset
from sklearn.preprocessing import StandardScaler
import os
import joblib
np.set_printoptions(suppress=True)

from sklearn.metrics import mean_absolute_error,mean_squared_error,median_absolute_error,r2_score,mean_squared_log_error

import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR

def reg_calculate(y_true, y_predict,test_sample_size,feature_size):

    # try except 的原因是有时候有些结果不适合用某种评估指标
    try:
        mse = mean_squared_error(y_true, y_predict)
    except:
        mse = np.inf
    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_predict))
    except:
        rmse = np.inf
    try:
        mae = mean_absolute_error(y_true, y_predict)
    except:
        mae= np.inf
    try:
        r2 = r2_score(y_true, y_predict)
    except:
        r2 = np.inf
    try:
        mad = median_absolute_error(y_true, y_predict)
    except:
        mad = np.inf
    try:
        mape = np.mean(np.abs((y_true - y_predict) / y_true)) * 100
    except:
        mape = np.inf
    try:
        if (test_sample_size > feature_size):
            r2_adjusted = 1 - ((1 - r2) * (test_sample_size - 1)) / (test_sample_size - feature_size - 1)
    except:
        r2_adjusted = np.inf
    try:
        rmsle = np.sqrt(mean_squared_log_error(y_true, y_predict))
    except:
        rmsle = np.inf


    print("mse: {},rmse: {},mae: {},r2: {},mad: {},mape: {},r2_adjusted: {},rmsle: {}".format(mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle))
    return {"mse":mse, "rmse":rmse, "mae":mae, "r2":r2, "mad":mad, "mape":mape, "r2_adjusted":r2_adjusted, "rmsle":rmsle}


def save_results(resultTitle, resultList, y_test, test_prediction, save_path):
    # 预测值不能小于0  否则会报错
    test_prediction[test_prediction < 0] = 0

    # 计算行数，匹配 prediciton 的保存
    save_result = "/".join([save_path, 'result.csv'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    try:
        count = len(open(save_result, 'rU').readlines())
    except:
        count = 1

    # 判断是否存在未见 没有则写入文件 有则追加写入
    resultTitle.insert(0, "count")
    resultList.insert(0, str(count))

    if not os.path.exists(save_result):
        with open(save_result, 'w') as f:
            titleStr = ",".join(resultTitle)
            f.write(titleStr)
            f.write('\n')

    with open(save_result, 'a+') as f:
        contentStr = ",".join(resultList)
        f.write(contentStr)
        f.write('\n')

    # 保存 prediction
    pred_path = os.path.join(save_path, 'Prediction')
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    save_prediction = os.path.join(pred_path, str(count) + '.csv')
    df = pd.DataFrame()
    df["y_test"] = y_test
    df["test_prediction"] = test_prediction
    df.to_csv(save_prediction, index=False)

    # np.savetxt(save_prediction, np.append(np.array(y_test), test_prediction, axis=1), delimiter=',')
    print('Save the value of prediction successfully!!')

    return count




# ===================   catboost =============#
# iterations = Vars[i, 0]  0-1000
# learning_rate = Vars[i, 1]
# depth = Vars[i, 2]
# l2_leaf_reg = int(Vars[i, 3])
# loss_function_index = int(Vars[i, 4])
# one_hot_max_size = int(Vars[i, 5])     # 0-2

lossList = ["MAE","MAPE","Poisson","Quantile","RMSE","MultiRMSE"]
class CBProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, PoolType, save_path, X_train, X_test, y_train, y_test):  # PoolType是取值为'Process'或'Thread'的字符串
        name = 'catboost'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）

        # 變量 dictionary

        Dim = 6  # 初始化Dim（决策变量维数）
        varTypes = [1, 0, 1, 0, 1, 1]  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [200, 2e-6, 1, 1, 0, 0]  # 决策变量下界
        ub = [2000, 1, 15, 5, len(lossList) - 1, 2]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 获取目标训练集和 测试集
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
        print(len(self.X_train), len(self.y_train), len(self.X_test), len(self.y_test))
        # 設置保存路徑
        self.save_path = save_path
        # 设置用多线程还是多进程
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(num_cores)  # 设置池的大小
            self.lock = Manager().Lock()  # 创建锁的方式不一样 其他都一样

    def aimFunc(self, pop):  # 目标函数，采用多线程加速计算
        Vars = pop.Phen  # 得到决策变量矩阵
        args = list(
            zip(list(range(pop.sizes)),
                [Vars] * pop.sizes,
                [self.X_train] * pop.sizes,
                [self.y_train] * pop.sizes,
                [self.X_test] * pop.sizes,
                [self.y_test] * pop.sizes,
                [self.lock] * pop.sizes,
                [self.save_path] * pop.sizes)
        )
        if self.PoolType == 'Thread':
            pop.ObjV = np.array(list(self.pool.map(subAimFuncCatboost, args)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFuncCatboost, args)
            result.wait()
            pop.ObjV = np.array(result.get())

def subAimFuncCatboost(args):
    i = args[0]
    Vars = args[1]
    X_train = args[2]
    y_train = args[3]
    X_test = args[4]
    y_test = args[5]
    lock = args[6]
    save_path = args[7]
    # print("X_train {} y_train {} X_test {} y_test {}".format(np.array(X_train).shape,  np.array(y_train).shape, np.array(X_test).shape, np.array(y_test).shape))
    # print("subAimFunc {} times".format(i))

    iterations = int(Vars[i, 0])
    learning_rate = Vars[i, 1]
    depth = int(Vars[i, 2])
    l2_leaf_reg = Vars[i, 3]
    loss_function_index = int(Vars[i, 4])
    one_hot_max_size = int(Vars[i, 5])

    parameterDict = {"iterations": iterations, "learning_rate": learning_rate, "depth": depth,
                     "loss_function": lossList[loss_function_index],
                     "l2_leaf_reg": l2_leaf_reg, "one_hot_max_size": one_hot_max_size,
                     "task_type": "CPU", "logging_level": "Silent"}
    Regressor = cb.CatBoostRegressor(**parameterDict)
    Regressor.fit(X_train, y_train)
    y_pre_CB = Regressor.predict(X_test)
    regDict = reg_calculate(y_test, y_pre_CB,X_test.shape[0], X_test.shape[1])
    ObjV_i = regDict["r2"]

    print("ObjV_i", ObjV_i)

    # make resultTitle
    resultTitle = []
    resultList = []

    for Title, Parameter in parameterDict.items():
        resultTitle.append(Title)
        resultList.append(str(Parameter))

    for Title, Parameter in regDict.items():
        resultTitle.append(Title)
        resultList.append(str(Parameter))

    with lock:
        count = save_results(resultTitle, resultList, y_test, y_pre_CB, save_path)
        # 保存模型
        model_path = os.path.join(save_path, 'Model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        Regressor.save_model(os.path.join(model_path, str(count) + ".model"))

    return [ObjV_i]

# ===============  XGBoost ==========#
# eta  [0,1]
# max_depth [1,∞]
# subsample [default=1] 取值范围为：(0,1]


class XGBProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, PoolType, save_path, X_train, X_test, y_train, y_test):  # PoolType是取值为'Process'或'Thread'的字符串
        name = 'XGBoost'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 3  # 初始化Dim（决策变量维数）
        varTypes = [0, 1, 0]  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0, 1, 0.2]  # 决策变量下界
        ub = [1, 15, 1]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 获取目标训练集和 测试集
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
        print(len(self.X_train), len(self.y_train), len(self.X_test), len(self.y_test))
        # 設置保存路徑
        self.save_path = save_path
        # 设置用多线程还是多进程
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(num_cores)  # 设置池的大小
            self.lock = Manager().Lock()  # 创建锁的方式不一样 其他都一样

    def aimFunc(self, pop):  # 目标函数，采用多线程加速计算
        Vars = pop.Phen  # 得到决策变量矩阵
        args = list(
            zip(list(range(pop.sizes)),
                [Vars] * pop.sizes,
                [self.X_train] * pop.sizes,
                [self.y_train] * pop.sizes,
                [self.X_test] * pop.sizes,
                [self.y_test] * pop.sizes,
                [self.lock] * pop.sizes,
                [self.save_path] * pop.sizes)
        )
        if self.PoolType == 'Thread':
            pop.ObjV = np.array(list(self.pool.map(subAimFuncXGBoost, args)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFuncXGBoost, args)
            result.wait()
            pop.ObjV = np.array(result.get())


def subAimFuncXGBoost(args):
    i = args[0]
    Vars = args[1]
    X_train = args[2]
    y_train = args[3]
    X_test = args[4]
    y_test = args[5]
    lock = args[6]
    save_path = args[7]
    # print("X_train {} y_train {} X_test {} y_test {}".format(np.array(X_train).shape,  np.array(y_train).shape, np.array(X_test).shape, np.array(y_test).shape))
    # print("subAimFunc {} times".format(i))

    # eta  [0,1]
    # max_depth [1,∞]
    # subsample [default=1] 取值范围为：(0,1]

    eta = Vars[i, 0]
    max_depth = int(Vars[i, 1])
    subsample = Vars[i, 2]

    parameterDict = {"eta": eta, "max_depth": max_depth, "subsample": subsample}

    Regressor = xgb.XGBRegressor(**parameterDict)
    Regressor.fit(X_train, y_train)
    y_pre_CB = Regressor.predict(X_test)
    regDict = reg_calculate(y_test, y_pre_CB,X_test.shape[0], X_test.shape[1])
    ObjV_i = regDict["r2"]

    print("ObjV_i", ObjV_i)

    # make resultTitle
    resultTitle = []
    resultList = []

    for Title, Parameter in parameterDict.items():
        resultTitle.append(Title)
        resultList.append(str(Parameter))

    for Title, Parameter in regDict.items():
        resultTitle.append(Title)
        resultList.append(str(Parameter))

    with lock:
        count = save_results(resultTitle, resultList, y_test, y_pre_CB, save_path)
        # 保存模型
        model_path = os.path.join(save_path, 'Model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))

    return [ObjV_i]

# ===============  Lightgbm =================
# boosting_type = ["gbdt","dart","goss","rf"]  # rf 還是會報錯 所以先沒放上去
# n_estimators = [50,100,200,300,400,500,600,700,800,900,1000,1200,1500]
# learning_rate = [0.01,0.05,0.1,0.15,0.2]
# subsample = [1.0,0.8,0.6]
# subsample_freq = [0,1,2,3]
# colsample_bytree = [1,0.8,0.6]
# reg_alpha = [0,1,2]
# reg_lambda = [0,1,2]



class LGBProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, PoolType, save_path, X_train, X_test, y_train, y_test):  # PoolType是取值为'Process'或'Thread'的字符串
        name = 'Lightgbm'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）

        # 變量 dictionary

        Dim = 8  # 初始化Dim（决策变量维数）
        varTypes = [1,1,0,0,1,0,1,1]  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0,50  ,1e-3,0.6,0,0.6,0,0]  # 决策变量下界
        ub = [2,1500,1e-1,1  ,3,1  ,2,2]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 获取目标训练集和 测试集
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
        print(len(self.X_train), len(self.y_train), len(self.X_test), len(self.y_test))
        # 設置保存路徑
        self.save_path = save_path
        # 设置用多线程还是多进程
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(num_cores)  # 设置池的大小
            self.lock = Manager().Lock()  # 创建锁的方式不一样 其他都一样

    def aimFunc(self, pop):  # 目标函数，采用多线程加速计算
        Vars = pop.Phen  # 得到决策变量矩阵
        args = list(
            zip(list(range(pop.sizes)),
                [Vars] * pop.sizes,
                [self.X_train] * pop.sizes,
                [self.y_train] * pop.sizes,
                [self.X_test] * pop.sizes,
                [self.y_test] * pop.sizes,
                [self.lock] * pop.sizes,
                [self.save_path] * pop.sizes)
        )
        if self.PoolType == 'Thread':
            pop.ObjV = np.array(list(self.pool.map(subAimFuncLgb, args)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFuncLgb, args)
            result.wait()
            pop.ObjV = np.array(result.get())

def subAimFuncLgb(args):
    i = args[0]
    Vars = args[1]
    X_train = args[2]
    y_train = args[3]
    X_test = args[4]
    y_test = args[5]
    lock = args[6]
    save_path = args[7]

    boosting_type_list = ["gbdt", "dart", "goss"]

    boosting_type = boosting_type_list[int(Vars[i, 0])]
    n_estimators = int(Vars[i, 1])
    learning_rate = Vars[i, 2]
    subsample = Vars[i, 3]
    subsample_freq = int(Vars[i, 4])
    colsample_bytree = Vars[i, 5]
    reg_alpha = Vars[i, 6]
    reg_lambda = Vars[i, 7]


    '''當 [LightGBM] [Fatal] Cannot use bagging in GOSS '''
    if (boosting_type == "goss" ):
        subsample_freq = 0 # 關閉bagging


    parameterDict = {"boosting_type": boosting_type, "n_estimators": n_estimators,
                     "learning_rate": learning_rate,"subsample": subsample,
                      "colsample_bytree": colsample_bytree,"max_depth":-1,
                     "reg_alpha": reg_alpha, "reg_lambda": reg_lambda,"objective":'regression',"random_state":19}

    Regressor = lgb.LGBMRegressor(**parameterDict)
    try:
        Regressor.fit(X_train, y_train)
    except:

        print("boosting_type", boosting_type)
        print("subsample_freq",subsample_freq)
    y_pre = Regressor.predict(X_test)
    regDict = reg_calculate(y_test, y_pre,X_test.shape[0], X_test.shape[1])
    ObjV_i = regDict["r2"]

    print("ObjV_i", ObjV_i)

    # make resultTitle
    resultTitle = []
    resultList = []

    for Title, Parameter in parameterDict.items():
        resultTitle.append(Title)
        resultList.append(str(Parameter))

    for Title, Parameter in regDict.items():
        resultTitle.append(Title)
        resultList.append(str(Parameter))

    with lock:
        count = save_results(resultTitle, resultList, y_test, y_pre, save_path)
        # 保存模型
        model_path = os.path.join(save_path, 'Model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))


    return [ObjV_i]


# ===============  Adaboost =================
# n_estimators = [50,100,200,300,400,500,600,700,800]
# learning_rate = [0.1,0.5,1,1.5,2]
# loss = ["linear","square","exponential"]
# criterion = ["mse","mae"]
# splitter = ["best","random"]
# # max_features = ["None"]
# # max_leaf_nodes = ["None"]
# # min_samples_split = [2]
# # min_samples_leaf = [1]

class AdaboostProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, PoolType, save_path, X_train, X_test, y_train, y_test):  # PoolType是取值为'Process'或'Thread'的字符串
        name = 'Adaboost'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        # 變量 dictionary

        Dim = 5  # 初始化Dim（决策变量维数）
        varTypes = [1, 0, 1, 1, 1]  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [50, 0.01, 0, 0, 0]  # 决策变量下界
        ub = [1500,  2, 2, 1, 1]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 获取目标训练集和 测试集
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
        print(len(self.X_train), len(self.y_train), len(self.X_test), len(self.y_test))
        # 設置保存路徑
        self.save_path = save_path
        # 设置用多线程还是多进程
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(num_cores)  # 设置池的大小
            self.lock = Manager().Lock()  # 创建锁的方式不一样 其他都一样

    def aimFunc(self, pop):  # 目标函数，采用多线程加速计算
        Vars = pop.Phen  # 得到决策变量矩阵
        args = list(
            zip(list(range(pop.sizes)),
                [Vars] * pop.sizes,
                [self.X_train] * pop.sizes,
                [self.y_train] * pop.sizes,
                [self.X_test] * pop.sizes,
                [self.y_test] * pop.sizes,
                [self.lock] * pop.sizes,
                [self.save_path] * pop.sizes)
        )
        if self.PoolType == 'Thread':
            pop.ObjV = np.array(list(self.pool.map(subAimFuncAdaBoost, args)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFuncAdaBoost, args)
            result.wait()
            pop.ObjV = np.array(result.get())

def subAimFuncAdaBoost(args):
    i = args[0]
    Vars = args[1]
    X_train = args[2]
    y_train = args[3]
    X_test = args[4]
    y_test = args[5]
    lock = args[6]
    save_path = args[7]


    lossList = ["linear", "square", "exponential"]
    criterionList = ["mse", "mae"]
    splitterList = ["best", "random"]

    n_estimators = int(Vars[i, 0])
    learning_rate = Vars[i, 1]
    loss = lossList[int(Vars[i, 2])]
    criterion = criterionList[int(Vars[i, 3])]
    splitter = splitterList[int(Vars[i, 4])]


    parameterDict = {"n_estimators": n_estimators, "learning_rate": learning_rate,"loss": loss,
                     "criterion": criterion,"splitter": splitter,"max_features":"None",
                     "max_leaf_nodes": "None", "min_samples_split": 2,"min_samples_leaf":1}

    Regressor = AdaBoostRegressor(n_estimators=n_estimators, learning_rate = learning_rate, loss = loss,
                                  base_estimator=DecisionTreeRegressor(min_samples_split=2,
                                                                       min_samples_leaf=1,
                                                                       splitter=splitter, criterion=criterion ))

    Regressor.fit(X_train, y_train)

    y_pre = Regressor.predict(X_test)
    regDict = reg_calculate(y_test, y_pre,X_test.shape[0], X_test.shape[1])
    ObjV_i = regDict["r2"]

    print("ObjV_i", ObjV_i)

    # make resultTitle
    resultTitle = []
    resultList = []

    for Title, Parameter in parameterDict.items():
        resultTitle.append(Title)
        resultList.append(str(Parameter))

    for Title, Parameter in regDict.items():
        resultTitle.append(Title)
        resultList.append(str(Parameter))

    with lock:
        count = save_results(resultTitle, resultList, y_test, y_pre, save_path)
        # 保存模型
        model_path = os.path.join(save_path, 'Model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))


    return [ObjV_i]



# ===============  SVR =================
# kernel = ["rbf","linear","poly","sigmoid"]
# degree = [2,3,4,5,6,7,8,9,10,11,12]
# gamma = ["auto","scale"]



class SVRProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, PoolType, save_path, X_train, X_test, y_train, y_test):  # PoolType是取值为'Process'或'Thread'的字符串
        name = 'SVR'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        # 變量 dictionary

        Dim = 3  # 初始化Dim（决策变量维数）
        varTypes = [1, 1, 1]  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0, 2, 0 ] # 决策变量下界
        ub = [3,  12, 1]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 获取目标训练集和 测试集
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
        print(len(self.X_train), len(self.y_train), len(self.X_test), len(self.y_test))
        # 設置保存路徑
        self.save_path = save_path
        # 设置用多线程还是多进程
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(num_cores)  # 设置池的大小
            self.lock = Manager().Lock()  # 创建锁的方式不一样 其他都一样

    def aimFunc(self, pop):  # 目标函数，采用多线程加速计算
        Vars = pop.Phen  # 得到决策变量矩阵
        args = list(
            zip(list(range(pop.sizes)),
                [Vars] * pop.sizes,
                [self.X_train] * pop.sizes,
                [self.y_train] * pop.sizes,
                [self.X_test] * pop.sizes,
                [self.y_test] * pop.sizes,
                [self.lock] * pop.sizes,
                [self.save_path] * pop.sizes)
        )
        if self.PoolType == 'Thread':
            pop.ObjV = np.array(list(self.pool.map(subAimFuncSVR, args)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFuncSVR, args)
            result.wait()
            pop.ObjV = np.array(result.get())

def subAimFuncSVR(args):
    i = args[0]
    Vars = args[1]
    X_train = args[2]
    y_train = args[3]
    X_test = args[4]
    y_test = args[5]
    lock = args[6]
    save_path = args[7]

    kernelList = ["rbf","linear","poly","sigmoid"]
    gammaList  = ["auto","scale"]


    kernel = kernelList[int(Vars[i, 0])]
    degree = int(Vars[i, 1])
    gamma = gammaList[int(Vars[i, 2])]

    if (kernel == "poly"):
        gamma = "scale"


    parameterDict = {"kernel": kernel, "degree": degree,"gamma": gamma}

    Regressor = SVR(**parameterDict)

    Regressor.fit(X_train, y_train)

    y_pre = Regressor.predict(X_test)
    regDict = reg_calculate(y_test, y_pre,X_test.shape[0], X_test.shape[1])
    ObjV_i = regDict["r2"]

    print("ObjV_i", ObjV_i)

    # make resultTitle
    resultTitle = []
    resultList = []

    for Title, Parameter in parameterDict.items():
        resultTitle.append(Title)
        resultList.append(str(Parameter))

    for Title, Parameter in regDict.items():
        resultTitle.append(Title)
        resultList.append(str(Parameter))

    with lock:
        count = save_results(resultTitle, resultList, y_test, y_pre, save_path)
        # 保存模型
        model_path = os.path.join(save_path, 'Model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))


    return [ObjV_i]



# ===============  gbdt =================
# random_state = 17
# n_estimators = [50,100,150,200,250,300]
# learning_rate = [0.05,0.1,0.15,0.2]
# loss = ["ls"]
# subsample = [1,0.8,0.6]
# min_samples_split = [2]
# max_depth = [3]
# min_samples_leaf = [1]



class gbdtProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, PoolType, save_path, X_train, X_test, y_train, y_test):  # PoolType是取值为'Process'或'Thread'的字符串
        name = 'gbdt'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        # 變量 dictionary

        Dim = 3  # 初始化Dim（决策变量维数）
        varTypes = [1, 0, 0]  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [50, 0.001, 0.6 ] # 决策变量下界
        ub = [800,  0.1, 1]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 获取目标训练集和 测试集
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
        print(len(self.X_train), len(self.y_train), len(self.X_test), len(self.y_test))
        # 設置保存路徑
        self.save_path = save_path
        # 设置用多线程还是多进程
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(num_cores)  # 设置池的大小
            self.lock = Manager().Lock()  # 创建锁的方式不一样 其他都一样

    def aimFunc(self, pop):  # 目标函数，采用多线程加速计算
        Vars = pop.Phen  # 得到决策变量矩阵
        args = list(
            zip(list(range(pop.sizes)),
                [Vars] * pop.sizes,
                [self.X_train] * pop.sizes,
                [self.y_train] * pop.sizes,
                [self.X_test] * pop.sizes,
                [self.y_test] * pop.sizes,
                [self.lock] * pop.sizes,
                [self.save_path] * pop.sizes)
        )
        if self.PoolType == 'Thread':
            pop.ObjV = np.array(list(self.pool.map(subAimFuncGBDT, args)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFuncGBDT, args)
            result.wait()
            pop.ObjV = np.array(result.get())

def subAimFuncGBDT(args):
    i = args[0]
    Vars = args[1]
    X_train = args[2]
    y_train = args[3]
    X_test = args[4]
    y_test = args[5]
    lock = args[6]
    save_path = args[7]

    # random_state = 17
    # n_estimators = [50,100,150,200,250,300]
    # learning_rate = [0.05,0.1,0.15,0.2]
    # loss = ["ls"]
    # subsample = [1,0.8,0.6]
    # min_samples_split = [2]
    # max_depth = [3]
    # min_samples_leaf = [1]


    n_estimators = int(Vars[i, 0])
    learning_rate = Vars[i, 1]
    subsample = Vars[i, 2]



    parameterDict = {"n_estimators": n_estimators, "learning_rate": learning_rate,"subsample": subsample}

    Regressor = GradientBoostingRegressor(**parameterDict)

    Regressor.fit(X_train, y_train)

    y_pre = Regressor.predict(X_test)
    regDict = reg_calculate(y_test, y_pre,X_test.shape[0], X_test.shape[1])
    ObjV_i = regDict["r2"]

    print("ObjV_i", ObjV_i)

    # make resultTitle
    resultTitle = []
    resultList = []

    for Title, Parameter in parameterDict.items():
        resultTitle.append(Title)
        resultList.append(str(Parameter))

    for Title, Parameter in regDict.items():
        resultTitle.append(Title)
        resultList.append(str(Parameter))

    with lock:
        count = save_results(resultTitle, resultList, y_test, y_pre, save_path)
        # 保存模型
        model_path = os.path.join(save_path, 'Model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))

    return [ObjV_i]


# ===============  ExtraTrees =================
# n_estimators = [10, 30, 50, 70, 90, 110, 130]
# min_samples_split = [2, 4, 6, 8]

class ExtraTreesProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, PoolType, save_path, X_train, X_test, y_train, y_test):  # PoolType是取值为'Process'或'Thread'的字符串
        name = 'ExtraTrees'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        # 變量 dictionary

        Dim = 2  # 初始化Dim（决策变量维数）
        varTypes = [1, 1]  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [30, 2] # 决策变量下界
        ub = [800,  10]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 获取目标训练集和 测试集
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
        print(len(self.X_train), len(self.y_train), len(self.X_test), len(self.y_test))
        # 設置保存路徑
        self.save_path = save_path
        # 设置用多线程还是多进程
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(num_cores)  # 设置池的大小
            self.lock = Manager().Lock()  # 创建锁的方式不一样 其他都一样

    def aimFunc(self, pop):  # 目标函数，采用多线程加速计算
        Vars = pop.Phen  # 得到决策变量矩阵
        args = list(
            zip(list(range(pop.sizes)),
                [Vars] * pop.sizes,
                [self.X_train] * pop.sizes,
                [self.y_train] * pop.sizes,
                [self.X_test] * pop.sizes,
                [self.y_test] * pop.sizes,
                [self.lock] * pop.sizes,
                [self.save_path] * pop.sizes)
        )
        if self.PoolType == 'Thread':
            pop.ObjV = np.array(list(self.pool.map(subAimFuncExtraTrees, args)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFuncExtraTrees, args)
            result.wait()
            pop.ObjV = np.array(result.get())

def subAimFuncExtraTrees(args):
    i = args[0]
    Vars = args[1]
    X_train = args[2]
    y_train = args[3]
    X_test = args[4]
    y_test = args[5]
    lock = args[6]
    save_path = args[7]

    n_estimators = int(Vars[i, 0])
    min_samples_split = int(Vars[i, 1])


    parameterDict = {"n_estimators": n_estimators, "min_samples_split": min_samples_split}

    Regressor = ExtraTreesRegressor(**parameterDict)
    Regressor.fit(X_train, y_train)

    y_pre = Regressor.predict(X_test)
    regDict = reg_calculate(y_test, y_pre,X_test.shape[0], X_test.shape[1])
    ObjV_i = regDict["r2"]

    print("ObjV_i", ObjV_i)

    # make resultTitle
    resultTitle = []
    resultList = []

    for Title, Parameter in parameterDict.items():
        resultTitle.append(Title)
        resultList.append(str(Parameter))

    for Title, Parameter in regDict.items():
        resultTitle.append(Title)
        resultList.append(str(Parameter))

    with lock:
        count = save_results(resultTitle, resultList, y_test, y_pre, save_path)
        # 保存模型
        model_path = os.path.join(save_path, 'Model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))


    return [ObjV_i]




# ===============  RandomForest =================
# n_estimators = [50,100,200,300,400,500,600,700,800]
# criterion = ["mse"]
# max_features = ["None"]
# max_leaf_nodes = ["None"]
# min_samples_split = [2,3]
# min_samples_leaf = [1,2,3]
# oob_score = ["True","False"]
# random_state = 17
# n_jobs = -1

class RandomForestProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, PoolType, save_path, X_train, X_test, y_train, y_test):  # PoolType是取值为'Process'或'Thread'的字符串
        name = 'RandomForest'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        # 變量 dictionary

        Dim = 4  # 初始化Dim（决策变量维数）
        varTypes = [1, 1 , 1 ,1]  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [30, 2 , 1 , 0] # 决策变量下界
        ub = [800,  3 , 3 , 1]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 获取目标训练集和 测试集
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
        print(len(self.X_train), len(self.y_train), len(self.X_test), len(self.y_test))
        # 設置保存路徑
        self.save_path = save_path
        # 设置用多线程还是多进程
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(num_cores)  # 设置池的大小
            self.lock = Manager().Lock()  # 创建锁的方式不一样 其他都一样

    def aimFunc(self, pop):  # 目标函数，采用多线程加速计算
        Vars = pop.Phen  # 得到决策变量矩阵
        args = list(
            zip(list(range(pop.sizes)),
                [Vars] * pop.sizes,
                [self.X_train] * pop.sizes,
                [self.y_train] * pop.sizes,
                [self.X_test] * pop.sizes,
                [self.y_test] * pop.sizes,
                [self.lock] * pop.sizes,
                [self.save_path] * pop.sizes)
        )
        if self.PoolType == 'Thread':
            pop.ObjV = np.array(list(self.pool.map(subAimFuncRandomForest, args)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFuncRandomForest, args)
            result.wait()
            pop.ObjV = np.array(result.get())

def subAimFuncRandomForest(args):
    i = args[0]
    Vars = args[1]
    X_train = args[2]
    y_train = args[3]
    X_test = args[4]
    y_test = args[5]
    lock = args[6]
    save_path = args[7]

    # n_estimators = [50,100,200,300,400,500,600,700,800]
    # criterion = ["mse"]
    # max_features = ["None"]
    # max_leaf_nodes = ["None"]
    # min_samples_split = [2,3]
    # min_samples_leaf = [1,2,3]
    # oob_score = ["True","False"]
    # random_state = 17
    # n_jobs = -1

    oob_score_List = ["True", "False"]
    n_estimators = int(Vars[i, 0])
    min_samples_split = int(Vars[i, 1])
    min_samples_leaf = int(Vars[i, 2])
    oob_score = oob_score_List[int(Vars[i, 3])]

    parameterDict = {"n_estimators": n_estimators, "min_samples_split": min_samples_split, "min_samples_leaf": min_samples_leaf, "oob_score": oob_score}

    Regressor = RandomForestRegressor(**parameterDict)
    Regressor.fit(X_train, y_train)

    y_pre = Regressor.predict(X_test)
    regDict = reg_calculate(y_test, y_pre,X_test.shape[0], X_test.shape[1])
    ObjV_i = regDict["r2"]

    print("ObjV_i", ObjV_i)

    # make resultTitle
    resultTitle = []
    resultList = []

    for Title, Parameter in parameterDict.items():
        resultTitle.append(Title)
        resultList.append(str(Parameter))

    for Title, Parameter in regDict.items():
        resultTitle.append(Title)
        resultList.append(str(Parameter))

    with lock:
        count = save_results(resultTitle, resultList, y_test, y_pre, save_path)
        # 保存模型
        model_path = os.path.join(save_path, 'Model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))


    return [ObjV_i]
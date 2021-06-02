from Frame import Genetic_geatpy # 导入库
from sklearn import datasets     # 波士頓房價數據集
import os


if __name__ == '__main__':


    boston = datasets.load_boston()  # 导入波士顿房价数据
    from sklearn.model_selection import train_test_split
    # check data shape
    print("boston.data.shape %s , boston.target.shape %s " %(boston.data.shape ,boston.target.shape))
    train = boston.data  # sample
    target = boston.target  # target
    # 切割数据样本集合测试集
    X_train, X_test, Y_train, Y_test = train_test_split(train, target, test_size=0.2)  # 20%测试集；80%训练集

    # initialize the geneticOptimizer   自定義種群數量和 最大迭代數
    cb_Optimizer = Genetic_geatpy.geneticOptimizer(NIND = 30, MAXGEN = 2)

    cb_Optimizer.run(X_train, X_test, Y_train, Y_test ,os.path.join("./regression/" ,"catboost"),ensemble_model = "catboost")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "xgboost"),ensemble_model = "xgboost")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "lightgbm"),ensemble_model="lightgbm")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "Adaboost"),ensemble_model="Adaboost")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "SVR"),ensemble_model="SVR")  # error: the program  gets stuck when it running
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "gbdt"), ensemble_model="gbdt")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "bagging") , ensemble_model="bagging")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "ExtraTrees"), ensemble_model="ExtraTrees")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "RandomForest"),ensemble_model="RandomForest")





#### 2,june 2021 ####
集成學習第一版訓練工具
提供 catboost  xgboost lightgbm Adaboost SVR gbdt bagging ExtraTrees  RandomForest 等的回歸模型參數調整
可以通過遺傳算法自行尋找到最優的參數

使用樣例在genetic_Test.py 可以查看

值得注意的是 
由於使用了多進程 切記函數入口必須在
if __name__ == '__main__': 



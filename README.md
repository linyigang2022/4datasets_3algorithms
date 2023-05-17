# 4datasets_3algorithms

在 Covertype、Watch_acc、SUSY、HIGGS 四个数据集上复现 XGBoost、LightGBM、Bagging 三个算法



## 使用说明

1. 下载代码

```bash
git clone git@github.com:linyigang2022/4datasets_3algorithms.git
```

2. 修改 config.json 文件

   - 修改algorithm参数，可选 XGBoost、LightGBM、Bagging

   - 修改dataset参数，可选 Covertype、Watch_acc、SUSY、HIGGS

3. 运行 Runner.py



## 结果

| 算法\数据集 | 参数配置                                            | Covertype | Watch_acc | SUSY   | HIGGS |
| ----------- | --------------------------------------------------- | --------- | --------- | ------ | ----- |
| XGBoost     | n_estimators=2,  max_depth=2, learning_rate=1       | 0.6874    |           | 0.7533 |       |
| LightGBM    | num_leaves=31, learning_rate=0.05,  n_estimators=20 | 0.7687    |           | 0.7914 |       |
| Bagging     | estimator=DecisionTreeClassifier,n_estimators=3     | 0.9477    |           | 0.7506 |       |



## 注意

1. Watch_acc有4个文件，不确定是哪个文件所以没跑
2. HIGGS在我的电脑上跑不动
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from load_data import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class XGBoost:
    def run(self, data, target):
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.2)
        print("X_train:", X_train.shape)
        print("y_train:", y_train.shape)

        # create model instance
        # objective即使设置为binary:logistic，似乎也能进行多分类，应该会根据喂入的target自动调整,只是要求标签的编号要从0开始
        # 以下的设置来自官方文档，文档中使用iris数据集，3分类，所以这里我也不修改为 objective='multi:softmax 了
        bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
        # fit model
        bst.fit(X_train, y_train)
        # make predictions
        y_train_pred = bst.predict(X_train)
        y_test_pred = bst.predict(X_test)
        print("Train data accuracy:", accuracy_score(y_true=y_train, y_pred=y_train_pred))
        print("Test data accuracy:", accuracy_score(y_true=y_test, y_pred=y_test_pred))
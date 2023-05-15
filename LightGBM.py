from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from load_data import load_data

class LightGBM:
    def run(self, data, target):

        # split data to [[0.8,0.2],01]
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.2)

        # train_data.save_binary("/home/fonttian/Data/UCI/wine/wine_lightgbm_train.bin")

        gbm = LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=20)
        gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        # make predictions
        y_train_pred = gbm.predict(X_train)
        y_test_pred = gbm.predict(X_test)
        print("Train data accuracy:", accuracy_score(y_true=y_train, y_pred=y_train_pred))
        print("Test data accuracy:", accuracy_score(y_true=y_test, y_pred=y_test_pred))
from sklearn import datasets
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from load_data import load_data

class Bagging:
    def run(self, data, target):
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.2)

        dtree = DecisionTreeClassifier(random_state = 22)
        dtree.fit(X_train, y_train)

        y_pred = dtree.predict(X_test)
        #
        #
        print("Train data accuracy:", accuracy_score(y_true=y_train, y_pred=dtree.predict(X_train)))
        print("Test data accuracy:", accuracy_score(y_true=y_test, y_pred=y_pred))

        bag = BaggingClassifier(
                  estimator=dtree,
                  n_estimators=3,
                  random_state=0)
        bag = bag.fit(X_train, y_train)
        # Predicting the training set
        y_train_pred = bag.predict(X_train)
        # Predicting the test set
        y_test_pred = bag.predict(X_test)
        print("Train data accuracy:", accuracy_score(y_true=y_train, y_pred=y_train_pred))
        print("Test data accuracy:", accuracy_score(y_true=y_test, y_pred=y_test_pred))
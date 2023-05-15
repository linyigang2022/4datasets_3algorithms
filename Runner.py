import importlib
import json

import numpy as np
import pandas as pd


class Runner:
    def __init__(self, config):
        self.config = json.load(open(config, encoding='utf-8'))
        print(self.config)
        self.dataset = self.config[self.config["dataset"]]
        algorithm_class = importlib.import_module(self.config["algorithm"])
        print(self.config["algorithm"])
        self.algorithm = getattr(algorithm_class, self.config["algorithm"])()
        print(self.algorithm)

    def load_data(self):
        if self.dataset["name"]=="Covertype":
            df = pd.read_csv(self.dataset["path"], header=None, sep=',')
            print(df.shape)
            np_dataset = np.array(df)
            # -1 是因为 ValueError: Invalid classes inferred from unique values of `y`.  Expected: [0 1 2 3 4 5 6], got [1 2 3 4 5 6 7]
            X, y = np_dataset[:, 0:-1], np_dataset[:, -1] - 1
            print(X.shape)
            print(y.shape)
            return X, y
        elif self.dataset["name"]=="Watch_acc":
            pass
        elif self.dataset["name"]=="SUSY" or self.dataset["name"]=="HIGGS":
            df = pd.read_csv(self.dataset["path"], header=None, sep=',')
            print(df.shape)
            np_dataset = np.array(df)
            X, y = np_dataset[:, 1:], np_dataset[:, 0]
            print(X.shape)
            print(y.shape)
            return X, y
        else:
            print("error dataset setting")

    def run(self):
        data, target = self.load_data()
        self.algorithm.run(data, target)

if __name__ == '__main__':
    config = "./config.json"
    runner = Runner(config)
    runner.run()

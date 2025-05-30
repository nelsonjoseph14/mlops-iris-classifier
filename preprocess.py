import pandas as pd

def load_and_preprocess(path="data/iris.csv"):
    df = pd.read_csv(path)
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y

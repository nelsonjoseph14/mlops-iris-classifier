from sklearn.datasets import load_iris
import pandas as pd

def save_iris_dataset():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df.to_csv('data/iris.csv', index=False)

if __name__ == "__main__":
    save_iris_dataset()

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

#Load boston housing dataset as an example

x = load_boston()
df = pd.DataFrame(x.data, columns = x.feature_names)
df["MEDV"] = x.target
X = df.drop("MEDV",1)   #Feature Matrix
Y = df["MEDV"]          #Target Variable
print(df.head())

# X = boston["data"]
# print(type(X),X.shape)
# Y = boston["target"]
names = x.feature_names
# print(names)
rf = RandomForestRegressor()
rf.fit(X, Y)
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True))
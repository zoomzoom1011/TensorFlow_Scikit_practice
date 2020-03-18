# 简易多层感知神经网络示例
from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd
# 加载，预处理数据集
dataset = pd.read_csv("pima-indians-diabetes.csv")

dataset.columns = ['country','year','population',
					'continent','life_exp','gdp_per_cap', 
					 'continent1','life_exp1','gdp_per_cap1']
print(dataset.head())
X = dataset.drop(['gdp_per_cap1'], axis=1)
# X = dataset.iloc[:, 0:8]
Y = dataset.loc[:, 'gdp_per_cap1']
# Y = dataset.loc[:,'dGH']
print(X.head())
print(Y.head())

print(X.shape)

# 1. 定义模型
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# 2. 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 3. 训练模型
history = model.fit(X, Y, nb_epoch=100, batch_size=10)
# 4. 评估模型
loss, accuracy = model.evaluate(X, Y)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
# 5. 数据预测
probabilities = model.predict(X)
# print(probabilities)
predictions = [float(nump.round(x)) for x in probabilities]
# print(predictions)
accuracy = numpy.mean(predictions == Y)
print("Prediction Accuracy: %.2f%%" % (accuracy*100))
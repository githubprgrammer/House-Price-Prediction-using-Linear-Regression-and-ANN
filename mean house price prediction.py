import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.neural_network import MLPRegressor

#load dataset
housing = fetch_california_housing()

#get input values
x = housing.data
#get output values
y = housing.target

regression = linear_model.LinearRegression()
#split data in train and test data with ratio of 80% and 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#training on train data and get model
regression.fit(x_train, y_train)
# calculate prediction of testdata
predict = regression.predict(x_test)

#mean square error of linear regression
mse = np.mean((predict - y_test) ** 2)

#Neural Network model
mlp = MLPRegressor(hidden_layer_sizes=(5, ), activation='logistic', solver='adam', learning_rate='constant', random_state=30, max_iter=1000)
#train neural network
mlp.fit(x_train, y_train)
# calculate prediction of testdata
predict2 = mlp.predict(x_test)

#mean square error of neural network
mse2 = np.mean((predict2 - y_test) ** 2)




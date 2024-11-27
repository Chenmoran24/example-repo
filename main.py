import numpy as np
from sklearn import datasets, linear_model, metrics

x = 7
print(x)
# Load diabetes dataset
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data # matrix of dimensions 442x10
# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# with scikit learn:
# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)
# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)
# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
mean_squared_error = metrics.mean_squared_error(diabetes_y_test, diabetes_y_pred)
print("Mean squared error: %.2f" % mean_squared_error)
print("="*80)

# train
X = diabetes_X_train
y = diabetes_y_train
# train: init
W = ...
b = ...
learning_rate = ...
epochs = ...
# train: gradient descent
for i in range(epochs):
 # calculate predictions
 # TODO
 # calculate error and cost (mean squared error - use can use th
e imported function metrics.mean_squared_error)
 # TODO
 # calculate gradients
 # TODO
 # update parameters
 # TODO
 # diagnostic output
 if i % 5000 == 0:
 print("Epoch %d: %f" % (i, mean_squared_error))
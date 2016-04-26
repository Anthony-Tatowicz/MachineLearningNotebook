from tensorflow.contrib import skflow
from sklearn.datasets import load_diabetes
from sklearn import cross_validation, metrics
from sklearn import preprocessing

diabetes = load_diabetes()

X, y = diabetes.data, diabetes.target
print('Shape : X {} y {}'.format(X.shape, y.shape))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
    test_size=0.1, random_state=42)

print(X_train.shape, X_test.shape)
scalar = preprocessing.StandardScaler()

X_train = scalar.fit_transform(X_train)

regressor = skflow.TensorFlowDNNRegressor(hidden_units=[10, 10, 10],
                    learning_rate=0.05, batch_size=40, steps=1000)

regressor.fit(X_train, y_train)

X_test = scalar.fit_transform(X_test)

pred = regressor.predict(X_test)

score = metrics.mean_squared_error(pred, y_test)

print('MSE : {0:f}'.format(score))

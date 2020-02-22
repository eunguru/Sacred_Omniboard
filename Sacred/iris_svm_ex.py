import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics

C = 1.0
gamma = 0.7
kernel = 'rbf'

data_url = 'https://github.com/pandas-dev/pandas/raw/master/pandas/tests/data/iris.csv'
iris_df = pd.read_csv(data_url)
print("iris_df.shape: {}".format(iris_df.shape))

iris_data = iris_df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
iris_target = iris_df['Name']

data = iris_data.values
target = iris_target.values
 
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size = 1/3,
                                                    random_state=0)

print("X_train.shape: {}, X_test.shape: {}".format(X_train.shape,
                                                    X_test.shape))
print("y_train.shape: {}, y_train.shape: {}".format(y_train.shape,
                                                    y_test.shape))
    
clf = svm.SVC(C, kernel, gamma=gamma)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("predict: {}".format(y_pred))

accuracy = metrics.accuracy_score(y_test, y_pred);
print("accuracy: {}".format(accuracy))





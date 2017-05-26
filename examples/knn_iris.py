from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from knn import knn

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, T_train, T_test = train_test_split(X, y, test_size=0.3)

pipeline = Pipeline([
    ("knn_clf", knn.KNN(7))
    ])
pipeline.fit(X_train, T_train)
y_pred = pipeline.predict(X_test)
print(accuracy_score(y_pred, T_test))

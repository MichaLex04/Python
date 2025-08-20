import sklearn 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris=load_iris()
x=iris.data
y=iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model= DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred=model.predict([[5.9, 3.0, 5.1, 1.8]])
print(y_pred)

iris.target_names[y_pred]
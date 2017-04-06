#!/usr/bin/env python
#1 - Import dataset
from sklearn.datasets import load_iris

iris = load_iris()
print(iris.feature_names)
print(iris.target_names)
print(iris.data[0])
print(iris.target[0])

for  i in range(len(iris.target)):
    print("Example %d:  label %s, features %s" % (i, iris.target[i], iris.data[i]))

#2 - Train a Classifier
import numpy as np
from sklearn import tree

test_idx = [0, 50, 100]

#2.1 - training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#2.2 - testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#3 - train
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))

#4 - Visualize the tree
tree.export_graphviz(clf,out_file='tree.dot')
#dot -Tpdf iris.dot -o iris.pdf
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))

# viz code
from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf,
        out_file=dot_data,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True, rounded=True,
        impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
print(test_data[0], test_target[0])
'''
Features: sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)
Labels: setosa, versicolor, virginica
Data of 1st Flower: 5.1, 3.5, 1.4, 0.2
Target (label): 0 (Setosa)

Fisher's Iris Data
#Sepal length    #Sepal width   #Petal length   #Petal width    #Species
5.1	3.5	1.4	0.2 I. setosa
4.9	3.0	1.4	0.2	I. setosa
4.7	3.2	1.3	0.2	I. setosa
4.6	3.1	1.5	0.2	I. setosa
5.0	3.6	1.4	0.2	I. setosa
5.4	3.9	1.7	0.4	I. setosa
4.6	3.4	1.4	0.3	I. setosa
5.0	3.4	1.5	0.2	I. setosa
4.4	2.9	1.4	0.2	I. setosa
4.9	3.1	1.5	0.1	I. setosa
5.4	3.7	1.5	0.2	I. setosa
4.8	3.4	1.6	0.2	I. setosa
4.8	3.0	1.4	0.1	I. setosa
4.3	3.0	1.1	0.1	I. setosa
5.8	4.0	1.2	0.2	I. setosa
5.7	4.4	1.5	0.4	I. setosa
5.4	3.9	1.3	0.4	I. setosa
5.1	3.5	1.4	0.3	I. setosa
5.7	3.8	1.7	0.3	I. setosa
5.1	3.8	1.5	0.3	I. setosa
5.4	3.4	1.7	0.2	I. setosa
5.1	3.7	1.5	0.4	I. setosa
4.6	3.6	1.0	0.2	I. setosa
5.1	3.3	1.7	0.5	I. setosa
4.8	3.4	1.9	0.2	I. setosa
5.0	3.0	1.6	0.2	I. setosa
5.0	3.4	1.6	0.4	I. setosa
5.2	3.5	1.5	0.2	I. setosa
5.2	3.4	1.4	0.2	I. setosa
4.7	3.2	1.6	0.2	I. setosa
4.8	3.1	1.6	0.2	I. setosa
5.4	3.4	1.5	0.4	I. setosa
5.2	4.1	1.5	0.1	I. setosa
5.5	4.2	1.4	0.2	I. setosa
4.9	3.1	1.5	0.2	I. setosa
5.0	3.2	1.2	0.2	I. setosa
5.5	3.5	1.3	0.2	I. setosa
4.9	3.6	1.4	0.1	I. setosa
4.4	3.0	1.3	0.2	I. setosa
5.1	3.4	1.5	0.2	I. setosa
5.0	3.5	1.3	0.3	I. setosa
4.5	2.3	1.3	0.3	I. setosa
4.4	3.2	1.3	0.2	I. setosa
5.0	3.5	1.6	0.6	I. setosa
5.1	3.8	1.9	0.4	I. setosa
4.8	3.0	1.4	0.3	I. setosa
5.1	3.8	1.6	0.2	I. setosa
4.6	3.2	1.4	0.2	I. setosa
5.3	3.7	1.5	0.2	I. setosa
5.0	3.3	1.4	0.2	I. setosa
7.0	3.2	4.7	1.4	I. versicolor
6.4	3.2	4.5	1.5	I. versicolor
6.9	3.1	4.9	1.5	I. versicolor
5.5	2.3	4.0	1.3	I. versicolor
6.5	2.8	4.6	1.5	I. versicolor
5.7	2.8	4.5	1.3	I. versicolor
6.3	3.3	4.7	1.6	I. versicolor
4.9	2.4	3.3	1.0	I. versicolor
6.6	2.9	4.6	1.3	I. versicolor
5.2	2.7	3.9	1.4	I. versicolor
5.0	2.0	3.5	1.0	I. versicolor
5.9	3.0	4.2	1.5	I. versicolor
6.0	2.2	4.0	1.0	I. versicolor
6.1	2.9	4.7	1.4	I. versicolor
5.6	2.9	3.6	1.3	I. versicolor
6.7	3.1	4.4	1.4	I. versicolor
5.6	3.0	4.5	1.5	I. versicolor
5.8	2.7	4.1	1.0	I. versicolor
6.2	2.2	4.5	1.5	I. versicolor
5.6	2.5	3.9	1.1	I. versicolor
5.9	3.2	4.8	1.8	I. versicolor
6.1	2.8	4.0	1.3	I. versicolor
6.3	2.5	4.9	1.5	I. versicolor
6.1	2.8	4.7	1.2	I. versicolor
6.4	2.9	4.3	1.3	I. versicolor
6.6	3.0	4.4	1.4	I. versicolor
6.8	2.8	4.8	1.4	I. versicolor
6.7	3.0	5.0	1.7	I. versicolor
6.0	2.9	4.5	1.5	I. versicolor
5.7	2.6	3.5	1.0	I. versicolor
5.5	2.4	3.8	1.1	I. versicolor
5.5	2.4	3.7	1.0	I. versicolor
5.8	2.7	3.9	1.2	I. versicolor
6.0	2.7	5.1	1.6	I. versicolor
5.4	3.0	4.5	1.5	I. versicolor
6.0	3.4	4.5	1.6	I. versicolor
6.7	3.1	4.7	1.5	I. versicolor
6.3	2.3	4.4	1.3	I. versicolor
5.6	3.0	4.1	1.3	I. versicolor
5.5	2.5	4.0	1.3	I. versicolor
5.5	2.6	4.4	1.2	I. versicolor
6.1	3.0	4.6	1.4	I. versicolor
5.8	2.6	4.0	1.2	I. versicolor
5.0	2.3	3.3	1.0	I. versicolor
5.6	2.7	4.2	1.3	I. versicolor
5.7	3.0	4.2	1.2	I. versicolor
5.7	2.9	4.2	1.3	I. versicolor
6.2	2.9	4.3	1.3	I. versicolor
5.1	2.5	3.0	1.1	I. versicolor
5.7	2.8	4.1	1.3	I. versicolor
6.3	3.3	6.0	2.5	I. virginica
5.8	2.7	5.1	1.9	I. virginica
7.1	3.0	5.9	2.1	I. virginica
6.3	2.9	5.6	1.8	I. virginica
6.5	3.0	5.8	2.2	I. virginica
7.6	3.0	6.6	2.1	I. virginica
4.9	2.5	4.5	1.7	I. virginica
7.3	2.9	6.3	1.8	I. virginica
6.7	2.5	5.8	1.8	I. virginica
7.2	3.6	6.1	2.5	I. virginica
6.5	3.2	5.1	2.0	I. virginica
6.4	2.7	5.3	1.9	I. virginica
6.8	3.0	5.5	2.1	I. virginica
5.7	2.5	5.0	2.0	I. virginica
5.8	2.8	5.1	2.4	I. virginica
6.4	3.2	5.3	2.3	I. virginica
6.5	3.0	5.5	1.8	I. virginica
7.7	3.8	6.7	2.2	I. virginica
7.7	2.6	6.9	2.3	I. virginica
6.0	2.2	5.0	1.5	I. virginica
6.9	3.2	5.7	2.3	I. virginica
5.6	2.8	4.9	2.0	I. virginica
7.7	2.8	6.7	2.0	I. virginica
6.3	2.7	4.9	1.8	I. virginica
6.7	3.3	5.7	2.1	I. virginica
7.2	3.2	6.0	1.8	I. virginica
6.2	2.8	4.8	1.8	I. virginica
6.1	3.0	4.9	1.8	I. virginica
6.4	2.8	5.6	2.1	I. virginica
7.2	3.0	5.8	1.6	I. virginica
7.4	2.8	6.1	1.9	I. virginica
7.9	3.8	6.4	2.0	I. virginica
6.4	2.8	5.6	2.2	I. virginica
6.3	2.8	5.1	1.5	I. virginica
6.1	2.6	5.6	1.4	I. virginica
7.7	3.0	6.1	2.3	I. virginica
6.3	3.4	5.6	2.4	I. virginica
6.4	3.1	5.5	1.8	I. virginica
6.0	3.0	4.8	1.8	I. virginica
6.9	3.1	5.4	2.1	I. virginica
6.7	3.1	5.6	2.4	I. virginica
6.9	3.1	5.1	2.3	I. virginica
5.8	2.7	5.1	1.9	I. virginica
6.8	3.2	5.9	2.3	I. virginica
6.7	3.3	5.7	2.5	I. virginica
6.7	3.0	5.2	2.3	I. virginica
6.3	2.5	5.0	1.9	I. virginica
6.5	3.0	5.2	2.0	I. virginica
6.2	3.4	5.4	2.3	I. virginica
5.9	3.0	5.1	1.8	I. virginica
'''

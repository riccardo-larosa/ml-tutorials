import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

print ('features: {}'.format(iris.feature_names))
print ('labels: {}'.format(iris.target_names))
# print ('first row of data: {} is a {}'.format(iris.data[0], iris.target[0]))
# for i in range(len(iris.target)):
#     print ("Example {}: label: {}, features {}".format(i, iris.target[i], iris.data[i]))
test_idx = [0, 50, 100]
# create training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)
# create testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# create classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# print (test_data)
print ('Results should be like this: {}'.format(test_target))
print ('Prediction results: {}'.format(clf.predict(test_data)))

# viz code
# from sklearn.externals.six import StringIO
# import pydot

import graphviz
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")

# show values for testing data so that you can look them up in
# decision tree
print ('features: {}'.format(iris.feature_names))
print (test_data[0])
print ('labels: {}'.format(iris.target_names))
print (test_target[0])

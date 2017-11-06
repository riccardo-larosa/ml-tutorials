from sklearn import tree

# features are weight and skin type where 0 = bumpy, 1 = smooth
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# labels are apple (=0) or oranges (=1)
labels = [0, 0, 1, 1]
# create a classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
# now we have a classifier, let's see if it can predict
# something that weighs 160gr and is bumpy
print (clf.predict([[160, 0]]))
# it outputs 1 :orange !

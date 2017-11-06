# import a dataset
from sklearn import datasets
iris = datasets.load_iris()

# features X
X = iris.data
# label y
y = iris.target

from sklearn.cross_validation import train_test_split
# split train and test 50/50
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)


def calculate_accuracy(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    # now that I have the classifier let's see how well it works on test data
    predictions = classifier.predict(X_test)
    # and calculate the accuracy
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, predictions)
    print('accuracy with {}: {}'.format(classifier, accuracy))



# use DecisionTreeClassifier
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
calculate_accuracy(my_classifier, X_train, X_test, y_train, y_test)

# use KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
my_knclassifier = KNeighborsClassifier()
calculate_accuracy(my_knclassifier, X_train, X_test, y_train, y_test)

# write our own classifer
from scrappy_knn import scrappy_knn
my_scrappy_knn_classifier = scrappy_knn()
calculate_accuracy(my_scrappy_knn_classifier, X_train, X_test, y_train, y_test)

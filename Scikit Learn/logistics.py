# python 3.7
# Scikit-learn ver. 0.23.2
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import RidgeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import plot_roc_curve
from sklearn.svm import SVC
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# matplotlib 3.3.1
from matplotlib import pyplot

digits = load_digits()
digitsX = digits.images.reshape(len(digits.images), 64)
digitsY = digits.target
trainX, testX, trainY, testY = train_test_split(
    digitsX, digitsY, test_size = 0.3, shuffle = True
    )

classifier = LogisticRegression(max_iter = 10000)
classifier.fit(trainX, trainY)
preds = classifier.predict(testX)

correct = 0
incorrect = 0
for pred, gt in zip(preds, testY):
    if pred == gt: correct += 1
    else: incorrect += 1
print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")

plot_confusion_matrix(classifier, testX, testY)
pyplot.show()


iris = load_iris()
print(iris)

digitsX = iris.data.reshape(len(iris.data), 4)
digitsY = iris.target
trainX, testX, trainY, testY = train_test_split(
    digitsX, digitsY, test_size = 0.3, shuffle = True
    )

classifier = RidgeClassifier(max_iter = 10000)
classifier.fit(trainX, trainY)
preds = classifier.predict(testX)


correct = 0
incorrect = 0
for pred, gt in zip(preds, testY):
    if pred == gt: correct += 1
    else: incorrect += 1
print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")

plot_confusion_matrix(classifier, testX, testY)
pyplot.show()



#X_train, X_test, y_train, y_test = train_test_split(digitsX, digitsY, random_state=42)
#svc = SVC(random_state=42)
#svc.fit(X_train, y_train)

#svc_disp = plot_roc_curve(svc, X_test, y_test)

#ax = pyplot.gca()
#rfc_disp = plot_roc_curve(iris, testX, testY, ax=ax, alpha=0.8)
#svc_disp.plot(ax=ax, alpha=0.8)


boston = load_boston()
print(boston)
digitsX = boston.data.reshape(len(boston.data), 13)
digitsY = boston.target
trainX, testX, trainY, testY = train_test_split(
    digitsX, digitsY, test_size = 0.3, shuffle = True
    )

classifier = SGDRegressor()
classifier.fit(trainX, trainY)
preds = classifier.predict(testX)

correct = 0
incorrect = 0
for pred, gt in zip(preds, testY):
    if pred == gt: correct += 1
    else: incorrect += 1
print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")

# House prices are difficult to scale because house prices are unpredictable, https://scikit-learn.org/stable/datasets/index.html#boston-dataset,
# some important variables are considered, but not all.

# Scaling would be a good idea for this dataset, PowerTransformer because nonlinear transformations, and data is mapped to a normal distribution to stabilize variance

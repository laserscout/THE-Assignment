import numpy as np
import pandas as pd

class bcolors:
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	YELLOW = '\033[93m'
	RED = '\033[91m'
	ENDC = '\033[0m'

def simpleTrain(dataset, target, model='all'):
	from sklearn.model_selection import train_test_split
	trainingSet, testSet, trainingTarget, testTarget = train_test_split(dataset, target,
		test_size=0.4, random_state=0)

	if model == 'svm' or model == 'all':
		# SVM training
		from sklearn.svm import SVC
		clf = SVC(gamma='scale')
		clf.fit(trainingSet, trainingTarget)
		svmAccuracy = clf.score(testSet, testTarget)

	if model == 'dtree' or model == 'all':
		# Decision tree
		from sklearn import tree
		clf = tree.DecisionTreeClassifier()
		clf.fit(trainingSet, trainingTarget)
		dtreeAccuracy = clf.score(testSet, testTarget)

	if model == 'nn' or model == 'all':
		# Multi-layer Perceptron
		from sklearn.neural_network import MLPClassifier
		clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=2)
		clf.fit(trainingSet, trainingTarget)
		nnAccuracy = clf.score(testSet, testTarget)

	if model == 'bayes' or model == 'all':
		# Naive Bayes
		from sklearn.naive_bayes import GaussianNB
		clf = GaussianNB()
		clf.fit(trainingSet, trainingTarget)
		bayesAccuracy = clf.score(testSet, testTarget)

	if model == 'all':
		return max([svmAccuracy, dtreeAccuracy, nnAccuracy, bayesAccuracy])
	elif model == 'svm':
		return svmAccuracy
	elif model == 'dtree':
		return dtreeAccuracy
	elif model == 'nn':
		return nnAccuracy
	elif model == 'bayes':
		return bayesAccuracy

def randomForest(dataset, target):
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.model_selection import train_test_split

	trainingSet, testSet, trainingTarget, testTarget = train_test_split(dataset,
		target, test_size=0.4, random_state=0)
	clf = RandomForestClassifier(n_estimators=500, criterion = 'entropy',
		n_jobs = -1, random_state = 4)
	clf = clf.fit(trainingSet, trainingTarget)
	print("Random forest accuracy: {0:.2f}".format(100*clf.score(testSet, testTarget)))

def kFCrossValid(dataset, target, model = 'svm'):
	from sklearn.model_selection import cross_val_score
	from sklearn import metrics
	from copy import deepcopy

	clf = None

	if model == 'svm':
		# SVM training
		from sklearn.svm import SVC
		clf = SVC(gamma='scale')
	elif model == 'dtree':
		# Decision tree
		from sklearn import tree
		clf = tree.DecisionTreeClassifier()
	elif model == 'nn':
		# Multi-layer Perceptron
		from sklearn.neural_network import MLPClassifier
		clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=2)
	elif model == 'bayes':
		# Naive Bayes
		from sklearn.naive_bayes import GaussianNB
		clf = GaussianNB()
	elif model == 'rndForest':
		from sklearn.ensemble import ExtraTreesClassifier
		clf = ExtraTreesClassifier(n_estimators=1500, criterion = 'entropy',
			n_jobs = -1, random_state = 4)
	else:
		print('Error. model specified not supported')
		return None

	from sklearn.model_selection import KFold
	kf = KFold(n_splits=5, shuffle=True, random_state=2)

	maxAccuracy = 0
	bestClf = None

	for k, (train_index, test_index) in enumerate(kf.split(dataset)):
		kTrainSet, kTestSet = dataset[train_index], dataset[test_index]
		kTrainTarget, kTestTarget = target[train_index], target[test_index]

		clf.fit(kTrainSet, kTrainTarget)
		acc = clf.score(kTestSet, kTestTarget)
		print("[fold {0}], score: {1:.2f}".format(k, 100*acc))

		if acc > maxAccuracy:
			maxAccuracy = acc
			bestClf = deepcopy(clf)

	return bestClf

# Prints a nice message to let the user know the module was imported
print(bcolors.BLUE + 'model_training loaded' + bcolors.ENDC)

# Enables executing the module as a standalone script
if __name__ == "__main__":
	import sys
	dataset = pd.read_pickle(sys.argv[1])
	target = dataset.pop('target')

	kFCrossValid(dataset.values, target, sys.argv[2])
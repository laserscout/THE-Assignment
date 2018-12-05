import numpy as np

class bcolors:
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	YELLOW = '\033[93m'
	RED = '\033[91m'
	ENDC = '\033[0m'

# def arrayFromJSON(JSONPath):

# Prints a nice message to let the user know the module was imported
print(bcolors.BLUE + 'model_training loaded' + bcolors.ENDC)

# Enables executing the module as a standalone script
if __name__ == "__main__":
	import sys
	dataset = np.load(sys.argv[1] + 'dataset.npy')
	target = np.load(sys.argv[1] + 'target.npy')
	featureKeys = np.load(sys.argv[1] + 'featureKeys.npy')

	row_idx = np.r_[0:10956, 13696:24653]
	trainingSet = np.copy(dataset[row_idx, :])
	trainingTarget = np.copy(target[row_idx])

	row_idx = np.r_[10956:13696, 24653:27392]
	testSet = np.copy(dataset[row_idx, :])
	testTarget = np.copy(target[row_idx])

	# ==========================================================================

	# SVM training
	from sklearn.svm import SVC
	print('Training...')
	clf = SVC(gamma='scale')
	clf.fit(trainingSet, trainingTarget)
	print('Testing...')
	print(clf.score(testSet, testTarget))

	# Χωρίς preprocessing 								=> 0.4999087424712539
	# Με Standardization 								=> 0.8906734805621463
	# Με Normalization 									=> 0.4999087424712539
	# Με stand. then norm. 								=> 0.7873699580215368
	# Με varReducedDataset + stand. 					=> 0.8826428180324877
	# Με perReducedDataset + stand. 					=> 0.81529476181785

	# Με varReducedDataset + stand. + gamma = scale 	=> 0.8828253330899799
	# Με varReducedDataset + stand. + sigmoid kernel 	=> 0.5875159700675305
	# Με varReducedDataset + stand. + poly kernel dgr 5 => 0.8441321409016244

	# Decision tree
	from sklearn import tree
	print('Training...')
	clf = tree.DecisionTreeClassifier()
	clf.fit(trainingSet, trainingTarget)
	print('Testing...')
	print(clf.score(testSet, testTarget))

	# Με varReducedDataset + stand. 					=> 0.7541522175579485

	# Multi-layer Perceptron
	from sklearn.neural_network import MLPClassifier
	print('Training...')
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=2)
	clf.fit(trainingSet, trainingTarget)
	print('Testing...')
	print(clf.score(testSet, testTarget))

	# Με varReducedDataset + stand. και rndState = 2	=> 0.8647563423982478

	# Naive Bayes
	from sklearn.naive_bayes import GaussianNB
	print('Training...')
	clf = GaussianNB()
	clf.fit(trainingSet, trainingTarget)
	print('Testing...')
	print(clf.score(testSet, testTarget))

	# Με varReducedDataset + stand. 					=> 0.6557766015696295
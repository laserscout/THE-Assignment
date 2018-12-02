from os import listdir
from os.path import isfile, join
import numpy as np
import json

def arrayFromJSONs(JSONPath):
	with open(JSONPath) as jsonFile:
		rawJSON = json.load(jsonFile)

	keys = np.array([])
	values = np.array([])
	for featureKey, featureValues in rawJSON.items():
		if keys.size == 0 or values.size == 0:
			keys = np.array(featureKey)
			values = np.array(featureValues)
		else:
			keys = np.append(keys, (np.array(featureKey)))
			values = np.vstack((values, np.array(featureValues)))

	values = np.transpose(values)
	return keys, values

def createSingleFeaturesArray(musicJSONsPath, speechJSONsPath):
	dataset = np.array([])
	featureKeys = np.array([])

	# Reads the extracted features for the music class
	featuresFiles = [file for file in listdir(musicJSONsPath) if isfile(join(musicJSONsPath, file))]
	for file in featuresFiles:
		if dataset.size == 0:
			# Gets feature arrays
			featureKeys, musicFeatures = arrayFromJSONs(musicJSONsPath + file)
			# Appends the class to the arrays (0 for music, 1 for speech)
			musicClass = np.zeros((musicFeatures.shape[0]), dtype=int)
			musicFeatures = np.c_[musicFeatures, musicClass]
			dataset = np.copy(musicFeatures)
		else:
			# Gets feature arrays
			musicFeatures = arrayFromJSONs(musicJSONsPath + file)[1]
			# Appends the class to the arrays (0 for music, 1 for speech)
			musicFeatures = np.c_[musicFeatures, musicClass]
			dataset = np.vstack((dataset, musicFeatures))

	# Reads the extracted features for the speech class
	featuresFiles = [file for file in listdir(speechJSONsPath) if isfile(join(speechJSONsPath, file))]
	for file in featuresFiles:
		# Gets feature arrays
		speechFeatures = arrayFromJSONs(speechJSONsPath + file)[1]
		# Appends the class to the arrays (0 for music, 1 for speech)
		speechClass = np.ones((speechFeatures.shape[0]), dtype=int)
		speechFeatures = np.c_[speechFeatures, speechClass]
		dataset = np.vstack((dataset, speechFeatures))

	return dataset, featureKeys

# Details about this part can be found in the link bellow:
# https://scikit-learn.org/stable/modules/feature_selection.html
def featureSelection(dataset, featureKeys):
	# Selects features based on a variance threshold
	from sklearn.feature_selection import VarianceThreshold

	varianceThreshold = 0.72
	selector = VarianceThreshold(threshold = (varianceThreshold * (1 - varianceThreshold)))
	varReducedDataset = selector.fit_transform(dataset)
	isRetained = selector.get_support()

	print('Retaining features:')
	for index, retain in enumerate(isRetained):
		if retain and index < featureKeys.size:
			print(featureKeys[index], end='\t', flush=True)

	print('\n\nRemoving features:')
	for index, retain in enumerate(isRetained):
		if not retain and index < featureKeys.size:
			print(featureKeys[index], end='\t', flush=True)
	print('\n')

	# Selects features based on univariate statistical tests
	from sklearn.datasets import load_digits
	from sklearn.feature_selection import SelectPercentile, mutual_info_regression

	perReducedDataset = SelectPercentile(mutual_info_regression,
		percentile=33).fit_transform(dataset[:, :-1], dataset[:, -1])

	# TODO: change the return value after the values of the parameters are decided
	# and the feature selection is complete
	return dataset

# Details about this part can be found in the link bellow:
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
def standardization(dataset):
	from sklearn import preprocessing

	# Standardization
	scaledDataset = preprocessing.scale(dataset[:, :-1])
	scaledDataset = np.c_[scaledDataset, dataset[:, -1]]

	# Normalization
	scaledDataset = preprocessing.normalize(dataset[:, :-1], norm='l2')
	scaledDataset = np.c_[scaledDataset, dataset[:, -1]]

	# TODO: change the return value after the values of the parameters are decided
	# and the feature selection is complete
	return dataset

# Details about this part can be found in the link bellow:
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
def PCA(dataset):
	from sklearn.decomposition import PCA

	pca = PCA(n_components=10,svd_solver='full')
	transformedDataset = pca.fit(dataset[:, :-1]).transform(dataset[:, :-1])
	transformedDataset = np.c_[transformedDataset, dataset[:, -1]]

	# TODO: change the return value after the values of the parameters are decided
	# and the feature selection is complete
	return dataset

# Prints a nice message to let the user know the module was imported
print('feature_preprocessing loaded')

# Enables executing the module as a standalone script
if __name__ == "__main__":
	import sys
	dataset, featureKeys = createSingleFeaturesArray(sys.argv[1], sys.argv[2])
	PCA(standardization(featureSelection(dataset, featureKeys)))
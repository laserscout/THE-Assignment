from os import listdir
from os.path import isfile, join
import numpy as np
import json

class bcolors:
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	YELLOW = '\033[93m'
	RED = '\033[91m'
	ENDC = '\033[0m'

def arrayFromJSON(JSONPath):
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
	print(bcolors.YELLOW + 'Creating single features array' + bcolors.ENDC)
	dataset = np.array([])
	featureKeys = np.array([])

	# Reads the extracted features for the music class
	featuresFiles = [file for file in listdir(musicJSONsPath) if isfile(join(musicJSONsPath, file))]
	for file in featuresFiles:
		if dataset.size == 0:
			# Gets feature arrays
			featureKeys, musicFeatures = arrayFromJSON(musicJSONsPath + file)
			# Initializes dataset array
			dataset = np.copy(musicFeatures)
		else:
			# Gets feature arrays
			musicFeatures = arrayFromJSON(musicJSONsPath + file)[1]
			dataset = np.vstack((dataset, musicFeatures))

	# Initializes target array (0 for music)
	target = np.zeros((dataset.shape[0]), dtype=int)

	# Reads the extracted features for the speech class
	featuresFiles = [file for file in listdir(speechJSONsPath) if isfile(join(speechJSONsPath, file))]
	for file in featuresFiles:
		# Gets feature arrays
		speechFeatures = arrayFromJSON(speechJSONsPath + file)[1]
		dataset = np.vstack((dataset, speechFeatures))

	# Appends the new class to the target array (1 for speech)
	target = np.hstack((target, np.ones((dataset.shape[0] - target.size), dtype=int)))

	return dataset, target, featureKeys

# Details about this part can be found in the link bellow:
# https://scikit-learn.org/stable/modules/feature_selection.html
def featureSelection(dataset, target, featureKeys):
	# Selects features based on a variance threshold
	from sklearn.feature_selection import VarianceThreshold

	print(bcolors.YELLOW + 'Running variance threshold feature selection' + bcolors.ENDC)
	varianceThreshold = 0.1
	selector = VarianceThreshold(threshold = (varianceThreshold * (1 - varianceThreshold)))
	varReducedDataset = selector.fit_transform(dataset)
	isRetained = selector.get_support()
	varReducedFeatureKeys = featureKeys[isRetained]

	print(bcolors.GREEN + 'Retaining features:' + bcolors.ENDC)
	for index, retain in enumerate(isRetained):
		if retain and index < featureKeys.size:
			print(featureKeys[index], end='\t', flush=True)

	print(bcolors.RED + '\n\nRemoving features:' + bcolors.ENDC)
	for index, retain in enumerate(isRetained):
		if not retain and index < featureKeys.size:
			print(featureKeys[index], end='\t', flush=True)
	print('\n')

	# Selects features based on univariate statistical tests
	# from sklearn.datasets import load_digits
	# from sklearn.feature_selection import SelectPercentile, mutual_info_classif

	# print(bcolors.YELLOW + 'Running feature selection based on mutual information' + bcolors.ENDC)
	# percentileSelector = SelectPercentile(mutual_info_classif, percentile=33)
	# perReducedDataset = percentileSelector.fit_transform(dataset, target)
	# isRetained = percentileSelector.get_support()
	# perReducedFeatureKeys = featureKeys[isRetained]

	# print(bcolors.BLUE + 'Scores of features:' + bcolors.ENDC)
	# for index, score in enumerate(percentileSelector.scores_):
	# 	print(featureKeys[index] + ' => ' + str(score), end='\t\t', flush=True)
	# 	if index%2:
	# 		print('')
	# print('')

	# print(bcolors.GREEN + 'Retaining features:' + bcolors.ENDC)
	# for index, retain in enumerate(isRetained):
	# 	if retain and index < featureKeys.size:
	# 		print(featureKeys[index], end='\t', flush=True)

	# print(bcolors.RED + '\n\nRemoving features:' + bcolors.ENDC)
	# for index, retain in enumerate(isRetained):
	# 	if not retain and index < featureKeys.size:
	# 		print(featureKeys[index], end='\t', flush=True)
	# print('\n')

	# TODO: change the return value after the values of the parameters are decided
	# and the feature selection is complete
	return varReducedDataset, varReducedFeatureKeys

# Details about this part can be found in the link bellow:
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
def standardization(dataset):
	from sklearn import preprocessing

	print(bcolors.YELLOW + 'Running standardization' + bcolors.ENDC)
	# Standardization
	scaledDataset = preprocessing.scale(dataset)

	print(bcolors.YELLOW + 'Running normalization' + bcolors.ENDC)
	# Normalization
	normalizedDataset = preprocessing.normalize(dataset, norm='l2')

	# TODO: change the return value after the values of the parameters are decided
	# and the feature selection is complete
	return scaledDataset

# Details about this part can be found in the link bellow:
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
def PCA(dataset):
	from sklearn.decomposition import PCA

	print(bcolors.YELLOW + 'Running PCA' + bcolors.ENDC)
	pca = PCA(n_components=10, svd_solver='full')
	transformedDataset = pca.fit(dataset).transform(dataset)

	# TODO: change the return value after the values of the parameters are decided
	# and the feature selection is complete
	return dataset

# Prints a nice message to let the user know the module was imported
print(bcolors.BLUE + 'feature_preprocessing loaded' + bcolors.ENDC)

# Enables executing the module as a standalone script
if __name__ == "__main__":
	import sys
	dataset, target, featureKeys = createSingleFeaturesArray(sys.argv[1], sys.argv[2])
	dataset, featureKeys = featureSelection(dataset, target, featureKeys)
	newDataset = PCA(standardization(dataset))

	print(bcolors.GREEN + 'Saving results to files' + bcolors.ENDC)
	np.save('dataset.npy', newDataset)
	np.save('target.npy', target)
	np.save('featureKeys.npy', featureKeys)
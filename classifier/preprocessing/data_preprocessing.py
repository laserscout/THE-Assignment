from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
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
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
def standardization(dataset):
	from sklearn import preprocessing

	print(bcolors.YELLOW + 'Running standardization' + bcolors.ENDC)
	# Standardization
	scaledDataset = preprocessing.scale(np.float64(dataset))

	return scaledDataset

# Details about this part can be found in the link bellow:
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
def PCA(dataset):
	from sklearn.decomposition import PCA

	print(bcolors.YELLOW + 'Running PCA' + bcolors.ENDC)
	pca = PCA(n_components=10, svd_solver='full', whiten = True)
	transformedDataset = pca.fit(dataset).transform(dataset)

	return pca, transformedDataset

# Prints a nice message to let the user know the module was imported
print(bcolors.BLUE + 'feature_preprocessing loaded' + bcolors.ENDC)

# Enables executing the module as a standalone script
if __name__ == "__main__":
	import sys
	dataset, target, featureKeys = createSingleFeaturesArray(sys.argv[1], sys.argv[2])
	scaledDataset = standardization(dataset)

	print(bcolors.GREEN + 'Saving scaled results to file' + bcolors.ENDC)
	datasetFrame = pd.DataFrame(scaledDataset, columns = featureKeys)
	datasetFrame = datasetFrame.assign(target=target)
	datasetFrame.to_pickle("./dataset.pkl")

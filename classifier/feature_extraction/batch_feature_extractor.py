from os import listdir
from os.path import isfile, join
import multiprocessing as mp
import pandas as pd
if __name__ == '__main__':
	from feature_extractor import extractFeatures
else:
	from .feature_extractor import extractFeatures

class bcolors:
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	YELLOW = '\033[93m'
	RED = '\033[91m'
	ENDC = '\033[0m'

def batchExtract(audioFilesPath, featureFilesPath, sampleRate):
	audioFiles = [file for file in listdir(audioFilesPath) if isfile(join(audioFilesPath, file))]

	dataframesList = [None]*len(audioFiles)

	pool = mp.Pool()
	for process, file in enumerate(audioFiles):
		dataframesList[process] = pool.apply_async(extractFeatures,args=(audioFilesPath + file,
			featureFilesPath + file[0:file.rfind('.')] + '.json',int(sampleRate))).get()
	pool.close()
	pool.join()

	joinedDataset = pd.concat(dataframesList)

	print('Batch feature extraction finished successfully.')

	return joinedDataset

# Prints a nice message to let the user know the module was imported
print(bcolors.BLUE + 'batch_feature_extractor loaded' + bcolors.ENDC)

# Enables executing the module as a standalone script
if __name__ == "__main__":
	import sys
	batchExtract(sys.argv[1], sys.argv[2], sys.argv[3])
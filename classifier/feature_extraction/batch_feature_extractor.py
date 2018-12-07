from os import listdir
from os.path import isfile, join
import multiprocessing as mp
from .feature_extractor import extractFeatures

class bcolors:
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	YELLOW = '\033[93m'
	RED = '\033[91m'
	ENDC = '\033[0m'

def batchExtract(audioFilesPath, featureFilesPath, sampleRate):
	audioFiles = [file for file in listdir(audioFilesPath) if isfile(join(audioFilesPath, file))]

	# Without multithreading
	# for file in audioFiles:
	# 	extractFeatures(audioFilesPath + file,
	# 		featureFilesPath + file[0:file.rfind('.')] + '.json', int(sampleRate))

	pool = mp.Pool(processes = 8)
	for file in audioFiles:
		pool.apply(extractFeatures,args=(audioFilesPath + file,
			featureFilesPath + file[0:file.rfind('.')] + '.json',int(sampleRate)))

	print('Batch feature extraction finished successfully.')

# Prints a nice message to let the user know the module was imported
print(bcolors.BLUE + 'batch_feature_extractor loaded' + bcolors.ENDC)

# Enables executing the module as a standalone script
if __name__ == "__main__":
	import sys
	batchExtract(sys.argv[1], sys.argv[2], sys.argv[3])
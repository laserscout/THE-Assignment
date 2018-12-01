import sys
from os import listdir
from os.path import isfile, join
import multiprocessing as mp
from feature_extractor import extractFeatures

audioFiles = [file for file in listdir(sys.argv[1]) if isfile(join(sys.argv[1], file))]

# Without multithreading
# for file in audioFiles:
# 	extractFeatures(sys.argv[1] + file,
# 		sys.argv[2] + file[0:file.rfind('.')] + '.json', int(sys.argv[3]))

pool = mp.Pool(processes = 8)
[pool.apply(extractFeatures, args=(sys.argv[1] + file,
		sys.argv[2] + file[0:file.rfind('.')] + '.json',
		int(sys.argv[3]))) for file in audioFiles]

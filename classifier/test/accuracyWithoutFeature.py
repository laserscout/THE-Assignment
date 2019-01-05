import numpy as np
import pandas as pd
from sys import path
path.append('..')
from feature_extraction.batch_feature_extractor import batchExtract
from preprocessing.data_preprocessing import standardization
from training.model_training import simpleTrain

musicFeatures = batchExtract('../../dataset/music_wav/', '../feature_extraction/music_features/', 22050)
musicFeatures = musicFeatures.assign(target=0)
speechFeatures = batchExtract('../../dataset/speech_wav/', '../feature_extraction/speech_features/', 22050)
speechFeatures = speechFeatures.assign(target=1)

dataset = pd.concat([musicFeatures, speechFeatures])
target = dataset.pop('target').values

dataset = pd.DataFrame(standardization(dataset), columns = dataset.columns.values)

wholeAccuracy = simpleTrain(dataset, target, 'svm')
print('Accuracy using whole dataset = ' + str(wholeAccuracy))

damages = np.zeros(dataset.columns.values.size)

for index, key in enumerate(dataset.columns.values):
	acc = simpleTrain(dataset.drop(key, axis=1), target, 'svm')
	damages[index] = 100*(wholeAccuracy-acc)
	print('Accuracy without ' + key + '\t= ' + str(acc) +
		',\tdamage\t= ' + "%.2f" % damages[index] + '%')
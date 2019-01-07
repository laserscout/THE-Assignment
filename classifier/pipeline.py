import numpy as np
import pandas as pd
from feature_extraction.feature_extractor import extractFeatures
from feature_extraction.batch_feature_extractor import batchExtract
from preprocessing.data_preprocessing import arrayFromJSON, standardization, PCA
from training.model_training import simpleTrain, kFCrossValid

musicFeatures = batchExtract('../dataset/music_wav/', 'feature_extraction/music_features/', 22050)
musicFeatures = musicFeatures.assign(target=0)
speechFeatures = batchExtract('../dataset/speech_wav/', 'feature_extraction/speech_features/', 22050)
speechFeatures = speechFeatures.assign(target=1)

dataset = pd.concat([musicFeatures, speechFeatures])
target = dataset.pop('target').values

dataset = standardization(dataset)
# _, dataset = PCA(dataset)

print('Simple train accuracy achieved = ' + str(simpleTrain(dataset, target)))
kFCrossValid(dataset, target, model = 'svm')
clf = kFCrossValid(dataset, target, model = 'rndForest')

features = extractFeatures('compined.wav', 'tmp.json', 22050)
features = standardization(features)
audioClass = clf.predict(features)
print(audioClass)
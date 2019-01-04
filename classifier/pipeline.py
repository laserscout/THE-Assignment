import numpy as np
from feature_extraction.feature_extractor import extractFeatures
from feature_extraction.batch_feature_extractor import batchExtract
from preprocessing.data_preprocessing import arrayFromJSON, createSingleFeaturesArray, standardization, PCA
from training.model_training import simpleTrain, kFCrossValid

batchExtract('../dataset/music_wav/', 'feature_extraction/music_features/', 22050)
batchExtract('../dataset/speech_wav/', 'feature_extraction/speech_features/', 22050)

dataset, target, featureKeys = createSingleFeaturesArray(
	'feature_extraction/music_features/',
	'feature_extraction/speech_features/')

dataset = standardization(dataset)
# dataset = PCA(dataset)
print('Simple train accuracy achieved = ' + str(simpleTrain(dataset, target)))
kFCrossValid(dataset, target, model = 'svm')
clf = kFCrossValid(dataset, target, model = 'rndForest')

extractFeatures('compined.wav', 'featuresStream/tmp.json', 22050)
values = arrayFromJSON('featuresStream/tmp.json')[1]
values = standardization(values)
audioClass = clf.predict(values)
print(audioClass)
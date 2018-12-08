import numpy as np
from preprocessing.data_preprocessing import createSingleFeaturesArray, standardization, PCA
from classification_model_training.model_training import simpleTrain, kFCrossValid

dataset, target, featureKeys = createSingleFeaturesArray(
	'feature_extraction/music_features/',
	'feature_extraction/speech_features/')

dataset = standardization(dataset)
# dataset = PCA(dataset)
print('Simple train accuracy achieved = ' + str(simpleTrain(dataset, target)))
kFCrossValid(dataset, target, model = 'svm')
kFCrossValid(dataset, target, model = 'rndForest')

dataset = PCA(dataset)
print('Simple train accuracy achieved = ' + str(simpleTrain(dataset, target)))
kFCrossValid(dataset, target, model = 'svm')
kFCrossValid(dataset, target, model = 'rndForest')
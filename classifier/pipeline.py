import numpy as np
from preprocessing.data_preprocessing import createSingleFeaturesArray, standardization, PCA
from classification_model_training.model_training import simpleTrain

dataset, target, featureKeys = createSingleFeaturesArray(
	'feature_extraction/music_features/',
	'feature_extraction/speech_features/')

dataset = standardization(dataset)
dataset = PCA(dataset)
print('Max accuracy achieved = ' + str(simpleTrain(dataset, target)))
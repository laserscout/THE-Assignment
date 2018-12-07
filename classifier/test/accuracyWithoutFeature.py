import numpy as np
from sys import path
path.append('..')
from feature_extraction.batch_feature_extractor import batchExtract
from preprocessing.data_preprocessing import createSingleFeaturesArray, standardization
from classification_model_training.model_training import simpleTrain

batchExtract('../../dataset/music_wav/', '../feature_extraction/music_features/', 22050)
batchExtract('../../dataset/speech_wav/', '../feature_extraction/speech_features/', 22050)

dataset, target, featureKeys = createSingleFeaturesArray(
	'../feature_extraction/music_features/',
	'../feature_extraction/speech_features/')

dataset = standardization(dataset)

wholeAccuracy = simpleTrain(dataset, target, 'svm')
print('Accuracy using whole dataset = ' + str(wholeAccuracy))

damages = np.zeros(featureKeys.size)

for index, key in enumerate(featureKeys):
	acc = simpleTrain(np.delete(dataset, index, axis=1), target, 'svm')
	damages[index] = 100*(wholeAccuracy-acc)
	print('Accuracy without ' + key + '\t= ' + str(acc) +
		',\tdamage\t= ' + "%.2f" % damages[index] + '%')

# Accuracy using whole dataset = 0.9456968148215752
# Accuracy without Flat	= 0.9468832709683307,	damage	= -0.12%
# Accuracy without HFC	= 0.9462444099662316,	damage	= -0.05%
# Accuracy without LAtt	= 0.9485260564022999,	damage	= -0.28%
# Accuracy without SC	= 0.9453317513918044,	damage	= 0.04%
# Accuracy without SComp	= 0.9408597243771105,	damage	= 0.48%
# Accuracy without SDec	= 0.9455142831066898,	damage	= 0.02%
# Accuracy without SEFlat	= 0.9464269416811171,	damage	= -0.07%
# Accuracy without SF	= 0.9426850415259651,	damage	= 0.30%
# Accuracy without SFlat	= 0.9414985853792096,	damage	= 0.42%
# Accuracy without SLAtt	= 0.9440540293876061,	damage	= 0.16%
# Accuracy without SR	= 0.9452404855343616,	damage	= 0.05%
# Accuracy without SSDec	= 0.9466094733960025,	damage	= -0.09%
# Accuracy without ZCR	= 0.9443278269599343,	damage	= 0.14%
# Accuracy without mfcc0	= 0.9422287122387515,	damage	= 0.35%
# Accuracy without mfcc1	= 0.9446016245322625,	damage	= 0.11%
# Accuracy without mfcc10	= 0.9432326366706215,	damage	= 0.25%
# Accuracy without mfcc11	= 0.943050104955736,	damage	= 0.26%
# Accuracy without mfcc12	= 0.9412247878068815,	damage	= 0.45%
# Accuracy without mfcc2	= 0.9399470658026832,	damage	= 0.57%
# Accuracy without mfcc3	= 0.9408597243771105,	damage	= 0.48%
# Accuracy without mfcc4	= 0.940677192662225,	damage	= 0.50%
# Accuracy without mfcc5	= 0.939673268230355,	damage	= 0.60%
# Accuracy without mfcc6	= 0.9383955462261568,	damage	= 0.73%
# Accuracy without mfcc7	= 0.9399470658026832,	damage	= 0.57%
# Accuracy without mfcc8	= 0.942411243953637,	damage	= 0.33%
# Accuracy without mfcc9	= 0.942046180523866,	damage	= 0.37%
# Accuracy without using negative damage features = 0.9381217486538286
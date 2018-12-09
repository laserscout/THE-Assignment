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

# Accuracy using whole dataset = 0.951902893127681
# Accuracy without 4HzMod	= 0.9456968148215752,	damage	= 0.62%
# Accuracy without Flat	= 0.9523592224148946,	damage	= -0.05%
# Accuracy without HFC	= 0.9526330199872228,	damage	= -0.07%
# Accuracy without LAtt	= 0.9524504882723374,	damage	= -0.05%
# Accuracy without SC	= 0.9520854248425664,	damage	= -0.02%
# Accuracy without SComp	= 0.948160992972529,	damage	= 0.37%
# Accuracy without SDec	= 0.9520854248425664,	damage	= -0.02%
# Accuracy without SEFlat	= 0.9513552979830245,	damage	= 0.05%
# Accuracy without SF	= 0.9492561832618417,	damage	= 0.26%
# Accuracy without SFlat	= 0.9496212466916126,	damage	= 0.23%
# Accuracy without SLAtt	= 0.9498950442639409,	damage	= 0.20%
# Accuracy without SR	= 0.9523592224148946,	damage	= -0.05%
# Accuracy without SSDec	= 0.9519941589851236,	damage	= -0.01%
# Accuracy without ZCR	= 0.9500775759788264,	damage	= 0.18%
# Accuracy without mfcc0	= 0.9502601076937118,	damage	= 0.16%
# Accuracy without mfcc1	= 0.9510815004106964,	damage	= 0.08%
# Accuracy without mfcc10	= 0.9503513735511545,	damage	= 0.16%
# Accuracy without mfcc11	= 0.9492561832618417,	damage	= 0.26%
# Accuracy without mfcc12	= 0.9482522588299717,	damage	= 0.37%
# Accuracy without mfcc2	= 0.9446928903897052,	damage	= 0.72%
# Accuracy without mfcc3	= 0.9465182075385599,	damage	= 0.54%
# Accuracy without mfcc4	= 0.9470658026832162,	damage	= 0.48%
# Accuracy without mfcc5	= 0.9463356758236744,	damage	= 0.56%
# Accuracy without mfcc6	= 0.9452404855343616,	damage	= 0.67%
# Accuracy without mfcc7	= 0.9462444099662316,	damage	= 0.57%
# Accuracy without mfcc8	= 0.9490736515469563,	damage	= 0.28%
# Accuracy without mfcc9	= 0.9472483343981016,	damage	= 0.47%
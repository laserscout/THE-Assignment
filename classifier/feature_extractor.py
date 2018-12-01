import essentia
import essentia.standard
from essentia.standard import *
import essentia.streaming
from pylab import plot, show, figure, imshow
import matplotlib.pyplot as plt

def extractFeatures(audioPath, outputPath, sampleRate):
	# Loads the audio file specified
	loader = essentia.standard.MonoLoader(filename = audioPath, sampleRate = sampleRate)
	audio = loader()

	# Sets up the functions that will be used
	# TODO check if zero phase windowing is something we might want
	window = Windowing(normalized = False, size = 6144, type = 'hamming',
		zeroPhase = False)
	spectrum = Spectrum()
	mfcc = MFCC(inputSize = 6144, sampleRate = sampleRate)
	zcr = ZeroCrossingRate()
	sc = SpectralCentroidTime(sampleRate = sampleRate)
	sr = RollOff(sampleRate = sampleRate)
	sf = Flux()

	# Creates a pool to collect the values of the features
	pool = essentia.Pool()

	# Slices the signal into frames
	for frame in FrameGenerator(audio, frameSize = 6144, hopSize = 3072,
		startFromZero = True , validFrameThresholdRatio = 0.7):
		# Applies a window function to the frame
		windowedFrame = window(frame)

		# Computes time domain features
		frameZCR = zcr(windowedFrame)
		frameSC = sc(windowedFrame)

		# Computes spectral features
		frameSpectrum = spectrum(windowedFrame)
		frameSR = sr(frameSpectrum)
		frameSF = sf(frameSpectrum)
		# Discards the bands
		mfcc_coeffs = mfcc(frameSpectrum)[1]

		# Adds the values to the pool
		pool.add('ZCR', frameZCR)
		pool.add('SC', frameSC)
		pool.add('SR', frameSR)
		pool.add('SF', frameSF)
		pool.add('mfcc', mfcc_coeffs)

	YamlOutput(filename = outputPath, format = 'json', writeVersion = False)(pool)

# Prints a nice message to let the user know the module was imported
print('feature_extractor loaded')

# Enables executing the module as a standalone script
if __name__ == "__main__":
	import sys
	extractFeatures(sys.argv[1], sys.argv[2], int(sys.argv[3]))
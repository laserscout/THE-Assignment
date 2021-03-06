import essentia
import pandas as pd
import numpy as np
from essentia.standard import (MonoLoader, Windowing, Spectrum, MFCC,
	ZeroCrossingRate, SpectralCentroidTime, RollOff, Flux, Envelope,
	FlatnessSFX, LogAttackTime, StrongDecay, FlatnessDB, HFC,
	SpectralComplexity, Energy, FrameGenerator, YamlOutput)

# Disable annoying info level logging
essentia.log.infoActive = False

class bcolors:
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	YELLOW = '\033[93m'
	RED = '\033[91m'
	ENDC = '\033[0m'

def extractFeatures(audio, outputPath, sampleRate):
	if isinstance(audio, str):
		# Loads the audio file specified
		loader = MonoLoader(filename = audio, sampleRate = sampleRate)
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
	env = Envelope(attackTime = 2, releaseTime = 300, sampleRate = sampleRate)
	flat = FlatnessSFX()
	logAtt = LogAttackTime(sampleRate = sampleRate)
	strDec = StrongDecay(sampleRate = sampleRate)
	flatDB = FlatnessDB()
	hfc = HFC(sampleRate = sampleRate)
	spcComp = SpectralComplexity(sampleRate = sampleRate, magnitudeThreshold = 2)
	energy = Energy()

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
		frameEFlatness = flat(env(windowedFrame))
		frameLogAtt = logAtt(env(windowedFrame))[1]
		frameStrDec = strDec(windowedFrame)

		# Computes spectral features
		frameSpectrum = spectrum(windowedFrame)
		frameSR = sr(frameSpectrum)
		frameSF = sf(frameSpectrum)
		frameSEFlatness = flat(env(frameSpectrum))
		frameSLogAtt = logAtt(env(frameSpectrum))[1]
		frameSStrDec = strDec(frameSpectrum)
		frameSFlat = flatDB(frameSpectrum)
		frameHFC = hfc(frameSpectrum)
		frameSComp = spcComp(frameSpectrum)

		# Computes cepstral features
		# Discards the bands
		melBandEnergies, mfcc_coeffs = mfcc(frameSpectrum)

		fHzMod = _4HzModulation(melBandEnergies, energy(frameSpectrum), sampleRate)

		# Adds the values to the pool
		pool.add('ZCR', frameZCR)
		pool.add('SC', frameSC)
		pool.add('Flat', frameEFlatness)
		pool.add('LAtt', frameLogAtt)
		pool.add('SDec', frameStrDec)

		pool.add('SR', frameSR)
		pool.add('SF', frameSF)
		pool.add('SEFlat', frameSEFlatness)
		pool.add('SFlat', frameSFlat)
		pool.add('SLAtt', frameSLogAtt)
		pool.add('SSDec', frameSStrDec)
		pool.add('HFC', frameHFC)
		pool.add('SComp', frameSComp)

		for index, coef in enumerate(mfcc_coeffs):
			pool.add('mfcc' + str(index), coef)

		pool.add('4HzMod', fHzMod)

	YamlOutput(filename = outputPath, format = 'json', writeVersion = False)(pool)

	return pd.DataFrame(np.array([pool[i] for i in pool.descriptorNames()]).T, columns = pool.descriptorNames())

def _4HzModulation(melEnergies, frameEnergy, sampleRate):
	from scipy.signal import butter, sosfilt, sosfreqz
	nyquist = 0.5 * sampleRate
	lowCut = 3 / nyquist
	highCut = 5 / nyquist
	sos = butter(N = 2, Wn = [lowCut, highCut], analog = False, btype = 'band',
		output = 'sos')
	filtered = sosfilt(sos = sos, x = melEnergies)

	energySum = sum(filtered)
	return energySum / frameEnergy

# Prints a nice message to let the user know the module was imported
print(bcolors.BLUE + 'feature_extractor loaded' + bcolors.ENDC)

# Enables executing the module as a standalone script
if __name__ == "__main__":
	import sys
	extractFeatures(sys.argv[1], sys.argv[2], int(sys.argv[3]))
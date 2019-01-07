import pyaudio
import struct
import math
import pandas as pd

from feature_extraction.feature_extractor import extractFeatures
from preprocessing.data_preprocessing import createSingleFeaturesArray, standardization
from training.model_training import kFCrossValid, simpleTrain

FORMAT = pyaudio.paInt16
SHORT_NORMALIZE = (1.0/32768.0)
CHANNELS = 1
RATE = 22050
INPUT_BLOCK_TIME = int(6144 / RATE)
INPUT_FRAMES_PER_BLOCK = 6144

def classify(block, clf):
	import numpy as np

	audio_data = np.fromstring(block, np.int16)
	audio_data = audio_data.astype(int)

	values = extractFeatures(audio_data, 'tmp.json', RATE)
	audioClass = clf.predict(values)

	return audioClass

class micListener(object):
	def __init__(self):
		self.pa = pyaudio.PyAudio()
		self.stream = self.open_mic_stream()

		print('Training...')
		dataset, target, featureKeys = createSingleFeaturesArray(
			'feature_extraction/music_features/',
			'feature_extraction/speech_features/')

		self.clf = kFCrossValid(dataset, target, model = 'rndForest')
		print('Training done!')

	def stop(self):
		self.stream.close()

	def find_input_device(self):
		device_index = None
		for i in range( self.pa.get_device_count() ):
			devinfo = self.pa.get_device_info_by_index(i)
			print( "Device %d: %s"%(i,devinfo["name"]) )

			for keyword in ["mic","input"]:
				if keyword in devinfo["name"].lower():
					print( "Found an input: device %d - %s"%(i,devinfo["name"]) )
					device_index = i
					return device_index

		if device_index == None:
			print( "No preferred input found; using default input device." )

		return device_index

	def open_mic_stream( self ):
		device_index = self.find_input_device()

		stream = self.pa.open(format = FORMAT, channels = CHANNELS, rate = RATE,
			input = True, input_device_index = device_index,
			frames_per_buffer = INPUT_FRAMES_PER_BLOCK)

		return stream

	def listen(self):
		try:
			block = self.stream.read(INPUT_FRAMES_PER_BLOCK)
		except IOError as e:
			# dammit.
			print('IOError!')
			return

		audioClass = classify(block, self.clf)
		print(audioClass)

if __name__ == "__main__":
	micStr = micListener()

	for i in range(300):
		micStr.listen()
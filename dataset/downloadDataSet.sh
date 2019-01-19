#!/bin/bash

wget http://opihi.cs.uvic.ca/sound/music_speech.tar.gz
tar xf music_speech.tar.gz music_speech/music_wav music_speech/speech_wav
mv music_speech/music_wav .
mv music_speech/speech_wav .
rmdir music_speech
rm music_speech.tar.gz

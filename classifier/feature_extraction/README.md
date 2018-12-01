# Feature extraction

The file `feature_extractor` is a python module that uses the open-source library [Essentia](http://essentia.upf.edu/documentation/index.html) to extract audio features from an audio file in the path specified in the first parameter and save the features' values to a json file in the path specified in the second parameter.

The module can be imported or executed as a script using one of the following commands
`python feature_extractor.py <audio_file_path> <extracted_features_file_path> <audio_file_sample_rate>`
or
`python3 feature_extractor.py <audio_file_path> <extracted_features_file_path> <audio_file_sample_rate>`

A python script is also provided for a batch feature extraction. The script can be executed using one of the following commands:
`python batch_feature_extractor.py <audio_files_directory> <feature_files_directory> <audio_files_sample_rate>`
or
`python3 batch_feature_extractor.py <audio_files_directory> <feature_files_directory> <audio_files_sample_rate>`

**Dependencies:**
- essentia
- numpy
- scipy

All dependencies are available both for python2 and python3 versions and can all be installed using the commands `pip install <package_name>` or `pip3 install <package_name>` for python2 and python3 respectively.
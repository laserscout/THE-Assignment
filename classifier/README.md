# Feature extraction

The file `feature_extractor` is a python module that uses the open-source library [Essentia](http://essentia.upf.edu/documentation/index.html) to extract audio features from a file in the path specified in the first parameter and save the features' values to a binary file in the path specified in the second parameter.

**Dependencies:**
- essentia
- numpy
- scipy
- matplotlib

All dependencies are available both for python2 and python3 versions and can all be installed using the commands `pip install <package_name>` or `pip3 install <package_name>` for python2 and python3 respectively.

The module can be imported or executed as a script using one of the following commands
`python feature_extractor.py <audio_file_path> <extracted_features_file_path> <audio_file_sample_rate>`
or
`python3 feature_extractor.py <audio_file_path> <extracted_features_file_path> <audio_file_sample_rate>`
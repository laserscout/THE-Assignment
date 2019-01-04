import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

datasetArray = np.load('../preprocessing/dataset.npy')
target = np.load('../preprocessing/target.npy')
featureKeysVector = np.load('../preprocessing/featureKeys.npy')

dataset = pd.DataFrame(datasetArray)
dataset.columns = featureKeysVector

sns.relplot(x="4HzMod", y="Flat", data=dataset[["4HzMod", "Flat"]], hue = target, style = target)
sns.jointplot(x="SLAtt", y="ZCR", data=dataset[["SLAtt", "ZCR"]]);
sns.pairplot(data=dataset[["SDec", "Flat", "mfcc2", "HFC"]]);
plt.show()

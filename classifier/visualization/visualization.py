import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

dataset = pd.read_pickle('../preprocessing/dataset.pkl')
target = dataset.pop('target')

# Feature evaluation
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier(n_estimators=1000)
clf = clf.fit(dataset, target)

model = SelectFromModel(clf, prefit=True, max_features = 6)
print('Retaining features:')
print(dataset.columns.values[model.get_support()])
reducedDataset = pd.DataFrame(model.transform(dataset),
	columns = dataset.columns.values[model.get_support()])

# Every combination of the 6 best features with length equal to 4 features
import itertools
featureCombinations = itertools.combinations(range(6), 4)

for plotIndex, subset in enumerate(featureCombinations):
	featurePlot = sns.pairplot(data=(reducedDataset.iloc[:, list(subset)]).assign(target=target),
		hue='target', palette='Set1', vars=reducedDataset.columns.values[list(subset)]);
	featurePlot.fig.savefig("output/figure_" + str(plotIndex+1) + ".png")

# sns.relplot(x="4HzMod", y="Flat", data=dataset[["4HzMod", "Flat"]], hue = target, style = target)
# sns.jointplot(x="SLAtt", y="ZCR", data=dataset[["SLAtt", "ZCR"]]);
# plt.show()

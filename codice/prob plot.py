# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as sc
import statsmodels.graphics.gofplots as sm

data = pd.read_csv("csvume/dataset.csv")
list = ['nameProject','testCase', "Unnamed: 0", "isFlaky"]
data = data.drop(list,axis = 1 )
data = data.drop(data[data.numCoveredLines > 20].index)
# define distributions
standard_norm = data.numCoveredLines
print('Describe di numCoveredLines')
print(standard_norm.describe())

# plots for standard distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 7))
sns.histplot(standard_norm, kde=True, color='blue', ax=ax[0])
sm.ProbPlot(standard_norm).qqplot(line='s', ax=ax[1])
plt.show()

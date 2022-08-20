import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import time
from subprocess import check_output

from sklearn import preprocessing

pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

data = pd.read_csv("csvume/dataset.csv")
data = data.drop(data[data.numCoveredLines > 18].index)
# y e j sono le potenziali variabili dipendenti
y = data.numCoveredLines
j = data.projectSourceLinesCovered
list = ['id','nameProject','testCase', "Unnamed: 0"]
x = data.drop(list,axis = 1 )
#print(y.describe())

y.loc[y < 9] = 0
y.loc[y >= 9] = 1

#grafo a violino per determinare l'info gain di una feature
# fractal_dimension_mean feature, median of the Malignant and Benign does not looks like separated so it does not gives good information for classification.
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,10:20]],axis=1)     #0:10 - 10:20 - 20:26
data = pd.melt(data,id_vars="numCoveredLines",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="numCoveredLines", data=data,split=True, inner="quart")
plt.xticks(rotation=90)
plt.show()
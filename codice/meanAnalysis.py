import pandas as pd
import seaborn as sns
import pylab

pd.options.display.width = 0
df = pd.read_csv("dataset.csv")
# calcolo il valore medio di numCoveredLines per ogni test
print(df['numCoveredLines'].mean()) # la media è 9 linee di codice per test
# vado a suddividere i tests del dataset in underDF e overDF
underDF = df[df['numCoveredLines'] < 9]
overDF = df[df['numCoveredLines'] >= 9]
print(df.shape, underDF.shape, overDF.shape) #3735 case test in overDF e 6050 in underDF

#riprovo col pearson per cercare correlazioni nelle features sulle entity flaky e non flaky
corr = overDF.corr(method='pearson')
sns.heatmap(corr)
pylab.show()
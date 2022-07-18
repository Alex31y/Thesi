import pandas as pd
import seaborn as sns
import pylab
from matplotlib import pyplot as plt

pd.options.display.width = 0
df = pd.read_csv("dataset.csv")
print(len(df.columns))
list = ['id','nameProject','testCase', "Unnamed: 0", "lcom2", "mpc", "halsteadLength", "halsteadVolume"]
df = df.drop(list,axis = 1 )

# calcolo il valore medio di numCoveredLines per ogni test
print(".describe di 'numCoveredLines'")
print(df['numCoveredLines'].describe()) # la media è 9 linee di codice per test
print(".describe di 'projectSourceLinesCovered'")
print(df['projectSourceLinesCovered'].describe())
# vado a suddividere i tests del dataset in underDF e overDF
underDF = df[df['numCoveredLines'] < 9]
overDF = df[df['numCoveredLines'] >= 9]
#print(df.shape, underDF.shape, overDF.shape) #3735 case test in overDF e 6050 in underDF

#riprovo col pearson per cercare correlazioni nelle features sulle entity flaky e non flaky
corr = underDF.corr(method='pearson')
#correlation map
f,ax = plt.subplots(figsize=(34, 34))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
pylab.show()

print(len(df.columns))
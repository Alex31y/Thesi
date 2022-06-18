import pandas as pd
import seaborn as sns
import pylab

pd.options.display.width = 0
df = pd.read_csv("dataset.csv")
print(df.columns.values)

corr = df.corr(method='pearson')
sns.heatmap(corr)
pylab.show()
# primi passi nell'esplorazione del dataset: ho cercato correlazioni tra le variabili in gioco, il problema è che non ho idea di cosa sto guardando


# vado a suddividere i tests del dataset in flaky e non
notFlaky = df[df['isFlaky'] == False]
isFlaky = df[df['isFlaky'] == True]
#print(df.shape, notFlaky.shape, isFlaky.shape) # come citato nella tesi, si può notare che il numero di test case flaky è nettamente inferiore a ai test case not flaky 9115 > 670


#riprovo col pearson per cercare correlazioni nelle features sulle entity flaky e non flaky
corr = isFlaky.corr(method='pearson')
sns.heatmap(corr)
pylab.show()

corr = notFlaky.corr(method='pearson')
sns.heatmap(corr)
pylab.show()

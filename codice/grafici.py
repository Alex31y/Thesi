pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

#grafo a violino per determinare l'info gain di una feature
# fractal_dimension_mean feature, median of the Malignant and Benign does not looks like separated so it does not gives good information for classification.
"""
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,20:26]],axis=1)     #0:10 - 10:20 - 20:26
data = pd.melt(data,id_vars="numCoveredLines",
                    var_name="features",
                    value_name='value')

Violino
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="numCoveredLines", data=data,split=True, inner="quart")
plt.xticks(rotation=90)

Boxplot
plt.figure(figsize=(10,10))
sns.boxplot(x="features", y="value", hue="numCoveredLines", data=data)
plt.xticks(rotation=90)
plt.show()
"""

#divido il dataset in tre parti: prod, code, text
"""
prod = data[['tloc', 'tmcCabe', 'lcom2', 'lcom5', 'cbo', 'wmc', 'rfc', 'mpc', 'halsteadVocabulary', 'halsteadLength', 'halsteadVolume', 'ExecutionTime', 'hIndexModificationsPerCoveredLine_window5',
       'hIndexModificationsPerCoveredLine_window10',
       'hIndexModificationsPerCoveredLine_window25',
       'hIndexModificationsPerCoveredLine_window50',
       'hIndexModificationsPerCoveredLine_window75',
       'hIndexModificationsPerCoveredLine_window100',
       'hIndexModificationsPerCoveredLine_window500',
       'hIndexModificationsPerCoveredLine_window10000',
       'num_third_party_libs']]
code = data[['classDataShouldBePrivate', 'complexClass', 'functionalDecomposition', 'godClass', 'spaghettiCode']]
test = data[['assertionDensity', 'assertionRoulette', 'mysteryGuest',
       'eagerTest', 'sensitiveEquality', 'resourceOptimism',
       'conditionalTestLogic', 'fireAndForget']]
print(prod)
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

df = pd.read_csv("csvume/dataset.csv")
#df = df[df['nameProject'].str.match('Achilles')]
list = ['nameProject','testCase', "Unnamed: 0", "projectSourceLinesCovered", "numCoveredLines", "godClass", "classDataShouldBePrivate", "complexClass", "fireAndForget", "resourceOptimism"]
df = df.drop(df[df.numCoveredLines > 30].index)
y = df.numCoveredLines
df = df.drop(list,axis = 1 )

# print(df.columns)
# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)

#print(x_train.columns)

# Create and train model
rf = RandomForestRegressor(n_estimators = 100, max_depth = 30, random_state=1)
rf.fit(x_train, y_train)
# Predict on test data
prediction = rf.predict(x_test)
# Compute mean squared error
mae = mean_absolute_error(y_test, prediction)
# Print results
print(mae)

predizioni = pd.DataFrame({"id":x_test.id, "coveredLines":prediction})
predizioni.to_csv("csvume\RFR.csv", index = False)

# Output feature importance coefficients, map them to their feature name, and sort values
coef = pd.Series(rf.feature_importances_, index = x_train.columns).sort_values(ascending=True)

print(coef)
"""
plt.figure(figsize=(10, 5))
coef.head(25).plot(kind='bar')
plt.title('Feature Significance')
plt.tight_layout()
plt.show()


import lime
import lime.lime_tabular
# LIME has one explainer for all the models
explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values, feature_names=x_train.columns.values.tolist(),
                                                  class_names=['numCoveredLines'], verbose=True, mode='regression')
# Choose the 5th instance and use it to predict the results
j = 5
exp = explainer.explain_instance(x_test.values[j], rf.predict, num_features=10)
# Show the predictions
exp.save_to_file('lime.html')
print(exp.as_list())

j = 25
exp = explainer.explain_instance(x_test.values[j], rf.predict, num_features=10)
exp.save_to_file('lime2.html')
"""
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
list = ['nameProject','testCase', "Unnamed: 0", "projectSourceLinesCovered", "numCoveredLines", "lcom2", "classDataShouldBePrivate",
        "functionalDecomposition", "lcom5", "resourceOptimism", "spaghettiCode", "id"]
df = df.drop(df[df.numCoveredLines > 18].index)
y = df.numCoveredLines
df = df.drop(list,axis = 1 )

# print(df.columns)
# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)

#print(x_train.columns)

#TUNING PARAMETRI
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = [1.0, 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(x_train, y_train)
print("best params::")
print(rf_random.best_params_)
prediction = rf_random.predict(x_test)
# Compute mean squared error
mae = mean_absolute_error(y_test, prediction)
# Print results
print(mae)
#FINE TUNING


# Create and train model
rf = RandomForestRegressor(n_estimators = 100, max_depth = 30, random_state=1)
rf.fit(x_train, y_train)
# Predict on test data
prediction = rf.predict(x_test)
# Compute mean squared error
mae = mean_absolute_error(y_test, prediction)
# Print results
print(mae)

#predizioni = pd.DataFrame({"id":x_test.id, "coveredLines":prediction})
#predizioni.to_csv("csvume\RFR.csv", index = False)

# Output feature importance coefficients, map them to their feature name, and sort values
coef = pd.Series(rf.feature_importances_, index = x_train.columns).sort_values(ascending=False)

#print(coef)

plt.figure(figsize=(10, 5))
coef.head(10).plot(kind='bar')
plt.title('Feature Significance')
plt.tight_layout()
plt.show()


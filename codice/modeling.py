import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score, train_test_split


df = pd.read_csv("dataset.csv")
#df = df[df['nameProject'].str.match('logback')]
list = ['nameProject','testCase', "Unnamed: 0", "projectSourceLinesCovered"]
y = df.numCoveredLines
df = df.drop(list,axis = 1 )
# print(df.columns)
# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=40)
y_train = np.log1p(y_train)

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, x_train, y_train, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean()
            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
print(cv_ridge.min())

#RIDGE
model_ridge = Ridge(alpha = 0.05).fit(x_train, y_train)
ridge_preds = np.expm1(model_ridge.predict(x_test))
predizioni = pd.DataFrame({"id":x_test.id, "coveredLines":ridge_preds})
predizioni.to_csv("ridge.csv", index = False)

#LASSO
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(x_train, y_train)
print(rmse_cv(model_lasso).mean())
coef = pd.Series(model_lasso.coef_, index = x_train.columns)
lasso_preds = np.expm1(model_ridge.predict(x_test))
predizioni = pd.DataFrame({"id":x_test.id, "coveredLines":lasso_preds})
predizioni.to_csv("lasso.csv", index = False)

#XGB
import xgboost as xgb
dtrain = xgb.DMatrix(x_train, label = y_train)
params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(x_train, y_train)
xgb_preds = np.expm1(model_xgb.predict(x_test))
predizioni = pd.DataFrame({"id":x_test.id, "coveredLines":xgb_preds})
predizioni.to_csv("xgb.csv", index = False)
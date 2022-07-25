import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.feature_selection import RFECV

pd.options.mode.chained_assignment = None  # default='warn'
import seaborn as sns
from matplotlib import pylab

df = pd.read_csv("dataset.csv")

list = ['id','nameProject','testCase', "Unnamed: 0", "projectSourceLinesCovered"]
y = df.numCoveredLines
df = df.drop(list,axis = 1 )


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score

df2 = df[['tloc', 'halsteadLength', 'loc', 'wmc', 'godClass', 'halsteadVolume']]
# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)
y_train = np.log1p(y_train)

#random forest classifier with n_estimators=10 (default)
#clf_rf = RandomForestClassifier(n_estimators = 100, random_state=42)
#clr_rf = clf_rf.fit(x_train,y_train)

#ac = accuracy_score(y_test,clf_rf.predict(x_test))
#print('Accuracy is: ',ac)
#cm = confusion_matrix(y_test,clf_rf.predict(x_test))
#sns.heatmap(cm,annot=True,fmt="d")
#pylab.show()

ols = linear_model.LinearRegression()

rfecv = RFECV(estimator=ols, step=1, scoring="neg_mean_squared_error", cv=4, verbose=0, n_jobs=4)
rfecv.fit(x_train, y_train)
rfecv.transform(x_train)
print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train.columns[rfecv.support_])
ols = ols.fit(x_train,y_train)
ac = accuracy_score(y_test,ols.predict(x_test))
print('Accuracy is: ',ac)


"""
from sklearn.feature_selection import SelectKBest, f_regression
# find best scored 5 features
select_feature = SelectKBest(f_regression, k=8).fit(x_train, y_train)

topfeature = "nome " + x_train.columns + "val: " + select_feature.scores_.astype(str)
# print('Score list:', topfeature)
filter = select_feature.get_support()
features = df.columns
print("Selected best 8:")
print(features[filter])

x_train_2 = select_feature.transform(x_train)
x_test_2 = select_feature.transform(x_test)
#random forest classifier with n_estimators=10 (default)
clf_rf_2 = RandomForestClassifier()
clr_rf_2 = clf_rf_2.fit(x_train_2,y_train)
ac_2 = accuracy_score(y_test,clf_rf_2.predict(x_test_2))
print('Accuracy is: ',ac_2)

#Recursive feature elimination
from sklearn.feature_selection import RFE
# Create the RFE object and rank each pixel
clf_rf_3 = RandomForestClassifier()
clr_rf_3 = clf_rf_3.fit(x_train,y_train)
rfe = RFE(estimator=clf_rf_3, n_features_to_select=8, step=1)
rfe = rfe.fit(x_train, y_train)

print('Chosen best 5 feature by rfe:',x_train.columns[rfe.support_])
ac_3 = accuracy_score(y_test,rfe.predict(x_test))
print('Accuracy is: ',ac_3)


#Recursive feature elimination with cross validation
from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_4 = RandomForestClassifier()
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(x_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train.columns[rfecv.support_])

"""


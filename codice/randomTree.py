import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import seaborn as sns
from matplotlib import pylab

df = pd.read_csv("dataset.csv")

list = ['id','nameProject','testCase', "Unnamed: 0", "lcom2", "numCoveredLines", "mpc", "halsteadLength", "halsteadVolume", "projectSourceLinesCovered"]
y = df.numCoveredLines
df = df.drop(list,axis = 1 )


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score

df2 = df[['tloc', 'assertionRoulette', 'hIndexModificationsPerCoveredLine_window10000', 'hIndexModificationsPerCoveredLine_window500', 'num_third_party_libs']]
# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)

#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier(n_estimators = 100, random_state=42)
clr_rf = clf_rf.fit(x_train,y_train)

ac = accuracy_score(y_test,clf_rf.predict(x_test))
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,clf_rf.predict(x_test))
#sns.heatmap(cm,annot=True,fmt="d")
#pylab.show()

from sklearn.feature_selection import SelectKBest, f_regression
# find best scored 5 features
select_feature = SelectKBest(f_regression, k=13).fit(x_train, y_train)

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
cm_2 = confusion_matrix(y_test,clf_rf_2.predict(x_test_2))
sns.heatmap(cm_2,annot=True,fmt="d")



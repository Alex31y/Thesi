import pandas as pd
import seaborn as sns
from matplotlib import pylab

df = pd.read_csv("dataset.csv")
list = ['id','nameProject','testCase', "Unnamed: 0", "lcom2", "mpc", "halsteadLength", "halsteadVolume"]
df = df.drop(list,axis = 1 )
y = df.numCoveredLines



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score

# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)

#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier(random_state=43)
clr_rf = clf_rf.fit(x_train,y_train)

ac = accuracy_score(y_test,clf_rf.predict(x_test))
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,clf_rf.predict(x_test))
# sns.heatmap(cm,annot=True,fmt="d")
# pylab.show()

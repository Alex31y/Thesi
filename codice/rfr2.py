import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

df = pd.read_csv("csvume/dataset.csv")
df = df.drop(df[df.numCoveredLines > 18].index)

list = ['nameProject','testCase', "Unnamed: 0", "projectSourceLinesCovered", "numCoveredLines"]
y = df.numCoveredLines

df = df.drop(list,axis = 1 )


# split data train 70 % and test 30 %
for i in range(1, 30):
    cx = df.iloc[:, [0, i]]
    #print(cx.columns)

    x_train, x_test, y_train, y_test = train_test_split(cx, y, test_size=0.3, random_state=42)

    print(x_train.columns)

    # Create and train model
    rf = RandomForestRegressor(n_estimators = 100, max_depth = 30, random_state=1)
    rf.fit(x_train, y_train)
    # Predict on test data
    prediction = rf.predict(x_test)
    # Compute mean squared error
    mae = mean_absolute_error(y_test, prediction)
    # Print results
    print(mae)

"""
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)
#print(x_train.columns)

# Create and train model
rf = RandomForestRegressor(n_estimators = 100, max_depth = 30)
rf.fit(x_train, y_train)
# Predict on test data


x_train, x_test, y_train, y_test = train_test_split(altroprog, yy, test_size=0.9, random_state=42)
prediction = rf.predict(x_test)
# Compute mean squared error
mae = mean_absolute_error(y_test, prediction)
# Print results
print(mae)
#print(rmse_cv(rf).mean())
predizioni = pd.DataFrame({"id":x_test.id, "coveredLines":prediction})
predizioni.to_csv("csvume\RFR.csv", index = False)
"""

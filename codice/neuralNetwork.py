import pandas  as pd
import tensorflow as tf
import numpy   as np
from   keras.models import Sequential
from   keras.layers import Dense             # i.e.fully connected
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

#===========================================================================
# read in the data from your local directory
#===========================================================================
from sklearn.model_selection import train_test_split

df = pd.read_csv("csvume/dataset.csv")
#df = df[df['nameProject'].str.match('logback')]
list = ['nameProject','testCase', "Unnamed: 0", "projectSourceLinesCovered", "numCoveredLines", "isFlaky"]
y = df.numCoveredLines
df = df.drop(list,axis = 1 )
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)
#x_train = np.asarray(x_train).astype(np.float32)
#x_test = np.asarray(x_test).astype(np.float32)


print(x_test.head(5))

#===========================================================================
# parameters for keras
#===========================================================================
input_dim        = x_train.shape[1] # number of neurons in the input layer
n_neurons        =  25       # number of neurons in the first hidden layer
epochs           = 150       # number of training cycles

#===========================================================================
# keras model
#===========================================================================
model = Sequential()        # a model consisting of successive layers
# input layer
model.add(Dense(n_neurons, input_dim=input_dim,
                kernel_initializer='normal',
                activation='relu'))
# output layer, with one neuron
model.add(Dense(1, kernel_initializer='normal'))
# compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

#===========================================================================
# train the model
#===========================================================================
model.fit(x_train, y_train, epochs=epochs, verbose=0)

#===========================================================================
# use the model to predict the prices for the test data
#===========================================================================
predictions = model.predict(x_test)

#===========================================================================
# write out CSV submission file
#===========================================================================

predizioni = pd.DataFrame({"id":x_test.id, "coveredLines":predictions.flatten()})
predizioni.to_csv("nn.csv", index = False)

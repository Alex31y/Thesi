import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import time

pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)


data = pd.read_csv("dataset.csv")
print(data.describe)
print(data.nameProject.unique())

# partiziono il dataset per progetto
logback_cols = data[data['nameProject'].str.match('logback')]
orbit = data[data['nameProject'].str.match('orbit')]
achille = data[data['nameProject'].str.match('Achilles')]
httprequest = data[data['nameProject'].str.match('http-request')]
hector = data[data['nameProject'].str.match('hector')]
okhttp = data[data['nameProject'].str.match('okhttp')]
ninja = data[data['nameProject'].str.match('ninja')]
elastic = data[data['nameProject'].str.match('elastic-job-lite')]
undertow = data[data['nameProject'].str.match('undertow')]
Activiti = data[data['nameProject'].str.match('Activiti')]
ambari = data[data['nameProject'].str.match('ambari')]
incubator = data[data['nameProject'].str.match('incubator-dubbo')]
hbase = data[data['nameProject'].str.match('hbase')]
httpcore = data[data['nameProject'].str.match('httpcore')]
Java = data[data['nameProject'].str.match('Java')]
spring = data[data['nameProject'].str.match('spring')]
wro4j = data[data['nameProject'].str.match('wro4j')]
alluxio = data[data['nameProject'].str.match('alluxio')]
print(spring.describe())





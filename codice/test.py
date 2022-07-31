import pandas as pd
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)


df = pd.read_csv("dataset.csv")
sol = pd.read_csv("ridge.csv")
row = df.loc[df['id'] == 10190]


scarto = 0
for row in sol.itertuples():
    predizione = row.coveredLines
    reale= df.loc[df['id'] == row.id].numCoveredLines.item()
    scarto = scarto + abs(reale - predizione)

print(scarto/len(sol.index))

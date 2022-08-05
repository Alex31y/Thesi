import pandas as pd
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)


df = pd.read_csv("csvume/dataset.csv")
sol = pd.read_csv("csvume/RFR.csv")
row = df.loc[df['id'] == 8207]
print(row.numCoveredLines)

scarto = 0
result = pd.DataFrame()

for row in sol.itertuples():
    predizione = row.coveredLines
    #reale= df.loc[df['id'] == row.id].numCoveredLines.item()
    reale = df.loc[df['id'] == row.id].numCoveredLines.tolist()
    #if(reale[0] > 25):      #skippando i valori maggiori di 20 l'accuratezza sale
       #continue
    scarto = scarto + abs(reale[0] - predizione)
    predizioni = pd.DataFrame({"id": row.id, "predizione": predizione, "effettivo": reale})
    result = result.append(predizioni, ignore_index = True)

result.to_csv("result.csv", index=False)
print(scarto/len(sol.index))

"""
Analyses benchmark files and compares them with one another
"""
import pandas as pd
import matplotlib.pyplot as plt
import sys

df1 = pd.read_csv(sys.argv[1]) #old
df2 = pd.read_csv(sys.argv[2]) #new
old = df1.columns[0]
new = df2.columns[0]
joined = df1.join(df2).sort_values(by=old).reset_index(drop=True)
#for idx, row in df1.join(df2).sort_values(by=old).iterrows():
#    print(idx.replace(";",",").replace("-"," -"), row[new]/row[old])
#sp = pd.DataFrame(joined['new']/joined['old'])
ax = plt.gca()
joined.reset_index().plot(kind='scatter', x='index', y=old, ax=ax, label = old, color='r')
joined.reset_index().plot(kind='scatter', x='index', y=new, ax=ax, label = new, color='g')
ax.legend(loc="upper left")
plt.savefig(f"{sys.argv[3]}", bbox_inches='tight', transparent=False)
plt.close()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

df = pd.read_csv("crosskernel/benchmark2.csv")
df.head()

x = np.asarray(df[['node', 'density', 'epoch']], dtype=np.float64)
y = np.asarray(df[['dgl']], dtype=np.float64)
z = np.asarray(df[['pyg']], dtype=np.float64)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

regr = ElasticNet()
regr.fit(x_train, y_train)

print(regr.score(x_test, y_test))

x_train, x_test, z_train, z_test = train_test_split(x, z, train_size=0.8, test_size=0.2)

regr = ElasticNet()
regr.fit(x_train, z_train)

print(regr.score(x_test, z_test))
import pandas as pd
import numpy as np
import pickle

df=pd.read_csv('NIRF-data.csv')

x=df[['TLR','RP','GO','OI','PR']]
y=df['RANK']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=10)

from sklearn.ensemble import RandomForestRegressor
rg=RandomForestRegressor()

rg.fit(x_train,y_train)

rg.score(x_test,y_test)


pickle.dump(rg, open('rank.pkl','wb'))
model=pickle.load(open('rank.pkl','rb'))


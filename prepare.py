import pandas as pd

vgsales=pd.read_csv('vgsales.csv')

vgsales['Nintendo'] = vgsales['Publisher'].apply(lambda x: 1 if x=='Nintendo' else 0)

Y = vgsales['Nintendo']
X = vgsales.drop(['Rank','Name','Platform','Year','Genre','Publisher','Nintendo'],axis = 1)


X.to_csv(r'10_x.csv', index=False)
Y.to_csv(r'10_y.csv', index=False)
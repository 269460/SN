import numpy as np #do roznych obliczen
import pandas as pd #do operacji na danych
import matplotlib.pyplot as plt #do wykresow
import seaborn as sns #do jeszcze ladniejszych wykresow
#%matplotlib inline
pd.set_option('display.max_columns', 500)
# Load the data, we set that index_col is the first column, therefore there will be standard index start from 0 for each data.
train_df = pd.read_csv('input/train.csv', header=0,index_col=0)
test_df = pd.read_csv('input/test.csv', header=0,index_col=0)
full = pd.concat([train_df , test_df]) # concatenate two dataframes
full.info()   # info about dataframe

# DATA SECTION
full['_Sex'] = pd.Categorical(full.Sex).codes
full['_Embarked'] = pd.Categorical(full.Embarked).codes
full['_CabinType'] = pd.Categorical(full['Cabin'].astype(str).str[0]).codes

pat = r",\s([^ .]+)\.?\s+"
full['Title'] =  full['Name'].str.extract(pat,expand=True)[0]
full.loc[full['Title'].isin(['Mlle','Ms','Lady']),'Title'] = 'Miss'
full.loc[full['Title'].isin(['Mme']),'Title'] = 'Mrs'
full.loc[full['Title'].isin(['Sir']),'Title'] = 'Mr'
full.loc[~full['Title'].isin(['Miss','Master','Mr','Mrs']),'Title'] = 'Other' # NOT IN
full['_Title'] = pd.Categorical(full.Title).codes

full['TicketCounts'] = full.groupby(['Ticket'])['Ticket'].transform('count')

# FILL N/A Values
full.at[1044,'Fare'] =  (7.25 + 6.2375)/2; # we set average for this values

## NaN: Age
# fill using LinearRegression
# full[['Age','_Embarked','Fare','Parch','Pclass','_Sex','SibSp','Survived','_CabinType','_Title']].corr()['Age'].abs().sort_values(ascending=False)
cols = ['Pclass','_CabinType','SibSp','Fare','Parch']

ageData = full[full['Age'].notnull()]
emptyData = full[full['Age'].isnull()]
Y = ageData['Age']
X = ageData[cols] # ,'Parch','Embarked']]

# Create linear regression object
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X,Y)

X = emptyData[cols]
Y = regr.predict(X)
# First we need to set index before

pred =  pd.concat([pd.Series(Y,emptyData.index),ageData['Age']]).sort_index()
full['_AgeLinear'] = pred

from sklearn.impute import SimpleImputer

# Utworzenie obiektu SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

# Użyj `fit_transform` do wypełnienia brakujących wartości w kolumnie 'Age'
full['_AgeImputed'] = imp.fit_transform(full[['Age']])

# Process _Fare
from sklearn import preprocessing
full['_Fare'] = preprocessing.scale(full[['Fare']]) [:,0]

# Calculate AgeCategory
full['AgeCategory'] = pd.cut(full['_AgeImputed'], [0, 9, 18, 30, 40, 50, 100], labels=[9, 18, 30, 40, 50, 100])
full['_AgeCategory'] = full['AgeCategory'].cat.codes

# Select columns
cols = ['Parch', 'Pclass', 'SibSp', '_Sex', '_Embarked', '_CabinType', '_Title', 'TicketCounts', '_AgeCategory', '_Fare']

SURV = 891
X = full[:SURV][cols]
Y = full[:SURV]['Survived']

# Classifier
# conda install py-xgboost
from xgboost import XGBClassifier
clf = XGBClassifier()

clf.fit(X,Y)
clf.score(X,Y) #0.8843995510662177

Xp = full[SURV:][cols]
result = pd.DataFrame({'PassengerID': full[SURV:].index })
result['Survived'] = clf.predict(Xp).T.astype(int)

result[['PassengerID','Survived']].to_csv('submission.csv',index=False)
print('hurra, we find the result.')

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier,XGBRFClassifier
from lazypredict.Supervised import  LazyClassifier
from imblearn.over_sampling import SMOTE


data=pd.read_csv('survey.csv')

print(data.columns)
print(data.info())
print(data.isna().sum())
print(data.describe())

for i in data.select_dtypes(include='object').columns.values:
    print(data[i].value_counts())

a=['work_interfere', 'no_employees', 'benefits', 'care_options',
       'wellness_program', 'anonymity', 'mental_health_interview',
       'mental_vs_physical']
print('------xxx-----------')
for i in a:
    print(data[i].value_counts().index)
print('---------xxx------------')

lab=LabelEncoder()
for i in data.select_dtypes(include='object').columns.values:
    if len(data[i].value_counts().values)<20:
        data[i]=lab.fit_transform(data[i])

data=data.drop(['Timestamp','Gender', 'Country', 'state'],axis=1)
print(data.info())
print('------------------------')
ib=["mental_health_issue"]
pred={}
for i in ib:
    if i=="mental_health_issue":
        corr=data.corr()[i]
        corr=corr.drop([i])
        print(corr)
        x=[]
        for i in corr.index:
            if corr[i] > 0:
                x.append(i)
        print(x)
        print('xxxxxxxxxxxxxxxxxxxxxxxxx')
        x=data[x]
        y=data[i]
        print(x.columns)

        smote=SMOTE()
        x,y=smote.fit_resample(x,y)
        x_train,x_test,y_train,y_test=train_test_split(x,y)
        print(y_test.values[0])
        models={"Random forest":RandomForestClassifier(),
               "LGBM":LGBMClassifier(),
               "XGB classifier":XGBClassifier()}
        for model_name,model in models.items():
            model.fit(x_train,y_train)
            prediction=model.predict([x_test.values[0]])
            print(model.score(x_test,y_test))
            pred[model_name]=prediction

    else:
        break


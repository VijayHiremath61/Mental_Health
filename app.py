import random

import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from lazypredict.Supervised import LazyClassifier
from imblearn.over_sampling import SMOTE
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import plotly.graph_objs as go

# Create the Streamlit app
st.title('Mental Health Survey Analysis')
data = pd.read_csv('survey.csv')

print(data.columns)
print(data.info())
print(data.isna().sum())
print(data.describe())

st.sidebar.header('Give me inputs here')

# Sidebar inputs
work_interfere = st.sidebar.selectbox('Work Interference', ['Sometimes', 'Never', 'Rarely', 'Often'])
no_employees = st.sidebar.selectbox('Number of Employees', ['Jun-25', '26-100', 'More than 1000', '100-500', '01-May', '500-1000'])
benefits = st.sidebar.selectbox('Benefits', ['Yes', "Don't know", 'No'])
care_options = st.sidebar.selectbox('Care Options', ['No', 'Yes', "Don't know"])
wellness_program = st.sidebar.selectbox('Wellness Program', ['No', 'Yes', "Don't know"])
anonymity = st.sidebar.selectbox('Anonymity', ["Don't know", 'Yes', 'No'])
mental_health_interview = st.sidebar.selectbox('Mental Health Interview', ["Don't know", 'Yes', 'No'])
mental_vs_physical = st.sidebar.selectbox('Mental vs Physical', ["Don't know", 'Yes', 'No'])

# Label encode the selected inputs
label_encoder = LabelEncoder()

work_interfere_encoded = label_encoder.fit_transform([work_interfere])
no_employees_encoded = label_encoder.fit_transform([no_employees])
benefits_encoded = label_encoder.fit_transform([benefits])
care_options_encoded = label_encoder.fit_transform([care_options])
wellness_program_encoded = label_encoder.fit_transform([wellness_program])
anonymity_encoded = label_encoder.fit_transform([anonymity])
mental_health_interview_encoded = label_encoder.fit_transform([mental_health_interview])
mental_vs_physical_encoded = label_encoder.fit_transform([mental_vs_physical])

# Convert the encoded inputs to a numpy array
user_inputs = np.array([work_interfere_encoded[0], no_employees_encoded[0], benefits_encoded[0],
                        care_options_encoded[0], wellness_program_encoded[0], anonymity_encoded[0],
                        mental_health_interview_encoded[0], mental_vs_physical_encoded[0]])

# Display selected options

lab = LabelEncoder()
for i in data.select_dtypes(include='object').columns.values:
    if len(data[i].value_counts().values) < 20:
        data[i] = lab.fit_transform(data[i])
data = data.drop(['Timestamp', 'Gender', 'Country', 'state'], axis=1)
print(data.info())

ib = ["mental_health_issue"]
score = []
pred = {}
for i in ib:
    if i == "mental_health_issue":
        corr = data.corr()[i]
        corr = corr.drop([i])
        print(corr)
        x = []
        for i in corr.index:
            if corr[i] > 0:
                x.append(i)
        print(x)
        print('xxxxxxxxxxxxxxxxxxxxxxxxx')
        x = data[x]
        y = data[i]
        print(x.columns)

        smote = SMOTE()
        x, y = smote.fit_resample(x, y)
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        print(y_test.values[0])
        models = {"Random forest": RandomForestClassifier(),
                  "LGBM": LGBMClassifier(),
                  "XGB classifier": XGBClassifier()}
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            prediction = model.predict([user_inputs])
            score.append(model.score(x_test, y_test))
            pred[model_name] = prediction

    else:
        break

output = [j for i, j in pred.items()]
final = []
for i in range(3):
    final.append(output[i][0])
name = [i for i, j in pred.items()]
print(name)
fin=[]
for i in final:
    if i==0:
        fin.append("Its anxiety")
    else:
        fin.append("its depression")

# Display the output in a table
table_data = pd.DataFrame({'Model': name, 'Accuracy Score': score,"output":fin})
st.table(table_data)

anxity_reco=['take deep breadth', "dont worry talk to your closed ones","set Boundaries","Do some physical activity"]
depression=["limit stress"," educate your self","meditate","do some Physical activity","Reach out to your closed ones"]
if fin[0]=="Its anxiety":
    st.write(random.choice(anxity_reco))
elif fin[0]=="its depression":
    st.write(random.choice(depression))

# Plot the accuracy scores
fig = px.bar(x=name, y=score, labels={'x': 'Model', 'y': 'Accuracy Score'}, title='Model Accuracy Scores')
st.plotly_chart(fig)

# Add another page for data visualizations
st.sidebar.title('Data Visualizations')

if st.sidebar.checkbox('Show boxplot'):
    st.subheader('Histogram')
    selected_column = st.selectbox('Select a column', data.columns)
    plt.boxplot(data[selected_column])
    st.pyplot()

if st.sidebar.checkbox('show AUTO-ML output'):
    st.subheader("AutoML output")
    data = pd.read_csv('survey.csv')
    print(data.columns)
    print(data.info())
    print(data.isna().sum())
    print(data.describe())

    for i in data.select_dtypes(include='object').columns.values:
        print(data[i].value_counts())

    a = ['work_interfere', 'no_employees', 'benefits', 'care_options',
         'wellness_program', 'anonymity', 'mental_health_interview',
         'mental_vs_physical']
    print('------xxx-----------')
    for i in a:
        print(data[i].value_counts().index)
    print('---------xxx------------')

    lab = LabelEncoder()
    for i in data.select_dtypes(include='object').columns.values:
        if len(data[i].value_counts().values) < 20:
            data[i] = lab.fit_transform(data[i])

    data = data.drop(['Timestamp', 'Gender', 'Country', 'state'], axis=1)
    print(data.info())
    print('------------------------')
    ib = ["mental_health_issue"]
    pred = {}
    for i in ib:
        if i == "mental_health_issue":
            corr = data.corr()[i]
            corr = corr.drop([i])
            print(corr)
            x = []
            for i in corr.index:
                if corr[i] > 0:
                    x.append(i)
            print(x)
            print('xxxxxxxxxxxxxxxxxxxxxxxxx')
            x = data[x]
            y = data[i]
            print(x.columns)

            smote = SMOTE()
            x, y = smote.fit_resample(x, y)
            x_train, x_test, y_train, y_test = train_test_split(x, y)
            print(y_test.values[0])
            lazy=LazyClassifier()
            models,prediction=lazy.fit(x_train,x_test,y_train,y_test)
            st.write(models)
        else:
            break
if st.sidebar.checkbox("Show the classification report"):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier, XGBRFClassifier
    from lazypredict.Supervised import LazyClassifier
    from imblearn.over_sampling import SMOTE

    data = pd.read_csv('survey.csv')

    print(data.columns)
    print(data.info())
    print(data.isna().sum())
    print(data.describe())

    for i in data.select_dtypes(include='object').columns.values:
        print(data[i].value_counts())

    a = ['work_interfere', 'no_employees', 'benefits', 'care_options',
         'wellness_program', 'anonymity', 'mental_health_interview',
         'mental_vs_physical']
    print('------xxx-----------')
    for i in a:
        print(data[i].value_counts().index)
    print('---------xxx------------')

    lab = LabelEncoder()
    for i in data.select_dtypes(include='object').columns.values:
        if len(data[i].value_counts().values) < 20:
            data[i] = lab.fit_transform(data[i])

    data = data.drop(['Timestamp', 'Gender', 'Country', 'state'], axis=1)
    print(data.info())
    print('------------------------')
    ib = ["mental_health_issue"]
    pred = {}
    for i in ib:
        if i == "mental_health_issue":
            corr = data.corr()[i]
            corr = corr.drop([i])
            print(corr)
            x = []
            for i in corr.index:
                if corr[i] > 0:
                    x.append(i)
            print(x)
            print('xxxxxxxxxxxxxxxxxxxxxxxxx')
            x = data[x]
            y = data[i]
            print(x.columns)

            smote = SMOTE()
            x, y = smote.fit_resample(x, y)
            x_train, x_test, y_train, y_test = train_test_split(x, y)
            print(y_test.values[0])
            models = {"Random forest": RandomForestClassifier(),
                      "LGBM": LGBMClassifier(),
                      "XGB classifier": XGBClassifier()}
            for model_name, model in models.items():
                model.fit(x_train, y_train)
                prediction = model.predict(x_test)
                print(model.score(x_test, y_test))
                pred[model_name] = prediction
                report=classification_report(y_test,prediction,output_dict=True)
                df_report = pd.DataFrame(report).T
                st.write(f"The model {model_name} classification report is ")
                st.table(df_report)

        else:
            break

if st.sidebar.checkbox('Show Heatmap'):
    st.subheader('Heatmap')
    fig_heatmap = px.imshow(data.corr(), color_continuous_scale='Viridis', title='Heatmap')
    st.plotly_chart(fig_heatmap)
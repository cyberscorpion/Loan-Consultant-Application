from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor

def model2(x2):
    df = pd.read_csv("media/data/Loan_Prediction_train.csv")
    df['Gender'].fillna(value='Male', inplace=True)
    df['Dependents'].fillna(value=0, inplace=True)
    df['Married'].fillna(value='Yes', inplace=True)
    df['Self_Employed'].fillna(value='No', inplace=True)
    df['Loan_Amount_Term'].fillna(value=360.0, inplace=True)
    df['Credit_History'].fillna(value=1, inplace=True)
    df['LoanAmount'].fillna(value=df['LoanAmount'].mean(), inplace=True)
    # create a list of features to dummy
    dummy_list = ['Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Loan_Status']

    # function to dummy all categorical variable used for modelling
    df2 = get_dummy(df, dummy_list)
    dummies = pd.get_dummies(df2['Property_Area'], prefix='Property_Area')
    df2 = df2.drop('Property_Area', 1)
    df2 = pd.concat([df2, dummies], axis=1)
    dummies = pd.get_dummies(df2['Dependents'], prefix='Dependents', drop_first=True)
    df2 = df2.drop('Dependents', 1)
    df2 = pd.concat([df2, dummies], axis=1)
    df2.head()
    df2.drop('Loan_ID', axis=1, inplace=True)
    x1 = df2.iloc[:, 4:5].values
    y1 = df2.iloc[:, 6].values
    train_x,test_x,train_y,test_y=train_test_split(x1,y1,test_size=0.25,random_state=0)
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(train_x, train_y)
    return regressor.predict(x2)



def get_dummy(df, dummy_list):
    for x in dummy_list:
        dummies = pd.get_dummies(df[x], drop_first=True)

        df[x] = dummies
        # df=pd.concat([df,dummies],axis=1)
    return df

def model(x1):
    df = pd.read_csv("media/data/Loan_Prediction_train.csv")
    df['Gender'].fillna(value='Male', inplace=True)
    df['Dependents'].fillna(value=0, inplace=True)
    df['Married'].fillna(value='Yes', inplace=True)
    df['Self_Employed'].fillna(value='No', inplace=True)
    df['Loan_Amount_Term'].fillna(value=360.0, inplace=True)
    df['Credit_History'].fillna(value=1, inplace=True)
    df['LoanAmount'].fillna(value=df['LoanAmount'].mean(), inplace=True)
    # create a list of features to dummy
    dummy_list = ['Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Loan_Status']

    # function to dummy all categorical variable used for modelling
    df2 = get_dummy(df, dummy_list)
    dummies = pd.get_dummies(df2['Property_Area'], prefix='Property_Area')
    df2 = df2.drop('Property_Area', 1)
    df2 = pd.concat([df2, dummies], axis=1)
    dummies = pd.get_dummies(df2['Dependents'], prefix='Dependents', drop_first=True)
    df2 = df2.drop('Dependents', 1)
    df2 = pd.concat([df2, dummies], axis=1)
    df2.head()
    df2.drop('Loan_ID', axis=1, inplace=True)
    x = df2.iloc[:, 0:9]
    y = df2.iloc[:, 9]

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=0)

    # feature scaling

    sc = StandardScaler()
    train_x = sc.fit_transform(train_x)
    test_x = sc.transform(test_x)


    classifier = LogisticRegression()
    classifier.fit(train_x, train_y)
    y=classifier.predict(x1)

    return y


# Create your views here.
def index(request):

    return render(request,'loanUI/index.html',{})

def detail(request):
    choice=request.POST['choice']
    if choice=="1":

        return render(request, 'loanUI/form1.html', {})
    else:

        return render(request, 'loanUI/form2.html', {})
def predict1(request):

    text1=request.POST['text1']
    text2=request.POST['text2']
    text3=request.POST['text3']
    text4=request.POST['text4']
    text8=request.POST['text8']
    edu=request.POST['edu']
    text1=float(text1)
    if edu=='Yes':
        edu=1
    else:
        edu=0

    marital=request.POST['marital']
    if marital=='Yes':
        marital=1
    else:
        marital=0
    gender=request.POST['gender']
    if gender=='female':
        gender=1
    else:
        gender=0
    self=request.POST['self']
    if self=='Yes':
        self=1
    else:
        self=0
    x1 = np.array([int(gender), int(marital), int(edu), int(self), float(text2), float(text3), float(text1),int(text8),int(text4)])
    x1 = x1.reshape(1, -1)

    #y = model([[0,1,1,1,1000,2000,400,360,1]])
    y=model(x1)
    if y==1:
        y='Eligible!'
    else:
        y='Not Eligible!'
    return render(request, 'loanUI/result.html', {'y':y})


    #return HttpResponse("<h1>hello</h1")
def predict2(request):
    text2 = request.POST['text2']
    income=np.array([float(text2)])
    income = income.reshape(1, -1)
    y=model2(income)
    return render(request, 'loanUI/result.html', {'y': y})
    #return HttpResponse("<h1>hello</h1")



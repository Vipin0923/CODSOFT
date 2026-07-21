import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

# Load dataset
df = pd.read_csv("train.csv")

features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
X = df[features]
y = df['Survived']

numeric = ['Pclass','Age','SibSp','Parch','Fare']
categorical = ['Sex','Embarked']

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ]), numeric),

    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical)
])

model = Pipeline([
    ('prep', preprocessor),
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

st.header("Passenger Information")

pclass = st.selectbox("Passenger Class", [1,2,3])
sex = st.selectbox("Gender", ["male","female"])
age = st.slider("Age",1,80,25)
sibsp = st.number_input("Siblings/Spouse",0,10,0)
parch = st.number_input("Parents/Children",0,10,0)
fare = st.number_input("Fare",0.0,600.0,50.0)
embarked = st.selectbox("Embarked",["S","C","Q"])

if st.button("Predict"):

    input_df = pd.DataFrame({
        "Pclass":[pclass],
        "Sex":[sex],
        "Age":[age],
        "SibSp":[sibsp],
        "Parch":[parch],
        "Fare":[fare],
        "Embarked":[embarked]
    })

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("✅ Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
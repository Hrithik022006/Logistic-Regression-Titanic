
import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('logistic_model.joblib')

st.title("🚢 Titanic Survival Prediction")

# User Inputs
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.radio("Sex", ['male', 'female'])
age = st.slider("Age", 0, 100, 25)
sibsp = st.slider("Number of Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
parch = st.slider("Number of Parents/Children Aboard (Parch)", 0, 6, 0)
fare = st.number_input("Fare Paid", min_value=0.0, max_value=600.0, value=50.0)
embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

# Encoding categorical inputs
sex = 0 if sex == 'male' else 1
embarked = {'C': 0, 'Q': 1, 'S': 2}[embarked]

# Predict on click
if st.button("Predict"):
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    prediction = model.predict(input_data)[0]
    result = "🟢 Survived" if prediction == 1 else "🔴 Did Not Survive"
    st.success(f"Prediction: {result}")

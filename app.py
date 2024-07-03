import streamlit as st
import requests
import pandas as pd

st.title("Application de Machine Learning")

# Section for training the model
st.header("Entraîner un modèle")
uploaded_file = st.file_uploader("Choisissez un fichier CSV pour l'entraînement", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write(data)
    if st.button("Entraîner"):
        response = requests.post("http://127.0.0.1:8000/training", json={
            "data": data.drop('target', axis=1).values.tolist(),
            "target": data['target'].tolist()
        })
        st.write(response.json())

# Section for making a prediction
st.header("Faire une prédiction")
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=0)
    sex = st.selectbox("Sex", [0, 1])
    chest_pain_type = st.selectbox("Type de douleur thoracique", [0, 1, 2, 3])
    cholesterol = st.number_input("Cholestérol", min_value=0)
    fasting_blood_sugar = st.selectbox("Glycémie à jeun > 120 mg/dl", [0, 1])
    rest_ecg = st.selectbox("Résultats de l'ECG au repos", [0, 1, 2])
    max_heart_rate_achieved = st.number_input("Fréquence cardiaque maximale atteinte", min_value=0)
    exercise_induced_angina = st.selectbox("Angine induite par l'exercice", [0, 1])
    st_depression = st.number_input("Dépression ST", min_value=0.0)
    st_slope_flat = st.selectbox("Pente ST Plate", [0, 1])
    st_slope_upsloping = st.selectbox("Pente ST Montante", [0, 1])

    submitted = st.form_submit_button("Prédire")
    if submitted:
        user_input = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'chest_pain_type': [chest_pain_type],
            'cholesterol': [cholesterol],
            'fasting_blood_sugar': [fasting_blood_sugar],
            'rest_ecg': [rest_ecg],
            'max_heart_rate_achieved': [max_heart_rate_achieved],
            'exercise_induced_angina': [exercise_induced_angina],
            'st_depression': [st_depression],
            'st_slope_flat': [st_slope_flat],
            'st_slope_upsloping': [st_slope_upsloping]
        })

        expected_columns = ['age', 'sex', 'chest_pain_type', 'cholesterol',
                            'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
                            'exercise_induced_angina', 'st_depression', 'st_slope_flat',
                            'st_slope_upsloping']

        # Ensure the columns are in the correct order
        user_input = user_input[expected_columns]
        
        response = requests.post("http://127.0.0.1:8000/predict", json={
            "data": user_input.values.tolist()
        })
        st.write(response.json())


st.title("GPT-3 Chatbot with FastAPI and Streamlit")

prompt = st.text_area("Enter your prompt:", "")

if st.button("Get Response"):
    if prompt.strip():
        response = requests.post(
            "http://127.0.0.1:8000/gpt-response",
            json={"message": prompt}
        )
        if response.status_code == 200:
            print("sifou ",response)
            gpt_response = response.json().get("response", "")
            st.text_area("GPT-3 Response:", gpt_response, height=200)
        else:
            st.error("Error: " + response.json().get("detail", "Unknown error"))
    else:
        st.warning("Please enter a prompt.")

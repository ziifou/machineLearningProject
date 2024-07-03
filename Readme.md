Machine Learning API with FastAPI and Streamlit
===============================================

This project provides a Machine Learning API using FastAPI for training and predicting with a logistic regression model, and a Streamlit web application for interacting with the API, including a GPT-3 chatbot.

Table of Contents
-----------------

-   [Features](#features)
-   [Requirements](#requirements)
-   [Installation](#installation)
-   [Usage](#usage)
-   [Endpoints](#endpoints)
-   [Streamlit App](#streamlit-app)
-   [GPT-3 Integration](#gpt-3-integration)
-   [License](#license)

Features
--------

-   **Model Training**: Train a logistic regression model using a CSV file.
-   **Prediction**: Make predictions based on user input.
-   **GPT-3 Chatbot**: Interact with a GPT-3 chatbot.

Requirements
------------

-   Python 3.7+
-   FastAPI
-   joblib
-   pandas
-   scikit-learn
-   openai
-   streamlit
-   requests

Installation
------------

1.  **Clone the repository**:

    sh

    Copier le code

    `git clone https://github.com/ziifou/machineLearningProject
    cd ml-api-fastapi-streamlit`

2.  **Create a virtual environment**:

    sh

    Copier le code

    `python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate``

3.  **Install the dependencies**:

    sh

    Copier le code

    `pip install -r requirements.txt`

4.  **Set up OpenAI API key**: Replace `'YOUR_API_KEY'` with your actual OpenAI API key in the FastAPI app.

Usage
-----

### Start the FastAPI server

Run the FastAPI server:

sh

Copier le code

`uvicorn main:app --reload`

### Run the Streamlit app

In a new terminal, run:

sh

Copier le code

`streamlit run app.py`

Endpoints
---------

### Model Training

-   **URL**: `/training`
-   **Method**: `POST`
-   **Description**: Trains a logistic regression model with the provided data.
-   **Request Body**:

    json

    Copier le code

    `{
        "data": [[...], [...], ...],
        "target": [...]
    }`

-   **Response**:

    json

    Copier le code

    `{
        "message": "Modèle entraîné avec succès"
    }`

### Prediction

-   **URL**: `/predict`
-   **Method**: `POST`
-   **Description**: Predicts the class for the provided data.
-   **Request Body**:

    json

    Copier le code

    `{
        "data": [[...]]
    }`

-   **Response**:

    json

    Copier le code

    `{
        "predictions": ["Malade", "Sain", ...]
    }`

### GPT-3 Response

-   **URL**: `/gpt-response`
-   **Method**: `POST`
-   **Description**: Get a response from GPT-3.5-turbo for the provided message.
-   **Request Body**:

    json

    Copier le code

    `{
        "message": "Your prompt here"
    }`

-   **Response**:

    json

    Copier le code

    `{
        "response": "GPT-3 response here"
    }`

Streamlit App
-------------

The Streamlit app provides a user-friendly interface for interacting with the FastAPI endpoints.

### Model Training

-   Upload a CSV file containing the training data.
-   Click "Entraîner" to train the model.

### Prediction

-   Fill in the form with the required features.
-   Click "Prédire" to get the prediction result.

### GPT-3 Chatbot

-   Enter a prompt in the text area.
-   Click "Get Response" to receive a response from GPT-3.

GPT-3 Integration
-----------------

Ensure you have an OpenAI API key. Replace `'YOUR_API_KEY'` in the FastAPI code with your actual API key. The GPT-3 endpoint in the FastAPI app uses this key to interact with OpenAI's API.
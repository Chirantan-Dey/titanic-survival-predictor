import streamlit as st
import pandas as pd
import pickle



titanic_model = pickle.load(open('titanic_model.pkl', 'rb'))

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🛳️", layout="centered")
# Create a title for the app
st.title("Titanic: Survival Prediction")



st.subheader("Select Passenger Information")

# Passenger Class
class_options = ['1', '2', '3']
selected_class = st.selectbox('Select Passenger Class', class_options)

# Age
age = st.slider('Select Passenger Age', min_value=0, max_value=100, value=25)

# Gender
gender_options = ['Male', 'Female']
selected_gender = st.selectbox('Select Passenger Gender', gender_options)
gender_map = {'Male': 0, 'Female': 1}



if st.button("Predict Survival"):
    # Create a dataframe with the user input
    input_data = [[int(selected_class), age, gender_map[selected_gender]]]
    # Make a prediction using the trained model
    prediction = titanic_model.predict(input_data)

    # Display the prediction
    if prediction == 1:
        st.success("The passenger is likely to survive.✅", icon="✅")
        st.balloons()
    else:
        st.error("The passenger is unlikely to survive.😢", icon="😢")

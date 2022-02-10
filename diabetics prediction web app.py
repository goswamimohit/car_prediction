import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('E:/Mohit_GitHub/car_prediction/trained_model.sav', 'rb'))

# creating a function

def diabetic_prediction(input_data):
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'




def main():

    #giving a title
    st.title('Diabetic Prediction Web App')

    # getting input data from csv
    
    Pregnancies = st.text_input('Number of Pregnancies:')
    Glucose = st.text_input('Glucose level:')
    BloodPressure = st.text_input('BloodPressure Level:')
    SkinThickness = st.text_input('Skin Thickness Value:')
    Insulin = st.text_input('Insulin Level:')
    BMI = st.text_input('BMI Value:')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value:')
    Age = st.text_input('Age of the Person:')



    #code for prediction
    daignosis = ''

    # creating a button for result

    if st.button('Diabetics Test Result'):
        daignosis = diabetic_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(daignosis)




if __name__ == "__main__":
    main()



















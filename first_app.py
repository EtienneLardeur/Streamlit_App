import streamlit as st
import pandas as pd
import os

DATA_URL = 'https://raw.githubusercontent.com/EtienneLardeur/Streamlit_App/main/'
DATA_PATH = os.path.join(DATA_URL, 'tiny.csv')

st.write("""
# Client application : credit scoring
""")
st.subheader('Overview - edition mode')
tiny = pd.read_csv(DATA_PATH)

st.write(tiny)

Client_ID_list=tiny['SK_ID_CURR'].tolist()

st.sidebar.header('User Input Values')



def user_input_features():


    Client_ID = st.sidebar.selectbox('Please select Client ID', Client_ID_list, 0)

    	##st.sidebar.add_rows

    data = {'Client_ID': Client_ID}
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Selected Client ID')

st.write(df)


# Display prediction
def predict(df):

    Risk_Flag = tiny['RISK_FLAG'][tiny['SK_ID_CURR'] == df['Client_ID']]
    return Risk_Flag

risk=predict(df)

st.subheader('Risk Prediction')

st.write(risk)


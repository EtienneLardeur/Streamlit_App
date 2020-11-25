import streamlit as st
import pandas as pd
import os

REMOTE_URL = 'https://raw.githubusercontent.com/EtienneLardeur/Streamlit_App/main/'
DATA_FILE_PATH = os.path.join(REMOTE_URL, 'tiny.csv')
DESC_FILE_PATH = os.path.join(REMOTE_URL, 'desc.csv')
# MDL_FILE_PATH = os.path.join(REMOTE_URL, 'model.sav')

tiny = pd.read_csv(DATA_FILE_PATH)
desc = pd.read_csv(DESC_FILE_PATH, encoding= 'unicode_escape')

# prepare lists
Client_ID_list = tiny['SK_ID_CURR'].tolist()
Field_list = desc['Row'].tolist()

st.write("""
# Credit scoring of client's applications
""")
st.subheader('Overview - edition mode')


st.write(tiny)

# Sidebar

st.sidebar.header('User Input Values')

# get (tbc adjustable features) of a given client by ID

def client_input_features():


    Client_ID = st.sidebar.selectbox('Please select Client ID', Client_ID_list, 0)

    data = {'Client_ID': Client_ID}
    
    features = pd.DataFrame(data, index=[0])
    return features

df_client_id = client_input_features()

# get description of a field

st.sidebar.header('Get full description of a field')

def field_description():


    Field = st.sidebar.selectbox('Please select a field', Field_list, 0)

    
    Description = desc[desc['Row'] == Field]['Description']
    pd.options.display.max_colwidth = len(Description)
    return Description

txt_field_desc = field_description()

st.sidebar.text(txt_field_desc)

# Main page

st.subheader('Selected Client ID')

st.write(df_client_id)


'''
# Display prediction
def predict():

    Risk_Flag = tiny['RISK_FLAG'][tiny['SK_ID_CURR'] == df_client_id['Client_ID']]
    return Risk_Flag

risk = predict()

st.subheader('Risk Prediction')

st.write(risk)
'''

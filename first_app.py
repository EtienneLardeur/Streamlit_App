import streamlit as st
import pandas as pd
import os
import base64
import pickle
from sklearn.pipeline import make_pipeline
import lime
from lime.lime_tabular import LimeTabularExplainer
import shap


REMOTE_URL = 'https://raw.githubusercontent.com/EtienneLardeur/Streamlit_App/main/'
DATA_FILE_PATH = os.path.join(REMOTE_URL, 'tiny')
DESC_FILE_PATH = os.path.join(REMOTE_URL, 'desc.csv')
MDL_FILE_PATH = os.path.join(REMOTE_URL, 'finalized_model.sav')

desc = pd.read_csv(DESC_FILE_PATH, encoding= 'unicode_escape')
with open(DATA_FILE_PATH, mode="rb") as df:
    tiny = pickle.load(df)

# prepare model - fitted with entire applications data
def load_model(model):
    loaded_model = pickle.load(model)
    return loaded_model

model = pickle.load(open(MDL_FILE_PATH, 'rb'))
pipe = make_pipeline(model)

# prepare lists
Client_ID_list = tiny.index.tolist()
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

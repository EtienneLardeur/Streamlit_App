"""Credit Scoring Dashboard App

Author: Etienne Lardeur https://github.com/EtienneLardeur
Source: https://github.com/EtienneLardeur/Streamlit_App

"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pathlib
import pickle
import urllib
from sklearn.pipeline import make_pipeline
import lime
from lime.lime_tabular import LimeTabularExplainer
import shap


MODEL_PKL_FILE = "finalized_model.sav"
DESC_PKL_FILE = 'desc.pkl'
TINY_PKL_FILE = 'tiny.pkl'
GITHUB_ROOT = (
    "https://raw.githubusercontent.com/EtienneLardeur/Streamlit_App/main/"
)

def get_pickle(file: str):
    """An instance of an object from the pickle file"""
    github_url = GITHUB_ROOT + file
    with urllib.request.urlopen(github_url) as open_file:  # type: ignore
        return pickle.load(open_file)

model = get_pickle(MODEL_PKL_FILE)
desc = get_pickle(DESC_PKL_FILE)
tiny = get_pickle(TINY_PKL_FILE)

# create pipe to call model
pipe = make_pipeline(model)

# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

# compute initial predictions
# default threshold
threshold = 0.1

def compute_predictions(threshold=threshold, data=tiny):
    tiny_proba = pipe.predict_proba(data)
    # labels for best threshold
    tiny_pred = to_labels(tiny_proba, threshold)[:, 1]
    # check & return failure rate is realistic applied on test set
    pred_good = (tiny_pred == 0).sum()
    pred_fail = (tiny_pred == 1).sum()
    failure_rate = pred_fail / (pred_good + pred_fail)
    return failure_rate, tiny_proba, tiny_pred


# prepare lists
Client_ID_list = tiny.index.tolist()
Field_list = desc['Row'].tolist()

st.write("""
# Credit scoring of client's applications
""")

# Sidebar ##################################################

st.sidebar.header('Inputs Panel')

def launch_new_session(tiny):
    if st.sidebar.button('Launch new session'):
        # initialize results
        tiny.insert(0, column='RISK_PROBA', value='na')
        tiny.insert(0, column='RISK_FLAG', value='na')
    return tiny
    
tiny = launch_new_session(tiny)


def threshold_prediction_component():
    st.sidebar.markdown('Threshold prediction')
    threshold = st.sidebar.number_input(
        'Threshold',
        min_value=0.,
        value=0.1,
        max_value=1.)

    if st.sidebar.button('Compute predictions'):
        failure_rate, tiny_proba, tiny_pred = compute_predictions(
            threshold,
            tiny)
        tiny.insert(0, column='RISK_PROBA', value=tiny_pred)
        tiny.insert(0, column='RISK_FLAG', value=tiny_proba[:, 1])
        st.sidebar.success(f'New failure rate: {failure_rate}')

threshold_prediction_component()


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

# Main page ##################################################

st.subheader('Generate application samples please compute predictions first')

def application_samples_component():
    ''' display samples
    '''
    if st.button('Samples'):
        st.markdown('predicted __without__ difficulty to repay - sample')
        st.write(tiny[tiny['RISK_FLAG'] == 0].shape)
        st.markdown('predicted __with__ difficulty to repay - sample')
        st.write(tiny[tiny['RISK_FLAG'] == 1].shape)

application_samples_component()
    
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

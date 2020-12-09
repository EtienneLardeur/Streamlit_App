"""Credit Scoring Dashboard App

Author: Etienne Lardeur https://github.com/EtienneLardeur
Source: https://github.com/EtienneLardeur/Streamlit_App
launch (local) with command line: streamlit run local_app.py

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

# path & files to load
MODEL_PKL_FILE = "finalized_model.sav"
DESC_PKL_FILE = 'desc.pkl'
TINY_PKL_FILE = 'tiny.pkl'
GITHUB_ROOT = (
    "https://raw.githubusercontent.com/EtienneLardeur/Streamlit_App/main/"
)

# cache means unique function execution at start
# use pickel to load an object
@st.cache
def get_pickle(file: str):
    """An instance of an object from the pickle file"""
    github_url = GITHUB_ROOT + file
    with urllib.request.urlopen(github_url) as open_file:  # type: ignore
        return pickle.load(open_file)

@st.cache
def get_desc(DESC_PKL_FILE):
    desc = get_pickle(DESC_PKL_FILE)
    return desc

@st.cache
def get_mdl(MODEL_PKL_FILE):
    mdl = get_pickle(MODEL_PKL_FILE)
    return mdl

@st.cache
# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

# compute native predictions and store in a result df
@st.cache(allow_output_mutation=True)
def compute_native_predictions(tiny):
    y_proba = pipe.predict_proba(tiny)
    # native labels
    y_pred = to_labels(y_proba, 0.1)[:, 1]
    # check & return native failure rate 
    pred_good = (y_pred == 0).sum()
    pred_fail = (y_pred == 1).sum()
    failure_rate = pred_fail / (pred_good + pred_fail)
    # create the first restults df
    results = tiny.copy()
    results.insert(0, column='RISK_PROBA', value=y_proba[:, 1])
    results.insert(0, column='RISK_FLAG', value=y_pred)
    return results, failure_rate, y_proba

# actualize predictions
@st.cache(allow_output_mutation=True)
def actualize_predictions(y_proba, threshold):
    # new predictions
    y_pred = to_labels(y_proba, threshold)[:, 1]
    # check & return new failure rate 
    pred_good = (y_pred == 0).sum()
    pred_fail = (y_pred == 1).sum()
    failure_rate = pred_fail / (pred_good + pred_fail)
    # actualize restults df
    results['RISK_FLAG'] = y_pred
    return results, failure_rate


# background tasks (no component to embed functions calls)
# load descriptions and create the list of features
desc = get_desc(DESC_PKL_FILE)
field_list = desc['Feature'].tolist()
# load new applications & store index
tiny = get_pickle(TINY_PKL_FILE)
sk_id_list = tiny.index.tolist()
# load model and create pipe for futher call
model = get_mdl(MODEL_PKL_FILE)
pipe = make_pipeline(model)
# create native restults df & failure rate
results, failure_rate, y_proba = compute_native_predictions(tiny)

st.write("""
# Credit scoring of client's applications
""")

# Sidebar ##################################################

st.sidebar.header('Inputs Panel')
st.sidebar.subheader('- Failure Rate Control')
st.sidebar.write('Initial Failure Rate:', failure_rate)

def threshold_prediction_component():
    threshold = st.sidebar.number_input(
        label='Adjust threshold value, then Actualize Predictions:',
        min_value=0.,
        value=0.5,
        max_value=1.)
        
    new_failure_rate = failure_rate
    if st.sidebar.button('Actualize Predictions'):
        results, new_failure_rate = actualize_predictions(
            y_proba,
            threshold)
    st.sidebar.write('Current Failure Rate', new_failure_rate)

threshold_prediction_component()

st.sidebar.subheader('- Client selection')
def client_input_features():
    sk_id_curr = st.sidebar.selectbox('Please select Client ID', sk_id_list, 0)
    sk_row = results.loc[[sk_id_curr]]
    return sk_row, sk_id_curr

select_sk_row, select_sk_id = client_input_features()

# get description of a field

st.sidebar.subheader('- Get full description of a feature')

def field_description():


    field = st.sidebar.selectbox('Please select a feature', field_list, 0)

    
    Description = desc[desc['Feature'] == field]['Description']
    pd.options.display.max_colwidth = len(Description)
    return Description

txt_field_desc = field_description()

st.sidebar.text(txt_field_desc)

# Main page ##################################################

st.subheader('__*demo_only:*__ Generate applications sample')

def application_samples_component():
    ''' display samples
    '''
    if st.button('Samples'):
        st.markdown('predicted __without__ difficulty to repay - sample')
        st.write(results[results['RISK_FLAG'] == 0].sample(3))
        st.markdown('predicted __with__ difficulty to repay - sample')
        st.write(results[results['RISK_FLAG'] == 1].sample(3))

application_samples_component()
    
st.subheader('Selected Client')

st.write(select_sk_row)

# Lime section ################################################
st.subheader('Generate LIME explainer')

def lime_explaination(sk_id_curr):
    ''' compute and display explainer
    '''
    if st.button("Explain Results"):
        with st.spinner('Calculating...'):
            explainer = LimeTabularExplainer(
                training_data = tiny.values,
                mode='classification',
                training_labels = results[['RISK_FLAG']],
                feature_names = tiny.columns)
            exp = explainer.explain_instance(
                tiny.loc[sk_id_curr].values,
                pipe.predict_proba,
                num_features=10)
            # Display explainer HTML object
            components.html(exp.as_html(), height=800)

lime_explaination(select_sk_id)

# SHAP section #################################################

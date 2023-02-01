import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import requests
from lime import lime_tabular 
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import plotly.express as px


# to deploye locally : streamlit run dashboard.py

pickle_in = open('loan_risk_model.pkl', 'rb')
model = pickle.load(pickle_in)

df = pd.read_csv("utils_features.csv")
df.set_index("SK_ID_CURR", inplace=True)
print(df['AMT_CREDIT'])
explainer = lime_tabular.LimeTabularExplainer(
    df.drop(columns={'TARGET'}).to_numpy(),
    mode='classification',
    class_names=df['TARGET'].unique(),
    feature_names = np.array(df.drop(columns={'TARGET'}).columns.tolist())
)

st.title("Estimation de la solvabilité du crédit d'un client")
st.markdown("Cet interface vous permert de rentrer l'identité d'un client pour consulter son dossier et estimer son potentiel de solvabilité.")
st.markdown("Vous pouvez également lui expliquer la qualité de son profil au regard des autres clients de la banque.")

st.sidebar.title("Action possibles")

# Function to enter the client id et get a response from model
def predict_solvability():
    st.write("Entrez le numéro de demande du client")
    client_id = st.number_input("Numéro de demande", format="%u")
    r = requests.post(f"https://dsp7-guimard-matthieu.azurewebsites.net/predict?customer={int(client_id)}")
    print(r.text)
    if r.status_code == 200:
        st.write("Votre client existe !")
        st.write(f"Votre indice de solvabilité est de {r.json()['probabilité']}. Votre crédit est {r.json()['prediction']}")
        further_data = st.checkbox("Avoir plus de détails sur le résultat de la simulation.")
        if further_data:
            show_interpretability(client_id, r)
    else:
        st.write("Le numéro renseigné n'apparait pas dans la base de données.")
    

def show_interpretability(client_id, prediction):
    explanation = explainer.explain_instance(
        np.array(df.drop(columns={'TARGET'}).iloc[int(client_id)]),
        model._model_impl.predict_proba,
    )
    idx_importance = {}
    real_value = {}
    for feature in explanation.local_exp[1]:
        df_feature = df.drop(columns={"TARGET"}).columns.tolist()[feature[0]]
        idx_importance[df_feature] = feature[1]
        real_value[df_feature] = df[df_feature].iloc[int(client_id)]
    local_importance = pd.DataFrame(idx_importance, index=[client_id]).fillna(0)
    real_df = pd.DataFrame(real_value, index=[client_id]).fillna(0)
    print(f"real_df:{real_df}")

    st.write("Importance des variables")
    st.plotly_chart(px.bar(local_importance.iloc[0].sort_values(ascending=False)))
    st.write("Valeur réelles des variables")
    st.plotly_chart(px.bar(real_df.iloc[0]))
    

st.sidebar.subheader("sélectionner un client")
to_predict =st.sidebar.checkbox("renseigner un numéro de demande")

if to_predict:
    predict_solvability()
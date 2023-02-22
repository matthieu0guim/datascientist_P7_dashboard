import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit.components.v1 as components



# to deploye locally : streamlit run dashboard.py

pickle_in = open('loan_risk_model.pkl', 'rb')
model = pickle.load(pickle_in)

df = pd.read_csv("client_info.csv")
df.set_index("SK_ID_CURR", inplace=True)

st.title("Estimation de la solvabilité du crédit d'un client")
st.markdown("Cet interface vous permert de rentrer l'identité d'un client pour consulter son dossier et estimer son potentiel de solvabilité.")
st.markdown("Vous pouvez également lui expliquer la qualité de son profil au regard des autres clients de la banque.")

st.sidebar.title("Action possibles")



# Function to enter the client id et get a response from model
def predict_solvability(data):
    st.write("Entrez le numéro de demande du client")
    client_id = st.selectbox("Numéro de demande:", data.index.tolist())
    submit_button = st.checkbox('lancer')
    # cloud
    if submit_button:
        r = requests.post(f"https://dsp7-guimard-matthieu.azurewebsites.net/predict?customer={int(client_id)}")
    # local 
    # r =requests.post(f"http://127.0.0.1:8000/predict?customer={int(client_id)}")
        if r.status_code == 200:
            if float(r.json()['probabilité']) < 0.5:
                color = 'red'
            else:
                color = 'green'
            fig = go.Figure(go.Indicator(
                mode='gauge+number',
                gauge={'axis' : {'range': [0, 100]},
                       'bar': {'color': color}},
                value=float(r.json()['probabilité'])*100,
                title = {'text': "Probabilité de solvabilité",
                         "font": {"size":20}},
                domain = {'x': [0,1], 'y': [0,1]},
            ))
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"Votre indice de solvabilité est de {r.json()['probabilité']}. Votre crédit est {r.json()['prediction']}")
            st.write("Informations générales du client")
            table = show_table(client_id)
            st.write("Quelques graphiques:")
            
            
            # | go.Indicator(value=df.loc[297172][to_graphes[0]])
            st.dataframe(table)
            show_bar(client_id)
            distribution = st.selectbox("Comparer le dossier au reste des clients.", data.columns.tolist())
            if distribution:
                show_distribution(client_id, distribution, data)
            further_data = st.checkbox("Avoir plus de détails sur le résultat de la simulation.")
            if further_data:
                show_interpretability(client_id)
        else:
            st.write("Le numéro renseigné n'apparait pas dans la base de données.")
        

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def show_interpretability(client_id):
    r = requests.get(f"https://dsp7-guimard-matthieu.azurewebsites.net/interpretability?client_id={client_id}").json()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    st.write(f"client : {client_id}")

    shap.decision_plot(r["expected_value"], np.array(r[f"client_{client_id}_interpretability"]), feature_names=r["feature_names"], return_objects=True)
    st.pyplot(bbox_inches='tight')
    plt.clf()

    shap.bar_plot(np.array(r[f"client_{client_id}_interpretability"]),
                  feature_names=r["feature_names"])
    st.pyplot(bbox_inches='tight')
    plt.clf()
    
    shap.force_plot(r["expected_value"], np.array(r[f"client_{client_id}_interpretability"]), feature_names=r["feature_names"])
    st.pyplot(bbox_inches="tight")
    plt.clf()

    
def show_table(client_id):
    table = ['FLAG_OWN_CAR', 'NAME_EDUCATION_TYPE_Highereducation', 'REGION_RATING_CLIENT', 'INSTAL_DPD_MAX']
    tableau = {"Vous possédez une voiture" : "", "Plus haut niveau d'étude" : "", "Attractivité du lieu de vie" : "", "Retards de remboursement" : ""}
    info_client = df.loc[client_id]
    for column in table:
        if column == "FLAG_OWN_CAR":
            if info_client[column] == 0.0:
                tableau["Vous possédez une voiture"] = "NON"
            else:
                tableau["Vous possédez une voiture"] = "OUI"
        elif column == "NAME_EDUCATION_TYPE_Highereducation":
            tableau["Plus haut niveau d'étude"] = info_client[column]
        elif column == "REGION_RATING_CLIENT":
            tableau["Attractivité du lieu de vie"] = info_client[column]
        elif column == "INSTAL_DPD_MAX":
            tableau["Retards de remboursement"] = info_client[column]
    return pd.DataFrame([tableau])

def show_bar(client_id):
    to_graphes = ['PAYMENT_RATE', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
    st.plotly_chart([go.Bar(y=df.loc[client_id][to_graphes[1:]].values, x=to_graphes[1:])])
    st.title("Taux de remboursement mensuel")
    st.plotly_chart([go.Indicator(value = df.loc[client_id][to_graphes[0]] * 100 )])

def show_distribution(client_id, var, data):
    client_value = data.loc[client_id][var]
    st.write(f"Votre valeur est de ")
    fig1 = px.histogram(data, x=var, color='TARGET', marginal='box')
    fig1.add_vline(x=client_value)
    st.plotly_chart(fig1)
    
st.sidebar.subheader("sélectionner un client")
to_predict = st.sidebar.checkbox("renseigner un numéro de demande")


if to_predict:
    predict_solvability(df)
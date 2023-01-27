import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px

df = pd.read_csv("utils_features.csv")

st.title("Estimation de la solvabilité du crédit d'un client")
st.markdown("Cet interface vous permert de rentrer l'identité d'un client pour consulter son dossier et estimer son potentiel de solvabilité.")
st.markdown("Vous pouvez également lui expliquer la qualité de son profil au regard des autres clients de la banque.")

st.sidebar.title("Action possibles")

# Function to enter the client id et get a response from model
def predict_solvability():
    st.write("Entrez le numéro de demande du client")
    client_id = st.number_input("Numéro de demande")
    


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Charger le modèle pré-entraîné
pipeline = joblib.load('logistic_model.pkl')  # Assurez-vous d'avoir sauvegardé votre pipeline avec joblib

# Créer l'interface Streamlit
st.title('Prédiction du Churn')

# Définir les champs de saisie pour les fonctionnalités
region = st.text_input('Région')
montant = st.number_input('Montant', min_value=0.0, value=0.0)
frequence_rech = st.number_input('Fréquence de Recherche', min_value=0.0, value=0.0)
revenue = st.number_input('Revenu', min_value=0.0, value=0.0)
arpu_segment = st.number_input('ARPU Segment', min_value=0.0, value=0.0)
frequence = st.number_input('Fréquence', min_value=0.0, value=0.0)
data_volume = st.number_input('Volume de données', min_value=0.0, value=0.0)
on_net = st.number_input('On Net', min_value=0.0, value=0.0)
orange = st.number_input('Orange', min_value=0.0, value=0.0)
tigo = st.number_input('Tigo', min_value=0.0, value=0.0)
zone1 = st.number_input('Zone 1', min_value=0.0, value=0.0)
zone2 = st.number_input('Zone 2', min_value=0.0, value=0.0)
regularity = st.number_input('Régularité', min_value=0.0, value=0.0)
top_pack = st.number_input('Top Pack', min_value=0.0, value=0.0)
freq_top_pack = st.number_input('Fréquence Top Pack', min_value=0.0, value=0.0)

# Bouton de validation
if st.button('Faire une prédiction'):
    # Créer un DataFrame à partir des entrées de l'utilisateur
    user_input = pd.DataFrame({
        'REGION': [region],
        'MONTANT': [montant],
        'FREQUENCE_RECH': [frequence_rech],
        'REVENUE': [revenue],
        'ARPU_SEGMENT': [arpu_segment],
        'FREQUENCE': [frequence],
        'DATA_VOLUME': [np.log(data_volume + 1) if data_volume > 0 else 0],
        'ON_NET': [on_net],
        'ORANGE': [orange],
        'TIGO': [tigo],
        'ZONE1': [zone1],
        'ZONE2': [zone2],
        'REGULARITY': [regularity],
        'TOP_PACK': [top_pack],
        'FREQ_TOP_PACK': [freq_top_pack]
    })

    # Faire une prédiction
    prediction = pipeline.predict(user_input)

    # Afficher le résultat
    st.write('Prédiction :', 'Churn' if prediction[0] == 1 else 'No Churn')

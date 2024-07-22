import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import streamlit as st
#pip inimport matplotlib.pyplot as plt
import joblib

# Charger les données
file_path = 'Expresso_churn_dataset.csv'
data = pd.read_csv(file_path)

# Supprimer les colonnes constantes
data = data.drop(columns=['MRG', 'TENURE', 'user_id'])

# Appliquer des transformations
data['DATA_VOLUME'] = data['DATA_VOLUME'].apply(lambda x: np.log(x + 1) if x > 0 else 0)

# Définir les colonnes catégorielles
categorical_columns = data.select_dtypes(include=['object']).columns

# Remplacer les valeurs manquantes dans les colonnes catégorielles avant d'encoder
data[categorical_columns] = data[categorical_columns].fillna('Unknown')

# Sélection de la variable cible et des fonctionnalités
target = 'CHURN'
features = data.drop(columns=[target])

X = features
y = data[target]

# Définir les colonnes numériques et catégorielles
numeric_features = X.select_dtypes(include=[np.number]).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Créer les transformateurs pour les caractéristiques numériques et catégorielles
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))  # Utilisation de drop='first' pour éviter la colinéarité
])

# Créer le préprocesseur
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Créer la pipeline du modèle de régression logistique
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le pipeline
pipeline.fit(X_train, y_train)

# Faire des prédictions
y_pred = pipeline.predict(X_test)

# Tester les performances du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Afficher le rapport de classification
print(classification_report(y_test, y_pred))

# Créer la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.show()

# Sauvegarder le modèle
joblib.dump(pipeline, 'logistic_model.pkl')
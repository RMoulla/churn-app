import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Base directory : répertoire où se trouve train.py
base_dir = os.path.dirname(os.path.abspath(__file__))

# Chargement des données
data_path = os.path.join(base_dir, 'data', 'customer_churn.csv')
data = pd.read_csv(data_path)

# Préparation des données
numeric_data = data[['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites', 'Churn']]
X = numeric_data.drop('Churn', axis=1)
y = numeric_data['Churn']

# Division en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Sauvegarde du modèle dans le dossier data
model_path = os.path.join(base_dir, 'churn_model.pkl')
print("Sauvegarde du modèle...")
joblib.dump(model, model_path)
print(f"Modèle sauvegardé dans {model_path}.")

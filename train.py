import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Chargement des données 
data = pd.read_csv('data/customer_churn.csv')

# Filtrer les 5 variables numériques et la variable cible 'Churn'
numeric_data = data[['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites', 'Churn']]

# Préparation des features (X) et de la cible (y)
X = numeric_data.drop('Churn', axis=1)
y = numeric_data['Churn']

# Division des données en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle RandomForest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Sauvegarde du modèle dans un fichier .pkl
print("Sauvegarde du modèle...")
joblib.dump(model, 'churn-model.pkl')
print("Modèle sauvegardé avec succès.")

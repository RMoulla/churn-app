import joblib
from sklearn.metrics import accuracy_score
import pandas as pd

# Charger le modèle et les données de test
model = joblib.load('churn-model.pkl')

# Charger les données
data = pd.read_csv('data/customer_churn.csv')
numeric_data = data[['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites', 'Churn']]

# Préparation des features et de la cible
X = numeric_data.drop('Churn', axis=1)
y = numeric_data['Churn']

# Prédictions
y_pred = model.predict(X)

# Calcul des performances
accuracy = accuracy_score(y, y_pred)

# Afficher la précision dans la sortie de CI/CD
print(f"Accuracy: {accuracy:.2f}")

# Vérification que la précision dépasse un seuil acceptable
assert accuracy > 0.8, "Le modèle n'a pas atteint une précision de 80% !"

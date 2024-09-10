import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Chargement des données
data = pd.read_csv('data/customer_churn.csv')

# Préparation des features et target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Division en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Sauvegarde du modèle
joblib.dump(model, 'churn-model.pkl')

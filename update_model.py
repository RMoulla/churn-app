import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Charger les données depuis le répertoire local `data/`
df = pd.read_csv('data/my-data.csv')

# Préparer les données
X = df.drop('target', axis=1)
y = df['target']

# Diviser les données en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Réentraîner le modèle
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Sauvegarder le nouveau modèle avec un numéro de version
joblib.dump(model, 'churn-model-v1.1.pkl')

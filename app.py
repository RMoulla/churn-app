from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialiser l'application Flask
app = Flask(__name__)

# Charger le modèle
model = joblib.load('churn-model.pkl')

# Définir la route pour la page d'accueil
@app.route('/')
def home():
    return "Bienvenue sur l'API de prédiction de churn !"



# Lancer l'application
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

    

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

# Définir la route pour effectuer une prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données envoyées dans la requête
        data = request.get_json()

        # Transformer les données en DataFrame
        input_data = pd.DataFrame([data])

        # Effectuer la prédiction avec le modèle
        prediction = model.predict(input_data)

        # Retourner le résultat sous forme de JSON
        return jsonify({'churn_prediction': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Lancer l'application
if __name__ == '__main__':
    app.run()

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np


# Charger le modèle de régression logistique sauvegardé
model = joblib.load('churn_model.pkl')

# Initialiser l'application Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Définir la route principale pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    age = float(request.form['Age'])
    account_manager = int(request.form['Account_Manager'])
    years = float(request.form['Years'])
    num_sites = int(request.form['Num_Sites'])

    # Créer un tableau numpy pour les données de prédiction
    features = np.array([[age, account_manager, years, num_sites]])

    # Ajouter une constante pour l'intercept
    features = np.insert(features, 0, 1, axis=1)

    # Effectuer la prédiction
    prediction = model.predict(features)

    # Convertir la prédiction en un format compréhensible
    result = int(prediction[0] > 0.5)

    return jsonify({'churn_prediction': result})

# Fonction pour lancer le serveur Flask dans un thread
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5012, debug=True) 
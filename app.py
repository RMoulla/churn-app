from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

# Initialiser l'application Flask
app = Flask(__name__)

# Charger le modèle depuis churn-model.pkl
model_path = os.path.join(os.path.dirname(__file__), 'churn_model.pkl')

try:
    model = joblib.load(model_path)
    print("Modèle chargé avec succès.")
except FileNotFoundError as e:
    print(f"Erreur : fichier modèle non trouvé -> {e}")
    model = None

# Route pour la page d'accueil
@app.route('/')
def home():
    return render_template('index.html')

# Route pour effectuer une prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            raise ValueError("Le modèle n'a pas été chargé correctement.")

        # Récupérer les données du formulaire (POST)
        data = {
            'Age': int(request.form['Age']),
            'Account_Manager': int(request.form['Account_Manager']),
            'Years': int(request.form['Years']),
            'Num_Sites': int(request.form['Num_Sites'])
        }

        # Transformer les données en DataFrame pour la prédiction
        columns = ['Age', 'Account_Manager', 'Years', 'Num_Sites']
        input_data = pd.DataFrame([data], columns=columns)

        # Faire la prédiction avec le modèle
        prediction = model.predict(input_data)

        # Retourner la prédiction sous forme de JSON
        return jsonify({'churn_prediction': int(prediction[0])})

    except Exception as e:
        # Retourner l'erreur sous forme de JSON
        return jsonify({'error': str(e)}), 400

# Lancer l'application
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

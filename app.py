from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

# Initialiser l'application Flask
app = Flask(__name__)

# Charger le modèle depuis churn-model.pkl
model_path = os.path.join(os.path.dirname(__file__), 'churn-model.pkl')
model = joblib.load(model_path)

# Route pour la page d'accueil
@app.route('/')
def home():
    return render_template('index.html')

# Route pour effectuer une prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données du formulaire (POST)
        data = {
            'Age': request.form['Age'],
            'Total_Purchase': request.form['Total_Purchase'],
            'Account_Manager': request.form['Account_Manager'],
            'Years': request.form['Years'],
            'Num_Sites': request.form['Num_Sites']
        }

        # Transformer les données en DataFrame pour la prédiction
        input_data = pd.DataFrame([data])

        # Faire la prédiction avec le modèle
        prediction = model.predict(input_data)

        # Afficher le résultat sur la page d'accueil
        return render_template('index.html', prediction=int(prediction[0]))
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Lancer l'application
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

    

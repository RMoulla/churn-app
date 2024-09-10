import unittest
from app import app
import json
import joblib

class TestChurnApp(unittest.TestCase):
    def setUp(self):
        # Initialiser un client de test pour l'application Flask
        self.app = app.test_client()
        self.app.testing = True

        # Charger le modèle réel pour effectuer les tests
        self.model = joblib.load('churn-model.pkl')

    # Test pour la route d'accueil ('/')
    def test_home(self):
        # Envoyer une requête GET à la route d'accueil
        response = self.app.get('/')
        # Vérifier que le statut de la réponse est 200 (succès)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode(), "Bienvenue sur l'API de prédiction de churn !")

    # Test pour la route de prédiction ('/predict') en utilisant le modèle réel
    def test_prediction(self):
        # Créer des données d'entrée pour la prédiction
        data = {
            'Age': 42,
            'Total_Purchase': 11000,
            'Account_Manager': 1,
            'Years': 5.5,
            'Num_Sites': 8
        }

        # Envoyer une requête POST à l'API avec les données
        response = self.app.post('/predict', 
                                 data=json.dumps(data), 
                                 content_type='application/json')
        
        # Vérifier que le statut de la réponse est 200 (succès)
        self.assertEqual(response.status_code, 200)
        
        # Vérifier que la réponse contient la clé 'churn_prediction'
        response_data = json.loads(response.data)
        self.assertIn('churn_prediction', response_data)
        # Vérifier que la prédiction est bien un entier (0 ou 1)
        self.assertIsInstance(response_data['churn_prediction'], int)

        # Faire également une prédiction directement avec le modèle pour comparer
        input_data = [[42, 11000, 1, 5.5, 8]]
        model_prediction = self.model.predict(input_data)

        # Vérifier que les prédictions du modèle et de l'API sont cohérentes
        self.assertEqual(response_data['churn_prediction'], model_prediction[0])

if __name__ == '__main__':
    unittest.main()

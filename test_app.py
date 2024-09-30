import pytest
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

# Test pour la route d'accueil ('/')
def test_home(client):
    # Envoyer une requête GET à la route d'accueil
    response = client.get('/')
    # Vérifier que le statut de la réponse est 200 (succès)
    assert response.status_code == 200
    # Vérifier que la page HTML contient un certain texte ou élément
    assert b'<title>' in response.data  # Vérifie que la page HTML contient un titre

# Test pour la route de prédiction ('/predict')
def test_prediction(client):
    # Créer des données d'entrée pour la prédiction (inclure toutes les features)
    data = {
        'Age': 42,
        'Total_Purchase': 11000,  # Ajout de la feature manquante
        'Account_Manager': 1,
        'Years': 5.5,
        'Num_Sites': 8
    }

    # Envoyer une requête POST à l'API avec les données en tant que formulaire
    response = client.post('/predict', data=data)

    # Vérifier que le statut de la réponse est 200 (succès)
    assert response.status_code == 200

    # Vérifier que la réponse contient la clé 'churn_prediction'
    response_data = response.get_json()
    assert 'churn_prediction' in response_data
    # Vérifier que la prédiction est bien un entier (0 ou 1)
    assert isinstance(response_data['churn_prediction'], int)

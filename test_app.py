import unittest
from app import app

class TestChurnApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_home(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_prediction(self):
        data = {'feature1': 1, 'feature2': 0}  # Exemples de features
        response = self.app.post('/predict', json=data)
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()

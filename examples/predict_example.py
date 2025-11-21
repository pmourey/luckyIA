"""
Predict example (archived): prefer using `ai_module.examples.predict_from_csv`.
This file is kept as a lightweight pointer to the recommended API.
"""

if __name__ == '__main__':
    print('Example archived. Use:')
    print("from ai_module import examples as ex")
    print("res = ex.predict_from_csv('data/processed/demo_input.csv', model_path='models/client_model.pth')")

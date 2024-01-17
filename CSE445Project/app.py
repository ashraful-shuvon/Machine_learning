from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the model and scaler from the file
with open('models/random_forest_model_with_scaler.pkl', 'rb') as file:
    loaded_rf_model, loaded_scaler = pickle.load(file)

# Define the expected column order for input data
expected_columns_order = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                          'EstimatedSalary', 'Geography_Germany', 'Geography_Spain', 'Gender_Male']

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    input_data = request.get_json()

    # Create a DataFrame from the input data
    new_data = pd.DataFrame(input_data, index=[0])

    # Ensure the specified categories are used during one-hot encoding
    new_data_encoded = pd.get_dummies(new_data, columns=['Geography', 'Gender'], drop_first=True, dtype=int)

    # Reorder columns to match the expected order during training
    new_data_encoded = new_data_encoded.reindex(columns=expected_columns_order, fill_value=0)

    # Apply MinMaxScaler on the numeric features
    numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    new_data_encoded = loaded_scaler.transform(new_data_encoded)

    # Use the loaded model for predictions
    loaded_predictions = loaded_rf_model.predict(new_data_encoded)

    # Return the predictions as JSON
    result = {'prediction': int(loaded_predictions[0])}
    return jsonify(result)

# Add a route to render the HTML template
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port = 5001)

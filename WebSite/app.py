# app.py
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify

# --- Configuration ---
# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to your saved Keras model
MODEL_PATH = os.path.join(BASE_DIR, 'my_ann_model.h5')

# LOCAL PATH for the dataset
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'DataSet.csv')


# --- Global Variables for Model and Data ---
model = None
original_df = None
all_training_columns = None


# --- Initialize Model and Data ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")

try:
    original_df = pd.read_csv(DATASET_PATH)
    X_raw_original = original_df.iloc[:, 1:]
    all_training_columns = pd.get_dummies(X_raw_original.astype(str), prefix=X_raw_original.columns.astype(str)).columns
    print(f"Successfully retrieved all training columns for one-hot encoding from {DATASET_PATH}.")
except FileNotFoundError:
    print(f"Error: Dataset file not found at {DATASET_PATH}. Please ensure it's in the 'data/' folder and the path is correct.")
except Exception as e:
    print(f"Error loading original dataset or processing columns from {DATASET_PATH}: {e}")

app = Flask(__name__)

# --- Preprocessing Function ---
def preprocess_user_input(input_data_dict):
    if all_training_columns is None or original_df is None:
        print("Preprocessing error: Training columns or original_df not available during function call.")
        return None

    input_df = pd.DataFrame([input_data_dict])

    expected_raw_columns = original_df.iloc[:, 1:].columns
    for col in expected_raw_columns:
        if col not in input_df.columns:
            input_df[col] = np.nan

    input_df = input_df[expected_raw_columns]
    input_df = input_df.astype(str)

    input_encoded = pd.get_dummies(input_df, prefix=input_df.columns.astype(str))
    final_input_features = input_encoded.reindex(columns=all_training_columns, fill_value=0)
    final_input_features = final_input_features.astype(np.float32)

    return final_input_features.values


# --- Routes ---

@app.route('/')
def home():
    if original_df is None or all_training_columns is None:
        return render_template('error.html', message="Application initialization failed. Dataset could not be loaded or processed. Please check server logs for details.")

    feature_options = {}
    numerical_features_to_sort_as_numbers = ['MentHlth', 'PhysHlth', 'Age', 'Education', 'Income'] # Add any other features that are numerical categories and need numerical sorting

    for col in original_df.iloc[:, 1:].columns:
        unique_values = original_df[col].astype(str).unique().tolist()

        # Check if the feature should be sorted numerically
        if col in numerical_features_to_sort_as_numbers:
            try:
                # Convert to int, sort, then convert back to string
                sorted_values = sorted([int(val) for val in unique_values if val.isdigit()]) # Only convert if it's a digit string
                unique_values_str = [str(val) for val in sorted_values]
                # Add any non-numeric values back if they exist (e.g., 'unknown')
                for val in unique_values:
                    if not val.isdigit() and val not in unique_values_str:
                        unique_values_str.append(val)
                feature_options[col] = unique_values_str
            except ValueError:
                # Fallback to string sort if conversion fails (e.g., unexpected non-numeric value)
                feature_options[col] = sorted(unique_values)
        else:
            # Default to string sorting for other categorical features
            feature_options[col] = sorted(unique_values)

    return render_template('index.html', feature_options=feature_options)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or original_df is None or all_training_columns is None:
        return render_template('result.html', prediction_text="Error: Model or preprocessing setup failed. Please check the server logs.")

    if request.method == 'POST':
        try:
            data = request.form.to_dict()
            print(f"Received form data: {data}")

            processed_data = preprocess_user_input(data)

            if processed_data is None:
                return render_template('result.html', prediction_text="Error: Invalid input data or preprocessing failed. Check server logs for details.")

            prediction_proba = model.predict(processed_data)[0][0]
            prediction_label = "Likely to have Diabetes" if prediction_proba >= 0.5 else "Unlikely to have Diabetes"

            return render_template('result.html',
                                   prediction_text=f'Prediction: {prediction_label} (Probability: {prediction_proba:.4f})')

        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('result.html', prediction_text=f"An error occurred during prediction: {e}")


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if model is None or original_df is None or all_training_columns is None:
        return jsonify({'error': 'Model or preprocessing setup failed.'}), 500

    if request.method == 'POST':
        try:
            json_data = request.get_json(force=True)
            print(f"Received JSON data: {json_data}")

            processed_data = preprocess_user_input(json_data)

            if processed_data is None:
                return jsonify({'error': 'Invalid input data or preprocessing failed.'}), 400

            prediction_proba = model.predict(processed_data)[0][0]
            prediction_label = "Likely to have Diabetes" if prediction_proba >= 0.5 else "Unlikely to have Diabetes"

            return jsonify({
                'prediction_label': prediction_label,
                'prediction_probability': float(prediction_proba)
            })

        except Exception as e:
            print(f"Error during API prediction: {e}")
            return jsonify({'error': f'An error occurred during API prediction: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
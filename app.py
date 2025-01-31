import re
import os
import shutil
from flask import Flask, jsonify, request, send_from_directory
from joblib import load, dump
import pandas as pd
from prophet import Prophet
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Load product-to-model mapping
product_to_model_mapping = {}
if os.path.exists('product_model_mapping.csv'):
    try:
        mapping_df = pd.read_csv('product_model_mapping.csv')
        product_to_model_mapping = dict(zip(mapping_df['ProductName'], mapping_df['ModelFilePath']))
        logger.info("Loaded existing model mapping")
    except Exception as e:
        logger.error(f"Error loading model mapping: {str(e)}")

def sanitize_product_name(product_name):
    return re.sub(r'[<>:"/\\|?*]', '_', product_name).strip()

def generate_predictions():
    """Shared prediction logic for both endpoints"""
    upcoming_reorders = []
    for product_name, model_path in product_to_model_mapping.items():
        try:
            model = load(model_path)
            future = model.make_future_dataframe(periods=4, freq='W', include_history=False)
            forecast = model.predict(future)
            
            # Get the category for the product
            category = historical_df[historical_df['ProductName'] == product_name]['Category'].iloc[0]
            
            for _, row in forecast[['ds', 'yhat']].iterrows():
                upcoming_reorders.append({
                    "Date": row['ds'].strftime('%Y-%m-%d'),
                    "ProductName": product_name,
                    "Category": category,  # Include category
                    "ReorderQuantity": int(round(row['yhat']))
                })
        except Exception as e:
            logger.error(f"Prediction error for {product_name}: {str(e)}")
            continue
    return upcoming_reorders

@app.route("/")
def home():
    return send_from_directory('static', 'index.html')

@app.route("/products", methods=["GET"])
def get_products():
    return jsonify({"products": list(product_to_model_mapping.keys())})

@app.route("/upload", methods=["POST"])
def upload():
    global historical_df  # Make historical_df global to access it in generate_predictions
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Read and validate CSV
        historical_df = pd.read_csv(file)
        required_columns = ['Date', 'ProductName', 'ReorderQuantity', 'Category']  # Include Category
        if not all(col in historical_df.columns for col in required_columns):
            return jsonify({"error": f"Missing required columns: {required_columns}"}), 400

        # Convert and sort dates
        historical_df['Date'] = pd.to_datetime(historical_df['Date'])
        historical_df = historical_df.sort_values('Date')

        # Track model creation
        new_models = 0
        existing_models = 0
        failed_models = 0

        # Train only new products
        for product_name, group in historical_df.groupby('ProductName'):
            if product_name in product_to_model_mapping:
                existing_models += 1
                continue

            try:
                product_df = group.rename(columns={'Date': 'ds', 'ReorderQuantity': 'y'})
                product_df = product_df.sort_values('ds').reset_index(drop=True)
                product_df['ds'] = pd.to_datetime(product_df['ds'])
                
                logger.debug(f"Training {product_name}, last date: {product_df['ds'].max()}")
                
                if product_df[['ds', 'y']].isnull().any().any():
                    logger.error(f"Missing values in {product_name}")
                    failed_models += 1
                    continue

                model = Prophet(daily_seasonality=False, changepoint_prior_scale=0.5)
                model.fit(product_df[['ds', 'y']])
                
                sanitized_name = sanitize_product_name(product_name)
                model_path = os.path.join(MODELS_DIR, f"{sanitized_name}_model.joblib")
                dump(model, model_path)
                product_to_model_mapping[product_name] = model_path
                new_models += 1

            except Exception as e:
                logger.error(f"Error training {product_name}: {str(e)}")
                failed_models += 1
                continue

        # Update mapping if new models were added
        if new_models > 0:
            pd.DataFrame(
                list(product_to_model_mapping.items()), 
                columns=['ProductName', 'ModelFilePath']
            ).to_csv('product_model_mapping.csv', index=False)

        return jsonify({
            "message": "Upload processed successfully",
            "stats": {
                "new_models_created": new_models,
                "existing_models_skipped": existing_models,
                "failed_model_creations": failed_models
            }
        })

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/upcoming_reorders", methods=["GET"])
def upcoming_reorders():
    try:
        if not product_to_model_mapping:
            return jsonify({"error": "No models found"}), 404
        return jsonify({"upcoming_reorders": generate_predictions()})
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/download_reorders", methods=["GET"])
def download_reorders():
    try:
        if not product_to_model_mapping:
            return jsonify({"error": "No models found"}), 404
            
        df = pd.DataFrame(generate_predictions())
        csv = df.to_csv(index=False)
        
        return app.response_class(
            response=csv,
            mimetype='text/csv',
            headers={'Content-disposition': 'attachment; filename=upcoming_reorders.csv'}
        )
    except Exception as e:
        logger.error(f"CSV generation error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
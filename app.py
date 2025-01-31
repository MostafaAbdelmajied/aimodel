import re
import os
import shutil
from flask import Flask, jsonify, request, send_from_directory
from joblib import load, dump
import pandas as pd
from prophet import Prophet
from flask_cors import CORS
import logging
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# base directory dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# models directories
MODELS_DIRS = [
    os.path.join(BASE_DIR, 'models1'),
    os.path.join(BASE_DIR, 'models2'),
    os.path.join(BASE_DIR, 'models3')
]

# Ensure all models directories exist
for dir_path in MODELS_DIRS:
    os.makedirs(dir_path, exist_ok=True)

#  path to the mapping file dynamically
MAPPING_FILE_PATH = os.path.join(BASE_DIR, 'product_model_mapping.csv')

# Load product-to-model mapping
product_to_model_mapping = {}
if os.path.exists(MAPPING_FILE_PATH):
    try:
        mapping_df = pd.read_csv(MAPPING_FILE_PATH)
        product_to_model_mapping = dict(zip(mapping_df['ProductName'], mapping_df['ModelFilePath']))
        logger.info("Loaded existing model mapping")
    except Exception as e:
        logger.error(f"Error loading model mapping: {str(e)}")

def sanitize_product_name(product_name):
    """Sanitize product names to make them safe for file paths."""
    return re.sub(r'[<>:"/\\|?*]', '_', product_name).strip()

def find_model_file(product_name):
    """Search for a model file across all models directories."""
    sanitized_name = sanitize_product_name(product_name)
    model_filename = f"{sanitized_name}_model.joblib"
    
    for models_dir in MODELS_DIRS:
        model_path = os.path.join(models_dir, model_filename)
        if os.path.exists(model_path):
            return model_path
    return None

def generate_predictions():
    """Shared prediction logic for both endpoints."""
    upcoming_reorders = []
    for product_name, model_path in product_to_model_mapping.items():
        try:
            # Check if the product exists in historical_df
            product_data = historical_df[historical_df['ProductName'] == product_name]
            if product_data.empty:
                logger.warning(f"No historical data found for product: {product_name}")
                continue

            # Get the category for the product
            category = product_data['Category'].iloc[0]

            # Load the model and generate predictions
            model = load(model_path)  # Use the correct path from product_to_model_mapping
            
            # Create a future dataframe starting from the current date
            current_date = datetime.now().date()
            future = model.make_future_dataframe(periods=4, freq='W', include_history=False)
            
            # Ensure the future dataframe starts from the current date
            future['ds'] = pd.date_range(start=current_date, periods=4, freq='W')
            
            forecast = model.predict(future)

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
    """Serve the home page."""
    return send_from_directory('static', 'index.html')

@app.route("/products", methods=["GET"])
def get_products():
    """Return a list of products with trained models."""
    return jsonify({"products": list(product_to_model_mapping.keys())})

@app.route("/upload", methods=["POST"])
def upload():
    """Upload a CSV file to train new models or update existing ones."""
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

        # Log the historical data for debugging
        logger.debug(f"Historical data after upload:\n{historical_df.head()}")

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
                model_filename = f"{sanitized_name}_model.joblib"
                
                # Save the model in the first available directory
                model_saved = False
                for models_dir in MODELS_DIRS:
                    model_path = os.path.join(models_dir, model_filename)
                    try:
                        dump(model, model_path)
                        product_to_model_mapping[product_name] = model_path  # Update the mapping with the correct path
                        new_models += 1
                        model_saved = True
                        break
                    except Exception as e:
                        logger.error(f"Error saving model for {product_name} in {models_dir}: {str(e)}")
                        continue

                if not model_saved:
                    failed_models += 1
                    continue

            except Exception as e:
                logger.error(f"Error training {product_name}: {str(e)}")
                failed_models += 1
                continue

        # Update mapping if new models were added
        if new_models > 0:
            pd.DataFrame(
                list(product_to_model_mapping.items()), 
                columns=['ProductName', 'ModelFilePath']
            ).to_csv(MAPPING_FILE_PATH, index=False)

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
    """Return a list of upcoming reorders based on trained models."""
    try:
        if not product_to_model_mapping:
            return jsonify({"error": "No models found"}), 404
        return jsonify({"upcoming_reorders": generate_predictions()})
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/download_reorders", methods=["GET"])
def download_reorders():
    """Download upcoming reorders as a CSV file."""
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
import joblib
import os
import pandas as pd
from django.conf import settings
from sklearn.metrics import mean_absolute_error, r2_score

import traceback

# 1. LOAD BOTH MODELS GLOBALLY
# Let Python handle the slashes by separating 'models' and the filename.
# (Standardized to lowercase 'models' to prevent Linux case-sensitive crashes)
original_model_path = os.path.join(settings.BASE_DIR, 'Models', 'salmonella_xgboost_model_1.pkl')
synthetic_model_path = os.path.join(settings.BASE_DIR, 'Models', 'salmonella_xgboost_model_2.pkl')
broth_model_path = os.path.join(settings.BASE_DIR, 'Models', 'salmonella_xgboost_model_3.pkl')
two_percent_model_path = os.path.join(settings.BASE_DIR, 'Models', 'salmonella_xgboost_model_2_percent_Fat.pkl')
data_cocktail_model_path = os.path.join(settings.BASE_DIR, 'Models', 'salmonella_xgboost_model_cocktail.pkl')

model_original = None
model_synthetic = None
model_broth = None
model_two_percent = None
model_data_cocktail = None

# Print out the exact resolved path to the Render logs so we can verify it!
print(f"DEBUG: BASE_DIR is -> {settings.BASE_DIR}")
print("DEBUG: Files inside BASE_DIR ->", os.listdir(settings.BASE_DIR))

models_folder = os.path.join(settings.BASE_DIR, 'Models')
if os.path.exists(models_folder):
    print("DEBUG: The 'Models' folder EXISTS! Files inside ->", os.listdir(models_folder))
else:
    print("DEBUG: The 'Models' folder DOES NOT EXIST on the Render server!")

# Load Original Model
try:
    model_original = joblib.load(original_model_path)
    print(f"DEBUG: Loaded Original model from {original_model_path}")
except Exception as e:
    print(f"ERROR: Failed to load Original model from {original_model_path}: {e}")

# Load Synthetic (Expanded) Model
try:
    model_synthetic = joblib.load(synthetic_model_path)
    print(f"DEBUG: Loaded Synthetic model from {synthetic_model_path}")
except Exception as e:
    print(f"ERROR: Failed to load Synthetic model from {synthetic_model_path}: {e}")

# Load Broth Model
try:
    model_broth = joblib.load(broth_model_path)
    print(f"DEBUG: Loaded Broth model from {broth_model_path}")
except Exception as e:
    print(f"ERROR: Failed to load Broth model from {broth_model_path}: {e}")

# Load 2% Fat Model
try:
    model_two_percent = joblib.load(two_percent_model_path)
    print(f"DEBUG: Loaded 2% Fat model from {two_percent_model_path}")
except Exception as e:
    print(f"ERROR: Failed to load 2% Fat model from {two_percent_model_path}: {e}")

# Load Data Cocktail Model
try:
    model_data_cocktail = joblib.load(data_cocktail_model_path)
    print(f"DEBUG: Loaded Data Cocktail model from {data_cocktail_model_path}")
except Exception as e:
    print(f"ERROR: Failed to load Data Cocktail model from {data_cocktail_model_path}: {e}")

# The exact 5 columns the model expects to see
EXPECTED_FEATURES = [
    'container Material',     
    'How Submerged', 
    'Temp(C)', 
    'Time(Min)', 
    'Starting Cell Count(Log CFU/g)'
]

def get_single_prediction(clean_df,model_choice):
    """
    Takes a clean 1-row Pandas DataFrame and returns the predicted survival count.
    """

    if model_choice == 'original':
        model = model_original
    elif model_choice == 'synthetic':
        model = model_synthetic
    elif model_choice == 'broth':
        model = model_broth
    elif model_choice == 'two_percent':
        model = model_two_percent
    elif model_choice == 'cocktail':
        model = model_data_cocktail
    else:
        raise ValueError(f"Invalid model choice: {model_choice}")

    if model is None:
        raise ValueError(f"The {model_choice} model failed to load on the server.")

    
    # Safely strip away anything that isn't one of the 5 core features
    features_only = clean_df[EXPECTED_FEATURES]
    
    
    # Predict using ONLY the 5 features
    raw_prediction = model.predict(features_only)
    
    # Grab that first item, make it a standard Python float, and round it
    final_answer = round(float(raw_prediction[0]), 2)
    
    return final_answer

def get_batch_predictions(clean_df, model_choice):
    """
    Takes a clean multi-row Pandas DataFrame and returns an array of predictions.
    """

    if model_choice == 'original':
        model = model_original
    elif model_choice == 'synthetic':
        model = model_synthetic
    elif model_choice == 'broth':
        model = model_broth
    elif model_choice == 'two_percent':
        model = model_two_percent
    elif model_choice == 'cocktail':
        model = model_data_cocktail
    else:
        raise ValueError(f"Invalid model choice: {model_choice}")

    if model is None:
        raise ValueError(f"The {model_choice} model failed to load on the server.")
        
    # Safely strip away the actual answers (and any other extra columns)
    features_only = clean_df[EXPECTED_FEATURES]
    
    # Predict all rows at once using ONLY the 5 features
    predictions = model.predict(features_only)
    
    return predictions.round(2)

def evaluate_model_accuracy(clean_df, actuals_column_name='Cell Count(Log CFU/g)', model_choice= None):
    """
    Compares the model's predictions to the actual known answers.
    Returns a dictionary of performance metrics.
    """
    if actuals_column_name not in clean_df.columns:
        return None # No actual answers provided to evaluate against

    if model_choice == 'original':
        model = model_original
    elif model_choice == 'synthetic':
        model = model_synthetic
    elif model_choice == 'broth':
        model = model_broth
    elif model_choice == 'two_percent':
        model = model_two_percent
    elif model_choice == 'cocktail':
        model = model_data_cocktail
    else:
        raise ValueError(f"Invalid model choice: {model_choice}")

    if model is None:
            raise ValueError(f"The {model_choice} model failed to load on the server.")
        
    # 1. Get the predictions using our existing function
    predictions = get_batch_predictions(clean_df, model_choice)
    
    # 2. Get the real answers from the dataset
    actuals = clean_df[actuals_column_name]
    
    # 3. Calculate the math
    mae = mean_absolute_error(actuals, predictions)
    r_squared = r2_score(actuals, predictions)
    
    # 4. Ported from ml_driver: Calculate deviation for every row
    # We use .tolist() to make sure Django can easily read the array!
    deviations = (predictions - actuals).round(2).tolist()
    
    return {
        "Mean Absolute Error": round(mae, 2),
        "R-Squared Score": round(r_squared, 3),
        "Deviation from actuals": deviations
    }
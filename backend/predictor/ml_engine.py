import joblib
import os
import pandas as pd
from django.conf import settings
from sklearn.metrics import mean_absolute_error, r2_score

import traceback

# 1. LOAD THE MODEL GLOBALLY
# Let Python handle the slashes by separating 'models' and the filename
model_path = os.path.join(settings.BASE_DIR, 'models', 'salmonella_xgboost_model.pkl')

# Print out the exact resolved path to the Render logs so we can verify it!
print(f"DEBUG: BASE_DIR is -> {settings.BASE_DIR}")
print("DEBUG: Files inside BASE_DIR ->", os.listdir(settings.BASE_DIR))

models_folder = os.path.join(settings.BASE_DIR, 'models')
if os.path.exists(models_folder):
    print("DEBUG: The 'models' folder EXISTS! Files inside ->", os.listdir(models_folder))
else:
    print("DEBUG: The 'models' folder DOES NOT EXIST on the Render server!")


# The exact 5 columns the model expects to see
EXPECTED_FEATURES = [
    'container Material',     
    'How Submerged', 
    'Temp(C)', 
    'Time(Min)', 
    'Starting Cell Count(Log CFU/g)'
]

def get_single_prediction(clean_df):
    """
    Takes a clean 1-row Pandas DataFrame and returns the predicted survival count.
    """
    if model is None:
        raise RuntimeError("The Machine Learning model is currently unavailable.")
    
    # Safely strip away anything that isn't one of the 5 core features
    features_only = clean_df[EXPECTED_FEATURES]
    
    
    # Predict using ONLY the 5 features
    raw_prediction = model.predict(features_only)
    
    # Grab that first item, make it a standard Python float, and round it
    final_answer = round(float(raw_prediction[0]), 2)
    
    return final_answer

def get_batch_predictions(clean_df):
    """
    Takes a clean multi-row Pandas DataFrame and returns an array of predictions.
    """
    if model is None:
        raise RuntimeError("The Machine Learning model is currently unavailable.")
        
    # Safely strip away the actual answers (and any other extra columns)
    features_only = clean_df[EXPECTED_FEATURES]
    
    # Predict all rows at once using ONLY the 5 features
    predictions = model.predict(features_only)
    
    return predictions.round(2)

def evaluate_model_accuracy(clean_df, actuals_column_name='Cell Count(Log CFU/g)'):
    """
    Compares the model's predictions to the actual known answers.
    Returns a dictionary of performance metrics.
    """
    if actuals_column_name not in clean_df.columns:
        return None # No actual answers provided to evaluate against
        
    # 1. Get the predictions using our existing function
    predictions = get_batch_predictions(clean_df)
    
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
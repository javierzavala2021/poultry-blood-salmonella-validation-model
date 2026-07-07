import pandas as pd
import numpy as np

# 1. Define exact target column names for the internal ML model
PREDICTOR_COLUMNS = ['Cell Count(Log CFU/g)']
EXPECTED_COLUMNS = ['Time(Min)']
POTENTIAL_TEMP_COLUMNS = ['Temp(C)', 'Temp(F)', 'Temp']
POTENTIAL_MISSING_COLS = [
    'Starting Cell Count(Log CFU/g)', 
    'container Material', 
    'How Submerged'
]

MODEL_COLUMNS = EXPECTED_COLUMNS + ['Temp(C)'] + POTENTIAL_MISSING_COLS
FINAL_COLUMNS = MODEL_COLUMNS + PREDICTOR_COLUMNS


def standardize_columns(df):
    """
    Fuzzy-matches column names so typos like 'tempature C' or 'Temperature (C)'
    are automatically renamed to your standard pipeline names.
    """
    rename_map = {}
    for col in df.columns:
        col_clean = str(col).lower().strip()
        
        # 1. Catch Temperature variations
        if 'temp' in col_clean:
            if 'f' in col_clean or 'fahrenheit' in col_clean:
                rename_map[col] = 'Temp(F)'
            elif 'c' in col_clean or 'celcius' in col_clean or 'celsius' in col_clean:
                rename_map[col] = 'Temp(C)'
            elif col_clean == 'temp' or col_clean == 'temperature':
                rename_map[col] = 'Temp'  # Unit unknown, will ask user or default to C
            else:
                rename_map[col] = 'Temp(C)' # Default assumption if "temp" is found
                
        # 2. Catch Time variations
        elif 'time' in col_clean or 'min' in col_clean:
            rename_map[col] = 'Time(Min)'
            
        # 3. Catch Material variations
        elif 'material' in col_clean or 'container' in col_clean:
            rename_map[col] = 'container Material'
            
        # 4. Catch Submersion variations
        elif 'submerg' in col_clean or 'immers' in col_clean:
            rename_map[col] = 'How Submerged'
            
        # 5. Catch Initial Count variations
        elif 'start' in col_clean or 'init' in col_clean:
            rename_map[col] = 'Starting Cell Count(Log CFU/g)'
            
        # 6. Catch Final Cell Count variations
        elif 'count' in col_clean or 'cfu' in col_clean:
            if 'start' not in col_clean and 'init' not in col_clean:
                rename_map[col] = 'Cell Count(Log CFU/g)'

    return df.rename(columns=rename_map)


def process_temperature_units(df):
    """
    Converts Temp(F) to Temp(C) if necessary and removes the old column.
    """
    if 'Temp(F)' in df.columns:
        df['Temp(C)'] = (df['Temp(F)'] - 32.0) * 5.0 / 9.0
        df = df.drop(columns=['Temp(F)'])
    return df


def process_batch_upload(uploaded_file):
    # 1. Read the uploaded file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        return {"error": "Unsupported file format. Please upload a CSV or Excel file."}
    
    # 2. FUZZY MATCH: Standardize column names (catches 'tempature C', etc.)
    df = standardize_columns(df)
    
    # 3. Check for required columns
    missing_required_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    missing_predictor_cols = [col for col in PREDICTOR_COLUMNS if col not in df.columns]
    
    # Check if ANY recognized temperature column exists
    has_temp = any(t_col in df.columns for t_col in POTENTIAL_TEMP_COLUMNS)
    missing_temp_cols = [] if has_temp else ['Temp(C)']

    missing_potential_cols = [col for col in POTENTIAL_MISSING_COLS if col not in df.columns]
    
    # 4. Hard Stop: If essential baseline columns are missing
    hard_missing = missing_required_cols + missing_predictor_cols
    if hard_missing:
        return {"error": f"Upload failed. Your dataset is missing these critical columns: {', '.join(hard_missing)}"}
    
    # 5. Identify columns that need user fallback input
    missing_columns = missing_potential_cols + missing_temp_cols
    
    if missing_columns:
        return {"success": False, "missing_columns": missing_columns, "df": df}
    else:
        return {"success": True, "missing_columns": [], "df": df}


def fill_missing_values(df, user_inputs):
    # Standardize column names first in case df was modified or reloaded
    df = standardize_columns(df)
    
    # 1. Handle Starting Cell Count if missing
    if "Starting Cell Count(Log CFU/g)" not in df.columns:
        df["Starting Cell Count(Log CFU/g)"] = np.nan
        zero_time_mask = (df["Time(Min)"] == 0)
        if zero_time_mask.any():
            baseline_count = df.loc[zero_time_mask, "Cell Count(Log CFU/g)"].iloc[0]
            df["Starting Cell Count(Log CFU/g)"] = df["Starting Cell Count(Log CFU/g)"].fillna(baseline_count)
        else:
            df["Starting Cell Count(Log CFU/g)"] = df["Starting Cell Count(Log CFU/g)"].fillna(user_inputs.get('fallback_starting_count', 0))

    # 2. Handle Container Material fallback
    if "container Material" not in df.columns:
        df["container Material"] = user_inputs.get('fallback_material', 1)
        
    # 3. Handle Submersion fallback
    if "How Submerged" not in df.columns:
        df["How Submerged"] = user_inputs.get('fallback_submerged', 2)

    # 4. Handle ambiguous "Temp" unit fallback
    if "Temp" in df.columns and "Temp(C)" not in df.columns and "Temp(F)" not in df.columns:
        temp_unit = str(user_inputs.get('fallback_temp', 'C')).upper().strip()
        if temp_unit == "C":
            df = df.rename(columns={"Temp": "Temp(C)"})
        elif temp_unit == "F":
            df = df.rename(columns={"Temp": "Temp(F)"})
            df = process_temperature_units(df)
        else:
            return {"error": "Invalid temperature unit provided. Please select either 'C' or 'F'."}
            
    # Convert Fahrenheit if present
    df = process_temperature_units(df)
    
    # Validate final required structure exists
    for col in MODEL_COLUMNS:
        if col not in df.columns:
            return {"error": f"Processing failed: Could not resolve column '{col}'."}

    # =========================================================================
    # ⚡️ THE XGBOOST CASTING BLOCK (Moved here so it runs automatically!)
    # =========================================================================
    
    # Map & cast 'How Submerged' to int
    submerged_str = df['How Submerged'].astype(str).str.lower().str.strip()
    submerged_map = {
        'partial': 1, 'partially': 1, '1': 1, '1.0': 1,
        'full': 2, 'fully': 2, 'entire': 2, 'entirely': 2, '2': 2, '2.0': 2
    }
    df['How Submerged'] = submerged_str.map(submerged_map).fillna(2)
    df['How Submerged'] = pd.to_numeric(df['How Submerged'], errors='coerce').fillna(2).astype(int)

    # Map & cast 'container Material' to int
    material_str = df['container Material'].astype(str).str.lower().str.strip()
    material_map = {
        'plastic': 1, '1': 1, '1.0': 1,
        'metal': 2, 'aluminum': 2, 'steel': 2, '2': 2, '2.0': 2
    }
    df['container Material'] = material_str.map(material_map).fillna(1)
    df['container Material'] = pd.to_numeric(df['container Material'], errors='coerce').fillna(1).astype(int)

    # Ensure continuous math columns are float64
    df['Temp(C)'] = pd.to_numeric(df['Temp(C)'], errors='coerce').astype(float)
    df['Time(Min)'] = pd.to_numeric(df['Time(Min)'], errors='coerce').astype(float)
    df['Starting Cell Count(Log CFU/g)'] = pd.to_numeric(df['Starting Cell Count(Log CFU/g)'], errors='coerce').astype(float)

    return df

def clean_dataset(df, user_inputs):
    # Ensure all missing values and columns are resolved
    df_clean = fill_missing_values(df, user_inputs)
    if isinstance(df_clean, dict) and "error" in df_clean:
        return df_clean, None
    if isinstance(df_clean, dict):
        raise ValueError("Unexpected dict returned from fill_missing_values without an error.")

    # 1. Normalize, map, and FORCE integer type for 'How Submerged'
    if df_clean['How Submerged'].dtype == object or isinstance(df_clean['How Submerged'].iloc[0], str):
        submerged_str = df_clean['How Submerged'].astype(str).str.lower().str.strip()
        submerged_map = {
            'partial': 1, 'partially': 1, '1': 1, '1.0': 1,
            'full': 2, 'fully': 2, 'entire': 2, 'entirely': 2, '2': 2, '2.0': 2
        }
        df_clean['How Submerged'] = submerged_str.map(submerged_map).fillna(2)
    
    # SOLUTION 1 APPLIED HERE: Force conversion to numeric and lock as integer
    df_clean['How Submerged'] = pd.to_numeric(df_clean['How Submerged'], errors='coerce').fillna(2).astype(int)

    # 2. Normalize, map, and FORCE integer type for 'container Material'
    if df_clean['container Material'].dtype == object or isinstance(df_clean['container Material'].iloc[0], str):
        material_str = df_clean['container Material'].astype(str).str.lower().str.strip()
        material_map = {
            'plastic': 1, '1': 1, '1.0': 1,
            'metal': 2, 'aluminum': 2, 'steel': 2, '2': 2, '2.0': 2
        }
        df_clean['container Material'] = material_str.map(material_map).fillna(1)
        
    # SOLUTION 1 APPLIED HERE: Force conversion to numeric and lock as integer
    df_clean['container Material'] = pd.to_numeric(df_clean['container Material'], errors='coerce').fillna(1).astype(int)

    # 3. Ensure numeric types for core continuous math columns (explicitly cast to float)
    df_clean['Temp(C)'] = pd.to_numeric(df_clean['Temp(C)'], errors='coerce').astype(float)
    df_clean['Time(Min)'] = pd.to_numeric(df_clean['Time(Min)'], errors='coerce').astype(float)
    df_clean['Starting Cell Count(Log CFU/g)'] = pd.to_numeric(df_clean['Starting Cell Count(Log CFU/g)'], errors='coerce').astype(float)

    # 4. Drop rows where critical numerical conversions failed
    df_clean = df_clean.dropna(subset=MODEL_COLUMNS).copy()
    df_clean = df_clean.reset_index(drop=True)

    # 5. Extract separated model features and predictors
    df_model = df_clean[MODEL_COLUMNS].copy()
    predictor_data = df_clean[PREDICTOR_COLUMNS].copy() if all(col in df_clean.columns for col in PREDICTOR_COLUMNS) else None

    # Pro-tip safety filter: Guarantee every single feature column is purely numeric
    df_model = df_model.apply(pd.to_numeric, errors='coerce')

    return df_model, predictor_data
import pandas as pd
def clean_single_input(raw_data):
    # 1. CHECK FOR MISSING OR EMPTY FIELDS FIRST
    required_keys = {
        'temp': 'Temperature',
        'time': 'Time',
        'starting_count': 'Initial Count',
        'material': 'Container Material',
        'submerged': 'Submersion Type'
    }
    
    missing_fields = []
    for key, display_name in required_keys.items():
        val = raw_data.get(key)
        # Check if the value is None or just whitespace/empty string
        if val is None or str(val).strip() == '':
            missing_fields.append(display_name)
            
    if missing_fields:
        fields_str = ", ".join(missing_fields)
        raise ValueError(f"All fields are required. Please fill in: {fields_str}")

    # 2. TYPE CASTING (Now we know strings aren't empty)
    try:
        temp = float(raw_data.get('temp'))
        time = float(raw_data.get('time'))
        starting_count = float(raw_data.get('starting_count'))
        material = int(raw_data.get('material'))
        submerged = int(raw_data.get('submerged'))
    except (TypeError, ValueError):
        raise ValueError("Please ensure temperature, time, and initial count are valid numbers.")

    # 3. BOUNDARY CHECKS: The Guardrails
    if temp < 20.0 or temp > 100.0:
        raise ValueError("Safety Alert: Temperature must be between 20°C and 100°C.")
    
    if time < 0.0 or time > 120.0:
        raise ValueError("Safety Alert: Time must be between 0 and 120 minutes.")
    
    if starting_count < 0.0 or starting_count > 50.0:
        raise ValueError("Safety Alert: Starting count must be between 0 and 50 Log CFU/g.")
    
    if material not in [1, 2] or submerged not in [1, 2]:
        raise ValueError("Invalid selection for Material or Submersion.")

# 4. RETURN THE READY-TO-USE DATAFRAME IN THE EXACT MODEL ORDER
    return pd.DataFrame({
        'container Material': [material],
        'How Submerged': [submerged],
        'Temp(C)': [temp],
        'Time(Min)': [time],
        'Starting Cell Count(Log CFU/g)': [starting_count]
    })
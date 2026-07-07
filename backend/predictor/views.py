from django.shortcuts import render
import pandas as pd
import os
from django.core.files.storage import FileSystemStorage

# Import our custom helper file
from . import ml_engine
from helpers import dataset_CheckerAndCleaner as datasetChecker
from helpers import singleEntry_Checker as singleChecker

def home(request):
    # Context dictionary to send data back to the HTML page
    context = {}
    
    if request.method == 'POST':
        # --- TRAFFIC COP: Which form was submitted? ---
        prediction_type = request.POST.get('prediction_type')

        # ---------------------------------------------------------
        # ROUTE 1: SINGLE PREDICTION
        # ---------------------------------------------------------
        if prediction_type == 'single':
            try:
                # 1. THE GATEKEEPER: Check inputs before they hit the Bouncer
                temp = request.POST.get('temp')
                time = request.POST.get('time')
                count = request.POST.get('starting_count')

                # Reject empty fields explicitly
                if not temp or not time or not count:
                    raise ValueError("All fields are required. Please fill in temperature, time, and initial count.")
                
                # Reject impossible negative values explicitly
                if float(temp) < 0: 
                    raise ValueError("Temperature cannot be negative.")
                if float(time) < 0: 
                    raise ValueError("Time cannot be negative.")
                if float(count) < 0: 
                    raise ValueError("Initial cell count cannot be negative.")

                # 2. THE BOUNCER: Clean the single HTML input
                clean_df = singleChecker.clean_single_input(request.POST)

                # 3. THE SCIENTIST: Get the prediction using the clean data
                print("⚡️ Cleaned single input DataFrame:")
                print(clean_df)
                
                # --- NEW BIOLOGICAL SURVIVAL LOGIC STARTS HERE ---
                # A. Get the raw prediction from your machine learning engine
                raw_pred = ml_engine.get_single_prediction(clean_df)
                
                # Handle whether your ML engine returns a single float or a list/array [float]
                if isinstance(raw_pred, (list, tuple)):
                    final_count = float(raw_pred[0])
                else:
                    final_count = float(raw_pred)
                
                # Ensure it doesn't display as a negative number conceptually
                final_count = max(0.0, final_count)

                # B. Apply the 1.0 Log CFU/g biological threshold check
                if final_count < 1.0:
                    is_killed = True
                    print("✅ Thermal treatment successful: Predicted survival count is below 1.0 Log CFU/g.")
                    status_title = "Successfully Killed"
                    status_message = "The thermal treatment successfully reduced Salmonella to below detectable/viable levels (0 Log CFU/g = 1 cell)."
                else:
                    is_killed = False
                    print("⚠️ Thermal treatment failed: Predicted survival count is above 1.0 Log CFU/g.")
                    status_title = "Cells Still Alive"
                    status_message = "Warning: Surviving Salmonella cells are predicted under these time and temperature conditions."

                # C. Save everything into context for the HTML template to use
                context['final_count'] = round(final_count, 2)
                context['is_killed'] = is_killed
                context['status_title'] = status_title
                context['status_message'] = status_message
                # --- NEW BIOLOGICAL SURVIVAL LOGIC ENDS HERE ---
                    
            except ValueError as e:
                # Catch our custom Gatekeeper errors OR built-in casting errors
                context['error'] = str(e)
            except Exception as e:
                context['error'] = f"An unexpected error occurred: {e}"
        
        # ---------------------------------------------------------
        # ROUTE 2: BATCH CSV UPLOAD
        # ---------------------------------------------------------
        elif prediction_type == 'batch':
            try:
                # Check if they are submitting the missing info fallback form
                if 'fallback_material' in request.POST:
                    # Load the file we temporarily saved
                    saved_path = request.POST.get('saved_file_path')
                    raw_df = pd.read_csv(saved_path)
                    
                    # Apply their chosen default answers to the entire dataset
                    raw_df['container Material'] = int(request.POST.get('fallback_material'))
                    raw_df['How Submerged'] = int(request.POST.get('fallback_submerged'))
                    
                    # Clean up the temp file from the server
                    if os.path.exists(saved_path):
                        os.remove(saved_path)
                
                # Otherwise, this is a brand new file upload
                else:
                    if 'dataset' not in request.FILES:
                        context['batch_error'] = "No file was uploaded. Please select a CSV."
                        return render(request, 'index.html', context)
                        
                    uploaded_file = request.FILES['dataset']

                    # 1. Pass the file to your dictionary-returning checker function
                    upload_result = datasetChecker.process_batch_upload(uploaded_file)

                    # 2. Hard Errors (Wrong file type or missing vital columns)
                    if "error" in upload_result:
                        context['batch_error'] = upload_result["error"]
                        return render(request, 'index.html', context)

                    # Extract the dataframe from the result dictionary
                    raw_df = upload_result["df"]

                    # 3. Needs Input (Missing potential/temp columns)
                    if not upload_result.get("success", True):
                        # Save the file temporarily so we don't lose it
                        fs = FileSystemStorage()
                        uploaded_file.seek(0) # Rewind the file cursor!
                        filename = fs.save(uploaded_file.name, uploaded_file)
                        
                        # Send the signal to the frontend to reveal the yellow warning box!
                        context['needs_user_input'] = True
                        context['temporary_file_path'] = fs.path(filename)
                        context['missing_columns'] = upload_result.get("missing_columns", [])
                        
                        # Stop the process here and show the form
                        return render(request, 'index.html', context)

                # 4. Clean the Dataset (Proceeds if file was perfect, or if fallback form was submitted)
                clean_df = datasetChecker.fill_missing_values(raw_df, {})

                # SAFETY CHECK: If fill_missing_values returned an error dictionary, stop here!
                if isinstance(clean_df, dict) and "error" in clean_df:
                    context['batch_error'] = clean_df["error"]
                    return render(request, 'index.html', context)
                
                # Safely print dtypes only if clean_df is a pandas DataFrame
                if isinstance(clean_df, pd.DataFrame):
                    print("⚡️ Verified XGBoost Input Dtypes:\n", clean_df.dtypes)
                else:
                    print("⚡️ clean_df is not a DataFrame; skipping dtypes print.")
                
                # Ensure we have a valid DataFrame before continuing.
                if not isinstance(clean_df, pd.DataFrame):
                    context['batch_error'] = "Cleaned data is not a valid DataFrame."
                    return render(request, 'index.html', context)
                
                # 5. THE SCIENTIST (EVALUATION): Did they upload the real answers?
                deviations = None
                if 'Cell Count(Log CFU/g)' in clean_df.columns:
                    report_card = ml_engine.evaluate_model_accuracy(clean_df, 'Cell Count(Log CFU/g)')
                    if report_card:
                        # Pop out the large array of deviation numbers so the report card only has the clean summary stats
                        deviations = report_card.pop("Deviation from actuals", None)
                        context['evaluation'] = report_card


                # 6. THE SCIENTIST (PREDICTION): Predict all the rows
                predictions = ml_engine.get_batch_predictions(clean_df)
                clean_df['Predicted Survival (Log CFU/g)'] = predictions
                
                # If we calculated deviations, add them as a column right next to the predictions!
                if deviations is not None:
                    clean_df['Deviation (Log CFU/g)'] = deviations
                
                # =========================================================
                # ⚡️ NEW: CALCULATE DATASET-WIDE AVERAGES & METRICS
                # =========================================================
                avg_predicted = round(clean_df['Predicted Survival (Log CFU/g)'].mean(), 2)
                
                summary_stats = {
                    'total_rows': len(clean_df),
                    'avg_predicted': avg_predicted,
                }
                
                # If actual cell counts exist, calculate average actuals and Mean Absolute Error (MAE)
                if 'Cell Count(Log CFU/g)' in clean_df.columns:
                    avg_actual = round(clean_df['Cell Count(Log CFU/g)'].mean(), 2)
                    # Mean Absolute Deviation across ALL rows
                    mean_abs_dev = round(abs(clean_df['Deviation (Log CFU/g)']).mean(), 3)
                    
                    summary_stats['avg_actual'] = avg_actual
                    summary_stats['mean_abs_dev'] = mean_abs_dev
                
                context['batch_summary'] = summary_stats
                # =========================================================

                context['batch_success'] = True
                # Keep preview to 10 rows for page speed, but now we have global stats!
                context['batch_table'] = clean_df.head(10).to_html(classes="min-w-full text-sm text-left", index=False)
            except ValueError as e:
                # Catch data drift or missing column errors raised by the Bouncer!
                context['batch_error'] = str(e)
            except Exception as e:
                context['batch_error'] = f"Could not process the uploaded file. Error: {e}"

    # Render the HTML page and pass all our collected variables to it
    return render(request, 'index.html', context)
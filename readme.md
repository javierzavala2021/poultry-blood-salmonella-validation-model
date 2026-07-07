# 🔬 Salmonella Predictive Microbiology & Validation Model

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-5.0%2B-092E20?style=for-the-badge&logo=django&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML_Engine-orange?style=for-the-badge)
![HTMX](https://img.shields.io/badge/HTMX-Interactive_UI-336699?style=for-the-badge)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)

An interactive, web-based predictive microbiology platform designed to calculate and validate the thermal inactivation (survival) of ***Salmonella*** in Poultry blood. 

Powered by a Django backend, an XGBoost machine learning engine, and a dynamic zero-reload frontend (HTMX + Tailwind CSS), this tool allows food safety scientists, researchers, and students to perform real-time single-condition calculations as well as large-scale bulk dataset validations against laboratory actuals. this was made to help the rendering indestry to see if the safety processes they are doing is successfully killing salmonella. 

---

## 🌟 Key Features

### 1. 🧮 Real-Time Single Prediction Calculator
* **Dynamic Parameter Inputs:** Users specify thermal treatment **Temperature (°C)**, **Time (minutes)**, **Initial Cell Count (Log CFU/g)**, **Container Material** (Plastic vs. Metal), and **Submersion Method** (Partial vs. Full).
* **Instant Biological Safety Alert:** * If the predicted final count drops below **`1.0 Log CFU/g`** ($10^0 = 1\text{ cell}$), the interface renders an emerald green **"Successfully Killed"** alert indicating viable cells have been reduced below detectable levels.
  * If the count is $\ge 1.0\text{ Log CFU/g}$, a rose-red **"Cells Still Alive"** warning alerts the user to potential biological hazard.
* **Zero Page Reloads:** Uses HTMX partial swapping to update only the result card without freezing input buttons or losing UI state.

### 2. 📊 Intelligent Batch Validation & CSV Processing
* **Automated Data Cleaning & Fuzzy Matching:** Upload laboratory CSV or Excel spreadsheets without worrying about minor column naming discrepancies. The pipeline automatically recognizes variations like `'tempature C'`, `'Temp(F)'`, `'time (min)'`, and `'init count'`.
* **Automatic Unit Conversion:** Detects Fahrenheit columns (`Temp(F)`) and automatically converts them to Celsius (`Temp(C)`) before feeding data into the ML model.
* **Text-to-Integer Categorical Translation:** The XGBoost engine requires strict numerical data types (`int64`). The preprocessing pipeline automatically translates scientific text descriptions into their corresponding machine-readable integers:
  * **Container Material:** `'plastic'`, `'1'` $\rightarrow$ `1` | `'metal'`, `'aluminum'`, `'steel'`, `'2'` $\rightarrow$ `2`
  * **Submersion Method:** `'partially'`, `'partial'`, `'1'` $\rightarrow$ `1` | `'entirely'`, `'full'`, `'fully'`, `'2'` $\rightarrow$ `2`
* **Interactive Missing-Data Fallback System:** If an uploaded research dataset is missing non-critical parameters (e.g., container material or submersion type), the application pauses execution, temporarily caches the file, and renders a yellow warning card prompting the user to select global default assumptions for the batch.
* **Global Summary KPI Dashboard:** Calculates dataset-wide statistical metrics before rendering previews:
  * Average Predicted Survival (Log CFU/g)
  * Average Actual Survival (if laboratory actuals are provided)
  * **Mean Absolute Deviation / Error (MAE)** between ML predictions and experimental laboratory outcomes.

---

## 🧠 Scientific & Machine Learning Architecture

The predictive engine is built on **XGBoost** (Extreme Gradient Boosting), trained on published empirical heat-resistance microbiological datasets (e.g., *Mvuyekure et al.*).

### The 5 Core Feature Matrix ($X$)
To make a prediction, the model expects a strictly ordered, pure numerical feature matrix with the following columns:
1. `container Material` *(int: 1 = Plastic, 2 = Metal)*
2. `How Submerged` *(int: 1 = Partial, 2 = Full)*
3. `Temp(C)` *(float: Continuous temperature in Celsius)*
4. `Time(Min)` *(float: Continuous exposure duration in minutes)*
5. `Starting Cell Count(Log CFU/g)` *(float: Initial bacterial load)*

### The Target Output ($y$)
* **`Predicted Survival (Log CFU/g)`**: The estimated surviving population of *Salmonella* following the thermal treatment.

---

## 🛠️ Technology Stack

* **Backend Framework:** Python 3.10+, Django 5.0+
* **Data Science & ML:** Pandas, NumPy, Scikit-Learn, XGBoost, Joblib
* **Frontend Architecture:** HTML5, HTMX (`hx-post`, `hx-target`, `hx-select`), Lucide Icons
* **Styling:** Tailwind CSS (via CDN / custom build)
* **File Handling:** Django `FileSystemStorage` for temporary fallback caching

---

## Training data 
This Model was trained from previous experiments conducted with Previous experiments in Texas A&M in Collage Station

* Jones-Ibarra, A.-M., Acuff, G. R., Alvarado, C. Z., & Taylor, T. M. (2017). Validation of thermal lethality against Salmonella enterica in poultry offal during rendering. Journal of Food Protection, 80(8), 1338–1343. https://doi.org/10.4315/0362-028X.JFP-16-554

* Mvuyekure, A. L. S., Moreira, R. G., & Taylor, T. M. (2023). Lethality validation for human pathogenic Salmonella enterica on chicken feathers and blood during simulated commercial low-temperature dry rendering. Microorganisms, 11(8), Article 2071. https://doi.org/10.3390/microorganisms11082071

* Wong de la Rosa, C., Daniels, K. A., Moreira, R. G., Kerth, C. R., & Taylor, T. M. (2020). Validating thermal lethality to Salmonella enterica in chicken blood by simulated commercial rendering. Foods, 9(12), 1883. https://doi.org/10.3390/foods9121883


---
from flask import Flask, render_template_string, request, redirect, url_for, flash
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from PIL import Image
import base64
import io
import os
import pickle
import pandas as pd
from interactive_support import *

# from dotenv import load_dotenv

# load_dotenv()

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (parent of CDM-Software)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Assume you have several trained models saved as files, here using torch as an example
# Please ensure you have the appropriate model files
MODEL_FILES = [
    os.path.join(PROJECT_ROOT, 'Software_FQE_models', 'discharge_decision_making', 'ocrl_agent_s1_fqe_obj_20250508_v0.pth'),
    os.path.join(PROJECT_ROOT, 'Software_FQE_models', 'discharge_decision_making', 'ocrl_agent_s1_fqe_con_rr_20250508_v0.pth'),
    os.path.join(PROJECT_ROOT, 'Software_FQE_models', 'discharge_decision_making', 'ocrl_agent_s1_fqe_con_los_20250508_v0.pth'),
]

# Path to the saved scaler file
SCALER_FILE = os.path.join(PROJECT_ROOT, 'models', 'minmax_scaler.pkl')

app = Flask(__name__)
# In production environment, secret key should be set using environment variables
app.secret_key = 'your-secret-key-here'

# Load the trained model
def load_model(model_index):
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(MODEL_FILES[model_index], map_location = device, weights_only = False)
    return model

# Load the saved scaler
def load_scaler():
    """
    Load the pre-trained scaler from file
    Returns scaler object and feature names (if available)
    """
    try:
        with open(SCALER_FILE, 'rb') as f:
            scaler_data = pickle.load(f)
        
        # Check if it's the format with feature names or simplified format
        if isinstance(scaler_data, dict):
            scaler = scaler_data['scaler']
            feature_names = scaler_data.get('feature_names', None)
            print("✓ Scaler with feature names loaded successfully")
            return scaler, feature_names
        else:
            # Simplified format: only scaler object
            scaler = scaler_data
            feature_names = None
            print("✓ Scaler (simplified) loaded successfully")
            return scaler, feature_names
            
    except FileNotFoundError:
        print(f"❌ Scaler file not found: {SCALER_FILE}")
        print("Please ensure you have run the data preprocessing and saved the scaler first.")
        return None, None
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        return None, None

# HTML template for the index page
INDEX_HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Select Model</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head>
<body>
<div class="container mt-5">
    <h1 class="text-center">FQE Estimation of Discharge Outcome</h1>
    <p class="text-center text-muted">Fitted Q-Evaluation for ICU Discharge Decision Making</p>
    <div class="text-center mt-4">
        <img src="data:image/jpeg;base64,{{ img_data }}" alt="Illustration" class="img-fluid">
    </div>
    
    <div class="card mt-4">
        <div class="card-body">
            <h5 class="card-title">Model Information</h5>
            <div class="row">
                <div class="col-md-4">
                    <h6 class="text-primary">Model 1</h6>
                    <p class="small"><strong>FQE Estimated Objective Cost</strong><br>
                    Evaluates the primary objective function for discharge decisions</p>
                </div>
                <div class="col-md-4">
                    <h6 class="text-success">Model 2</h6>
                    <p class="small"><strong>FQE Estimated Constraint Cost 1</strong><br>
                    Evaluates the first constraint function (e.g., readmission risk)</p>
                </div>
                <div class="col-md-4">
                    <h6 class="text-info">Model 3</h6>
                    <p class="small"><strong>FQE Estimated Constraint Cost 2</strong><br>
                    Evaluates the second constraint function (e.g., length of stay)</p>
                </div>
            </div>
        </div>
    </div>
    
    <form method="post" class="mt-4">
        <div class="form-group">
            <label for="model_selection"><strong>Select FQE Model:</strong></label>
            <select class="form-control" id="model_selection" name="model_selection">
                <option value="1">Model 1 - FQE Estimated Objective Cost</option>
                <option value="2">Model 2 - FQE Estimated Constraint Cost 1</option>
                <option value="3">Model 3 - FQE Estimated Constraint Cost 2</option>
            </select>
            <small class="form-text text-muted">
                Choose the appropriate model based on your analysis requirements:
                <br>• <strong>Objective Cost:</strong> Primary outcome estimation
                <br>• <strong>Constraint Cost 1 & 2:</strong> Additional constraint evaluations
            </small>
        </div>
        <button type="submit" class="btn btn-primary btn-lg">Confirm Selection</button>
    </form>
</div>
</body>
</html>
"""

# HTML template for the predict page
PREDICT_HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Enter Patient Data</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        .section-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 15px;
            border-radius: 8px 8px 0 0;
            margin-bottom: 0;
            font-weight: bold;
        }
        .section-card {
            border: 1px solid #dee2e6;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .form-group label {
            font-weight: 500;
            color: #495057;
        }
        .progress {
            height: 12px;
            margin-bottom: 20px;
        }
        .btn-submit {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            padding: 12px 30px;
            font-size: 16px;
            font-weight: bold;
        }
    </style>
</head>
<body>
<div class="container-fluid mt-4">
    <h1 class="text-center mb-2">Enter Patient Data</h1>
    <p class="text-center text-muted mb-4">Model {{ model_index + 1 }} Selected</p>
    
    <!-- Progress Bar -->
    <div class="progress">
        <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%" id="progress-bar">
            0/37 completed
        </div>
    </div>
    
    <form method="post" class="mt-4" onsubmit="return validateForm()">
        <!-- Generate form sections with detailed field configurations -->
        {% set field_configs = {
            0: {'min': 0, 'max': 120, 'step': 0.1, 'placeholder': 'e.g.: 65'},
            1: {'type': 'select'},
            2: {'min': 30, 'max': 200, 'step': 0.1, 'placeholder': 'e.g.: 70'},
            3: {'min': 30, 'max': 200, 'step': 0.1, 'placeholder': 'e.g.: 85'},
            4: {'min': 60, 'max': 120, 'step': 0.1, 'placeholder': 'e.g.: 95'},
            5: {'min': 5, 'max': 20, 'step': 0.1, 'placeholder': 'e.g.: 12.5'},
            6: {'min': 25, 'max': 60, 'step': 0.1, 'placeholder': 'e.g.: 40'},
            7: {'min': 7.0, 'max': 7.8, 'step': 0.01, 'placeholder': 'e.g.: 7.35'},
            8: {'min': 15, 'max': 60, 'step': 0.1, 'placeholder': 'e.g.: 38'},
            9: {'min': 2, 'max': 20, 'step': 0.1, 'placeholder': 'e.g.: 8.5'},
            10: {'min': 90, 'max': 120, 'step': 0.1, 'placeholder': 'e.g.: 102'},
            11: {'min': 0.5, 'max': 10, 'step': 0.1, 'placeholder': 'e.g.: 1.2'},
            12: {'min': 50, 'max': 400, 'step': 1, 'placeholder': 'e.g.: 110'},
            13: {'min': 1.0, 'max': 5.0, 'step': 0.1, 'placeholder': 'e.g.: 2.0'},
            14: {'min': 130, 'max': 155, 'step': 1, 'placeholder': 'e.g.: 140'},
            15: {'min': 7.2, 'max': 7.6, 'step': 0.01, 'placeholder': 'e.g.: 7.4'},
            16: {'min': 21, 'max': 100, 'step': 1, 'placeholder': 'e.g.: 40'},
            17: {'min': -15, 'max': 15, 'step': 0.1, 'placeholder': 'e.g.: -2'},
            18: {'min': 5, 'max': 100, 'step': 1, 'placeholder': 'e.g.: 25'},
            19: {'min': 0.8, 'max': 1.5, 'step': 0.01, 'placeholder': 'e.g.: 1.15'},
            20: {'min': 0.2, 'max': 10, 'step': 0.1, 'placeholder': 'e.g.: 1.5'},
            21: {'min': 50, 'max': 400, 'step': 1, 'placeholder': 'e.g.: 110'},
            22: {'min': 3.0, 'max': 6.0, 'step': 0.1, 'placeholder': 'e.g.: 4.0'},
            23: {'min': 15, 'max': 35, 'step': 0.1, 'placeholder': 'e.g.: 24'},
            24: {'min': 50, 'max': 500, 'step': 1, 'placeholder': 'e.g.: 250'},
            25: {'min': 8, 'max': 20, 'step': 0.1, 'placeholder': 'e.g.: 12'},
            26: {'min': 20, 'max': 80, 'step': 0.1, 'placeholder': 'e.g.: 35'},
            27: {'min': 0.8, 'max': 5.0, 'step': 0.1, 'placeholder': 'e.g.: 1.2'},
            28: {'min': 60, 'max': 250, 'step': 0.1, 'placeholder': 'e.g.: 120'},
            29: {'min': 40, 'max': 120, 'step': 0.1, 'placeholder': 'e.g.: 70'},
            30: {'min': 50, 'max': 150, 'step': 0.1, 'placeholder': 'e.g.: 90'},
            31: {'min': 35, 'max': 42, 'step': 0.1, 'placeholder': 'e.g.: 36.5'},
            32: {'min': 70, 'max': 100, 'step': 0.1, 'placeholder': 'e.g.: 98'},
            33: {'type': 'select'},
            34: {'min': 10, 'max': 40, 'step': 1, 'placeholder': 'e.g.: 18'},
            35: {'min': 200, 'max': 800, 'step': 10, 'placeholder': 'e.g.: 450'},
            36: {'min': 0, 'max': 10, 'step': 1, 'placeholder': 'e.g.: 0'}
        } %}
        
        {% set sections = [
            {'name': 'Basic Information', 'icon': 'fa-user', 'indices': [0, 1, 2]},
            {'name': 'Vital Signs', 'icon': 'fa-heartbeat', 'indices': [3, 28, 29, 30, 31, 32]},
            {'name': 'Blood Tests - Hematology', 'icon': 'fa-tint', 'indices': [5, 8, 9, 24, 25, 26, 27]},
            {'name': 'Blood Tests - Chemistry Panel 1', 'icon': 'fa-flask', 'indices': [10, 11, 12, 13, 14, 18, 22]},
            {'name': 'Blood Tests - Chemistry Panel 2', 'icon': 'fa-vial', 'indices': [19, 20, 21]},
            {'name': 'Blood Gas Analysis', 'icon': 'fa-lungs', 'indices': [4, 6, 7, 15, 16, 17, 23]},
            {'name': 'Neurological & Respiratory', 'icon': 'fa-brain', 'indices': [33, 34, 35]},
            {'name': 'Clinical History', 'icon': 'fa-history', 'indices': [36]}
        ] %}
        
        {% for section in sections %}
        <div class="row">
            <div class="col-12">
                <div class="section-card">
                    <h5 class="section-header">
                        <i class="fas {{ section.icon }}"></i> {{ section.name }}
                    </h5>
                    <div class="card-body">
                        <div class="row">
                            {% for idx in section.indices %}
                            <div class="col-md-6 col-lg-4">
                                <div class="form-group">
                                    <label for="input_{{ idx }}">{{ labels[idx] }}</label>
                                    {% if idx == 1 %}
                                        <select class="form-control" id="input_{{ idx }}" name="input_{{ idx }}" required>
                                            <option value="">Please select gender</option>
                                            <option value="1">Male</option>
                                            <option value="0">Female</option>
                                        </select>
                                    {% elif idx == 33 %}
                                        <select class="form-control" id="input_{{ idx }}" name="input_{{ idx }}" required>
                                            <option value="">Please select GCS Score</option>
                                            <option value="3">3 - Deep Coma</option>
                                            <option value="4">4 - Severe Coma</option>
                                            <option value="5">5 - Severe Coma</option>
                                            <option value="6">6 - Severe Coma</option>
                                            <option value="7">7 - Severe Coma</option>
                                            <option value="8">8 - Severe Coma</option>
                                            <option value="9">9 - Moderate Impairment</option>
                                            <option value="10">10 - Moderate Impairment</option>
                                            <option value="11">11 - Moderate Impairment</option>
                                            <option value="12">12 - Mild Impairment</option>
                                            <option value="13">13 - Mild Impairment</option>
                                            <option value="14">14 - Mild Impairment</option>
                                            <option value="15">15 - Normal</option>
                                        </select>
                                    {% else %}
                                        {% set config = field_configs[idx] %}
                                        <input type="number" class="form-control" id="input_{{ idx }}" name="input_{{ idx }}" 
                                               min="{{ config.min }}" max="{{ config.max }}" 
                                               step="{{ config.step }}" placeholder="{{ config.placeholder }}" required>
                                    {% endif %}
                                </div>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        {% endfor %}
        
        <div class="text-center mt-4 mb-4">
            <button type="submit" class="btn btn-primary btn-lg btn-submit">
                <i class="fas fa-calculator"></i> Submit Analysis
            </button>
        </div>
    </form>
    
    <div class="mt-4">
        {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
            {% for category, message in messages %}
            <div class="alert alert-{{ category }}" role="alert">
                {{ message }}
            </div>
            {% endfor %}
        {% endif %}
        {% endwith %}
    </div>
</div>

<script>
    document.addEventListener("DOMContentLoaded", function() {
        let inputs = document.querySelectorAll("input, select");
        let totalFields = inputs.length;
        
        // Update progress bar
        function updateProgress() {
            let filledFields = 0;
            inputs.forEach(input => {
                if (input.value.trim() !== '') {
                    filledFields++;
                }
            });
            
            let percentage = (filledFields / totalFields) * 100;
            let progressBar = document.getElementById('progress-bar');
            progressBar.style.width = percentage + '%';
            progressBar.textContent = filledFields + '/' + totalFields + ' completed';
        }
        
        // Add event listeners
        inputs.forEach((input, index) => {
            input.addEventListener("input", updateProgress);
            input.addEventListener("keypress", function(event) {
                if (event.key === "Enter") {
                    event.preventDefault();
                    let nextInput = inputs[index + 1];
                    if (nextInput) {
                        nextInput.focus();
                    }
                }
            });
        });
        
        // Initial progress update
        updateProgress();
    });
    
    function validateForm() {
        let inputs = document.querySelectorAll("input, select");
        let emptyFields = [];
        
        inputs.forEach((input, index) => {
            if (input.value.trim() === '') {
                emptyFields.push(input.previousElementSibling.textContent);
            }
        });
        
        if (emptyFields.length > 0) {
            alert('Please fill in the following required fields:\\n' + emptyFields.join('\\n'));
            return false;
        }
        
        return confirm('Confirm to submit patient data for analysis?');
    }
</script>
</body>
</html>
"""

@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'POST':
        model_index = int(request.form.get('model_selection')) - 1
        return redirect(url_for('predict', model_index = model_index))

    # Load image and convert to base64 for HTML rendering
    img = Image.open(os.path.join(PROJECT_ROOT, 'image', 'ConCare-RL Logo.png'))
    
    # Convert RGBA to RGB if necessary (PNG with transparency)
    if img.mode in ('RGBA', 'LA', 'P'):
        # Create a white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
        img = background
    
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return render_template_string(INDEX_HTML_TEMPLATE, model_files = len(MODEL_FILES), img_data = img_str)

@app.route('/predict/<int:model_index>', methods = ['GET', 'POST'])
def predict(model_index):
    model = load_model(model_index)
    
    # Load the pre-trained scaler from training phase
    scaler, feature_names = load_scaler()
    if scaler is None:
        flash("Error: Could not load the trained scaler. Please check if the scaler file exists.", "error")
        return redirect(url_for('index'))
    
    # Define patient data labels based on state_var_table_train column names
    labels = [
        'Age (years)',  # age
        'Gender (Male=1, Female=0)',  # M
        'Weight (kg)',  # weight
        'Heart Rate (bpm)',  # Heart Rate
        'Arterial O2 Pressure (mmHg)',  # Arterial O2 pressure
        'Hemoglobin (g/dL)',  # Hemoglobin
        'Arterial CO2 Pressure (mmHg)',  # Arterial CO2 Pressure
        'pH (Venous)',  # PH (Venous)
        'Hematocrit (serum %)',  # Hematocrit (serum)
        'White Blood Cell Count (WBC, x10^9/L)',  # WBC
        'Chloride (serum, mEq/L)',  # Chloride (serum)
        'Creatinine (serum, mg/dL)',  # Creatinine (serum)
        'Glucose (serum, mg/dL)',  # Glucose (serum)
        'Magnesium (mg/dL)',  # Magnesium
        'Sodium (serum, mEq/L)',  # Sodium (serum)
        'pH (Arterial)',  # PH (Arterial)
        'Inspired Oxygen Fraction (FiO2, %)',  # Inspired O2 Fraction
        'Arterial Base Excess (mmol/L)',  # Arterial Base Excess
        'Blood Urea Nitrogen (BUN, mg/dL)',  # BUN
        'Ionized Calcium (mmol/L)',  # Ionized Calcium
        'Total Bilirubin (mg/dL)',  # Total Bilirubin
        'Glucose (whole blood, mg/dL)',  # Glucose (whole blood)
        'Potassium (serum, mEq/L)',  # Potassium (serum)
        'Bicarbonate (HCO3, mEq/L)',  # HCO3 (serum)
        'Platelet Count (x10^9/L)',  # Platelet Count
        'Prothrombin Time (sec)',  # Prothrombin time
        'Partial Thromboplastin Time (PTT, sec)',  # PTT
        'International Normalized Ratio (INR)',  # INR
        'Systolic Blood Pressure (mmHg)',  # Blood Pressure Systolic
        'Diastolic Blood Pressure (mmHg)',  # Blood Pressure Diastolic
        'Mean Blood Pressure (mmHg)',  # Blood Pressure Mean
        'Temperature (°C)',  # Temperature C
        'Oxygen Saturation (SaO2, %)',  # SaO2
        'Glasgow Coma Scale (GCS) Score',  # GCS score
        'Respiratory Rate (breaths/min)',  # Respiratory Rate
        'Tidal Volume (mL)',  # Tidal Volume
        'Readmission Count'  # readmission_count
    ]

    if request.method == 'POST':
        try:
            input_values = [float(request.form.get(f"input_{i}")) for i in range(len(labels))]
            if len(input_values) != 37:
                flash("Please enter all 37 variables, including the patient's gender.", "error")
                return redirect(url_for('predict', model_index = model_index))

            # Process all 37 features
            # Gender (index 1) and Readmission Count (index 36) should not be scaled
            all_features = np.array(input_values).reshape(1, -1)
            
            # Apply scaler to features except gender (index 1) and readmission count (index 36)
            features_to_scale = np.concatenate([
                all_features[:, :1],      # age (index 0)
                all_features[:, 2:36]     # indices 2-35 (medical indicators)
            ], axis = 1)
            scaled_features = scaler.transform(features_to_scale)
            
            # Reconstruct the final input with gender and readmission count in correct positions
            # Insert gender at position 1
            temp_input = np.insert(scaled_features, 1, input_values[1], axis = 1)
            # Insert readmission count at position 36
            final_input = np.insert(temp_input, 36, input_values[36], axis = 1)

            # Convert input to torch tensor
            final_input_tensor = torch.tensor(final_input, dtype = torch.float).to(device)
            print("Input tensor shape:", final_input_tensor.shape)

            # Use the model to make predictions and return the result
            response = model.avg_Q_value_est(final_input_tensor)
            flash(f"Model Response: {response}", "success")
        except ValueError:
            flash("Invalid input, please ensure all values are numeric.", "error")

    return render_template_string(PREDICT_HTML_TEMPLATE, labels = labels, model_index = model_index)

if __name__ == '__main__':
    # Development environment run
    app.run(host = '0.0.0.0', port = 5000, debug = False)
    
# Gunicorn entry point
def create_app():
    return app
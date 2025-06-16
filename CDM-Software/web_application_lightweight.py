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
from lightweight_model import create_lightweight_model

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (parent of CDM-Software)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Model files
MODEL_FILES = [
    os.path.join(PROJECT_ROOT, 'Software_FQE_models', 'discharge_decision_making', 'ocrl_agent_s1_fqe_obj_20250508_v0.pth'),
    os.path.join(PROJECT_ROOT, 'Software_FQE_models', 'discharge_decision_making', 'ocrl_agent_s1_fqe_con_rr_20250508_v0.pth'),
    os.path.join(PROJECT_ROOT, 'Software_FQE_models', 'discharge_decision_making', 'ocrl_agent_s1_fqe_con_los_20250508_v0.pth'),
]

# Path to the saved scaler file
SCALER_FILE = os.path.join(PROJECT_ROOT, 'models', 'minmax_scaler.pkl')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Global variable to store loaded models
loaded_models = {}

# Load the lightweight model
def load_lightweight_model(model_index):
    """Load lightweight model without depending on complete training framework"""
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # If model is already loaded, return directly
    if model_index in loaded_models:
        print(f"✓ Using cached model {model_index}")
        return loaded_models[model_index]
    
    try:
        # Create lightweight FQE model
        lightweight_model = create_lightweight_model('fqe', state_dim=37, action_dim=2, device=device)
        
        # Load weights from full model file
        success = lightweight_model.load_from_full_model(MODEL_FILES[model_index])
        
        if success:
            # Cache model
            loaded_models[model_index] = lightweight_model
            print(f"✓ Successfully loaded lightweight model {model_index}")
            return lightweight_model
        else:
            print(f"❌ Unable to load model {model_index}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to load lightweight model: {e}")
        return None

# Load the saved scaler
def load_scaler():
    """Load the pre-trained scaler from file"""
    try:
        with open(SCALER_FILE, 'rb') as f:
            scaler_data = pickle.load(f)
        
        if isinstance(scaler_data, dict):
            scaler = scaler_data['scaler']
            feature_names = scaler_data.get('feature_names', None)
            print("✓ Scaler with feature names loaded successfully")
            return scaler, feature_names
        else:
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
    <title>Select Model - Lightweight Version</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head>
<body>
<div class="container mt-5">
    <h1 class="text-center">FQE Estimation of Discharge Outcome</h1>
    <p class="text-center text-muted">Fitted Q-Evaluation for ICU Discharge Decision Making (Lightweight Version)</p>
    <div class="alert alert-info">
        <strong>✓ Lightweight Version</strong> - Independent of complete training framework, contains only inference components
    </div>
    
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

# Placeholder for HTML template (keeping it simple for now)
PREDICT_HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Lightweight Patient Data Entry</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head>
<body>
<div class="container mt-5">
    <h1 class="text-center">Enter Patient Data (Lightweight Version)</h1>
    <div class="alert alert-success">
        <strong>✓ Lightweight Version</strong> - Uses inference module only, no complete training framework required
    </div>
    
    <p class="text-center">Model {{ model_index + 1 }} Selected</p>
    
    <form method="post" class="mt-4">
        {% for i in range(37) %}
        <div class="form-group">
            <label for="input_{{ i }}">{{ labels[i] }}</label>
            {% if i == 1 %}
                <select class="form-control" id="input_{{ i }}" name="input_{{ i }}" required>
                    <option value="">Please select gender</option>
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            {% elif i == 33 %}
                <select class="form-control" id="input_{{ i }}" name="input_{{ i }}" required>
                    <option value="">Please select GCS Score</option>
                    <option value="15">15 - Normal</option>
                    <option value="14">14 - Mild Impairment</option>
                    <option value="13">13 - Mild Impairment</option>
                    <option value="12">12 - Mild Impairment</option>
                    <option value="11">11 - Moderate Impairment</option>
                    <option value="10">10 - Moderate Impairment</option>
                    <option value="9">9 - Moderate Impairment</option>
                    <option value="8">8 - Severe Coma</option>
                    <option value="7">7 - Severe Coma</option>
                    <option value="6">6 - Severe Coma</option>
                    <option value="5">5 - Severe Coma</option>
                    <option value="4">4 - Severe Coma</option>
                    <option value="3">3 - Deep Coma</option>
                </select>
            {% else %}
                <input type="number" class="form-control" id="input_{{ i }}" name="input_{{ i }}" 
                       step="0.01" required>
            {% endif %}
        </div>
        {% endfor %}
        
        <button type="submit" class="btn btn-primary btn-lg">Submit Analysis (Lightweight)</button>
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
</body>
</html>
"""

@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'POST':
        model_index = int(request.form.get('model_selection')) - 1
        return redirect(url_for('predict', model_index = model_index))

    # Load image and convert to base64 for HTML rendering
    try:
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
    except Exception as e:
        print(f"Warning: Could not load image: {e}")
        img_str = ""

    return render_template_string(INDEX_HTML_TEMPLATE, model_files = len(MODEL_FILES), img_data = img_str)

@app.route('/predict/<int:model_index>', methods = ['GET', 'POST'])
def predict(model_index):
    # Load lightweight model
    model = load_lightweight_model(model_index)
    if model is None:
        flash("Error: Could not load the lightweight model. Please check if the model file exists.", "error")
        return redirect(url_for('index'))
    
    # Load the pre-trained scaler from training phase
    scaler, feature_names = load_scaler()
    if scaler is None:
        flash("Error: Could not load the trained scaler. Please check if the scaler file exists.", "error")
        return redirect(url_for('index'))
    
    # Define patient data labels
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

            # Use the lightweight model to make predictions
            response = model.avg_Q_value_est(final_input_tensor)
            flash(f"Lightweight Model Response: {response} ✓", "success")
            
        except ValueError as e:
            flash(f"Invalid input, please ensure all values are numeric. Error: {e}", "error")
        except Exception as e:
            flash(f"Error during prediction: {e}", "error")

    return render_template_string(PREDICT_HTML_TEMPLATE, labels = labels, model_index = model_index)

if __name__ == '__main__':
    # Pre-load all models at startup for better performance
    print("🔄 Pre-loading lightweight models...")
    for i in range(len(MODEL_FILES)):
        model = load_lightweight_model(i)
        if model:
            print(f"✓ Model {i+1} loaded successfully")
        else:
            print(f"❌ Failed to load Model {i+1}")
    
    print("🚀 Starting lightweight web application...")
    # Development environment run
    app.run(host = '0.0.0.0', port = 5001, debug = False)
    
# Gunicorn entry point
def create_app():
    return app 
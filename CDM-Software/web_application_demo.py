"""
ICU Decision Support System - Web Application Demo

🔴 IMPORTANT NOTICE FOR PRODUCTION USE:
===========================================
This is a DEMONSTRATION VERSION of the ICU Decision Support System.
Several components are simplified or use placeholder data for demo purposes.

KEY REQUIREMENTS FOR PRODUCTION DEPLOYMENT:
1. SCALERS: Replace temporary scalers with actual trained scalers from preprocessing
   - Discharge decision making: Requires trained MinMaxScaler
   - Extubation decision making: Requires trained StandardScaler
   - Scaler files should be saved during the data preprocessing phase

2. MODEL FILES: Ensure all FQE model files are properly trained and saved
   - Models should be trained on appropriate datasets
   - Verify model compatibility with input data format

3. DATA VALIDATION: Implement proper input validation and error handling
   - Add range checking for medical parameters
   - Implement data quality checks
   - Add logging for audit trails

4. SECURITY: Add proper authentication and authorization
   - Implement user authentication
   - Add session management
   - Implement access controls

⚠️  WARNING: Current temporary scalers are for demonstration only!
             Do NOT use in production without proper trained scalers!
"""

from flask import Flask, render_template_string, request, redirect, url_for, flash, session
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from PIL import Image
import base64
import io
import os
import pickle
import pandas as pd
import sys
from interactive_support import *
# Import FQE class specifically to ensure model loading works
from interactive_support import FQE

# from dotenv import load_dotenv

# load_dotenv()

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (parent of CDM-Software)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Model file mapping based on decision type and threshold setting
def get_model_files(decision_type, threshold_set):
    """Get model files based on decision type and threshold setting"""
    base_path = os.path.join(PROJECT_ROOT, 'Software_FQE_models', f'{decision_type}_decision_making', 'demo_pseudo_fqe_model', threshold_set)
    
    # Extract threshold number from threshold_set (e.g., "threshold_set_1" -> "1")
    threshold_num = threshold_set.split('_')[-1]
    
    if decision_type == 'discharge':
        return [
            os.path.join(base_path, f'disch_FQE_agent_obj_{threshold_num}.pth'),
            os.path.join(base_path, f'disch_FQE_agent_con_rr_{threshold_num}.pth'),
            os.path.join(base_path, f'disch_FQE_agent_con_los_{threshold_num}.pth'),
        ]
    elif decision_type == 'extubation':
        return [
            os.path.join(base_path, f'ext_FQE_agent_obj_{threshold_num}.pth'),
            os.path.join(base_path, f'ext_FQE_agent_con_{threshold_num}.pth'),
        ]
    return []

def get_patient_labels(decision_type):
    """
    Get patient data labels based on decision type
    
    Args:
        decision_type (str): 'discharge' or 'extubation'
    
    Returns:
        list: List of patient data labels
    """
    if decision_type == 'discharge':
        # Discharge decision making: 37 variables
        return [
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
    elif decision_type == 'extubation':
        # Extubation decision making: 30 variables
        return [
            'Age (years)',  # age
            'Gender (Male=1, Female=0)',  # M
            'Weight (kg)',  # weight
            'Heart Rate (bpm)',  # Heart Rate
            'Arterial O2 Pressure (mmHg)',  # Arterial O2 pressure
            'Hemoglobin (g/dL)',  # Hemoglobin
            'Arterial CO2 Pressure (mmHg)',  # Arterial CO2 Pressure
            'Hematocrit (serum %)',  # Hematocrit (serum)
            'White Blood Cell Count (WBC, x10^9/L)',  # WBC
            'Chloride (serum, mEq/L)',  # Chloride (serum)
            'Creatinine (serum, mg/dL)',  # Creatinine (serum)
            'Glucose (serum, mg/dL)',  # Glucose (serum)
            'Magnesium (mg/dL)',  # Magnesium
            'Sodium (serum, mEq/L)',  # Sodium (serum)
            'Arterial pH',  # PH (Arterial)
            'Inspired Oxygen Fraction (FiO2, %)',  # Inspired O2 Fraction
            'Arterial Base Excess (mmol/L)',  # Arterial Base Excess
            'Blood Urea Nitrogen (BUN, mg/dL)',  # BUN
            'Potassium (serum, mEq/L)',  # Potassium (serum)
            'Bicarbonate (HCO3, mEq/L)',  # HCO3 (serum)
            'Platelet Count (x10^9/L)',  # Platelet Count
            'Systolic Blood Pressure (mmHg)',  # Blood Pressure Systolic
            'Diastolic Blood Pressure (mmHg)',  # Blood Pressure Diastolic
            'Mean Blood Pressure (mmHg)',  # Blood Pressure Mean
            'Temperature (°C)',  # Temperature C
            'Oxygen Saturation (SaO2, %)',  # SaO2
            'Glasgow Coma Scale (GCS) Score',  # GCS score
            'Positive End-Expiratory Pressure (PEEP, cmH2O)',  # PEEP
            'Respiratory Rate (breaths/min)',  # Respiratory Rate
            'Tidal Volume (mL)',  # Tidal Volume
        ]
    else:
        return []

def get_field_configs(decision_type):
    """
    Get field configurations for input validation based on decision type
    
    Args:
        decision_type (str): 'discharge' or 'extubation'
    
    Returns:
        dict: Dictionary of field configurations
    """
    if decision_type == 'discharge':
        # Discharge decision making field configurations (37 fields)
        return {
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
        }
    elif decision_type == 'extubation':
        # Extubation decision making field configurations (30 fields)
        return {
            0: {'min': 0, 'max': 120, 'step': 0.1, 'placeholder': 'e.g.: 65'},
            1: {'type': 'select'},
            2: {'min': 30, 'max': 200, 'step': 0.1, 'placeholder': 'e.g.: 70'},
            3: {'min': 30, 'max': 200, 'step': 0.1, 'placeholder': 'e.g.: 85'},
            4: {'min': 60, 'max': 120, 'step': 0.1, 'placeholder': 'e.g.: 95'},
            5: {'min': 5, 'max': 20, 'step': 0.1, 'placeholder': 'e.g.: 12.5'},
            6: {'min': 25, 'max': 60, 'step': 0.1, 'placeholder': 'e.g.: 40'},
            7: {'min': 15, 'max': 60, 'step': 0.1, 'placeholder': 'e.g.: 38'},
            8: {'min': 2, 'max': 20, 'step': 0.1, 'placeholder': 'e.g.: 8.5'},
            9: {'min': 90, 'max': 120, 'step': 0.1, 'placeholder': 'e.g.: 102'},
            10: {'min': 0.5, 'max': 10, 'step': 0.1, 'placeholder': 'e.g.: 1.2'},
            11: {'min': 50, 'max': 400, 'step': 1, 'placeholder': 'e.g.: 110'},
            12: {'min': 1.0, 'max': 5.0, 'step': 0.1, 'placeholder': 'e.g.: 2.0'},
            13: {'min': 130, 'max': 155, 'step': 1, 'placeholder': 'e.g.: 140'},
            14: {'min': 7.2, 'max': 7.6, 'step': 0.01, 'placeholder': 'e.g.: 7.4'},
            15: {'min': 21, 'max': 100, 'step': 1, 'placeholder': 'e.g.: 40'},
            16: {'min': -15, 'max': 15, 'step': 0.1, 'placeholder': 'e.g.: -2'},
            17: {'min': 5, 'max': 100, 'step': 1, 'placeholder': 'e.g.: 25'},
            18: {'min': 3.0, 'max': 6.0, 'step': 0.1, 'placeholder': 'e.g.: 4.0'},
            19: {'min': 15, 'max': 35, 'step': 0.1, 'placeholder': 'e.g.: 24'},
            20: {'min': 50, 'max': 500, 'step': 1, 'placeholder': 'e.g.: 250'},
            21: {'min': 60, 'max': 250, 'step': 0.1, 'placeholder': 'e.g.: 120'},
            22: {'min': 40, 'max': 120, 'step': 0.1, 'placeholder': 'e.g.: 70'},
            23: {'min': 50, 'max': 150, 'step': 0.1, 'placeholder': 'e.g.: 90'},
            24: {'min': 35, 'max': 42, 'step': 0.1, 'placeholder': 'e.g.: 36.5'},
            25: {'min': 70, 'max': 100, 'step': 0.1, 'placeholder': 'e.g.: 98'},
            26: {'type': 'select'},
            27: {'min': 0, 'max': 20, 'step': 0.1, 'placeholder': 'e.g.: 5.0'},
            28: {'min': 10, 'max': 40, 'step': 1, 'placeholder': 'e.g.: 18'},
            29: {'min': 200, 'max': 800, 'step': 10, 'placeholder': 'e.g.: 450'}
        }
    else:
        return {}

def get_form_sections(decision_type):
    """
    Get form sections configuration based on decision type
    
    Args:
        decision_type (str): 'discharge' or 'extubation'
    
    Returns:
        list: List of form sections with their field indices
    """
    if decision_type == 'discharge':
        return [
            {'name': 'Basic Information', 'icon': 'fa-user', 'indices': [0, 1, 2]},
            {'name': 'Vital Signs', 'icon': 'fa-heartbeat', 'indices': [3, 28, 29, 30, 31, 32]},
            {'name': 'Blood Tests - Hematology', 'icon': 'fa-tint', 'indices': [5, 8, 9, 24, 25, 26, 27]},
            {'name': 'Blood Tests - Chemistry Panel 1', 'icon': 'fa-flask', 'indices': [10, 11, 12, 13, 14, 18, 22]},
            {'name': 'Blood Tests - Chemistry Panel 2', 'icon': 'fa-vial', 'indices': [19, 20, 21]},
            {'name': 'Blood Gas Analysis', 'icon': 'fa-lungs', 'indices': [4, 6, 7, 15, 16, 17, 23]},
            {'name': 'Neurological & Respiratory', 'icon': 'fa-brain', 'indices': [33, 34, 35]},
            {'name': 'Clinical History', 'icon': 'fa-history', 'indices': [36]}
        ]
    elif decision_type == 'extubation':
        return [
            {'name': 'Basic Information', 'icon': 'fa-user', 'indices': [0, 1, 2]},
            {'name': 'Vital Signs', 'icon': 'fa-heartbeat', 'indices': [3, 21, 22, 23, 24, 25]},
            {'name': 'Blood Tests - Hematology', 'icon': 'fa-tint', 'indices': [5, 7, 8, 20]},
            {'name': 'Blood Tests - Chemistry Panel', 'icon': 'fa-flask', 'indices': [9, 10, 11, 12, 13, 17, 18, 19]},
            {'name': 'Blood Gas Analysis', 'icon': 'fa-lungs', 'indices': [4, 6, 14, 15, 16]},
            {'name': 'Neurological & Respiratory', 'icon': 'fa-brain', 'indices': [26, 27, 28, 29]}
        ]
    else:
        return []

# Path to the saved scaler files
# In production, these should be the actual trained scalers saved during preprocessing
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DISCHARGE_SCALER_FILE = os.path.join(MODELS_DIR, 'discharge_minmax_scaler.pkl')
EXTUBATION_SCALER_FILE = os.path.join(MODELS_DIR, 'extubation_standard_scaler.pkl')

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

app = Flask(__name__)
# In production environment, secret key should be set using environment variables
app.secret_key = 'your-secret-key-here'

# Global device variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom unpickler to handle module mapping issues
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Remap __main__ module references to current module
        if module == '__main__':
            if name == 'FQE':
                return FQE
            # Try to find the class in interactive_support module
            try:
                return getattr(sys.modules['interactive_support'], name)
            except (AttributeError, KeyError):
                pass
        return super().find_class(module, name)

# Load the trained model
def load_model(model_file_path):
    try:
        # First try standard loading
        model = torch.load(model_file_path, map_location = device, weights_only = False)
        return model
    except AttributeError as e:
        if "Can't get attribute" in str(e):
            print(f"⚠️  Standard loading failed: {e}")
            print("🔄 Trying custom unpickler...")
            # Try custom unpickler for models saved from __main__
            with open(model_file_path, 'rb') as f:
                model = CustomUnpickler(f).load()
            print("✓ Successfully loaded model with custom unpickler")
            return model
        else:
            raise e

# Load the saved scaler
def load_scaler(decision_type):
    """
    Load the pre-trained scaler from file based on decision type
    
    Args:
        decision_type (str): 'discharge' or 'extubation'
    
    Returns:
        tuple: (scaler object, feature names if available)
    
    Note:
        - Discharge decision making uses MinMaxScaler
        - Extubation decision making uses StandardScaler
        - In production, these should be the actual trained scalers saved during preprocessing
    """
    
    # Select appropriate scaler file and type based on decision type
    if decision_type == 'discharge':
        scaler_file = DISCHARGE_SCALER_FILE
        scaler_type = 'MinMaxScaler'
    elif decision_type == 'extubation':
        scaler_file = EXTUBATION_SCALER_FILE
        scaler_type = 'StandardScaler'
    else:
        print(f"❌ Invalid decision type: {decision_type}")
        return None, None
    
    try:
        with open(scaler_file, 'rb') as f:
            scaler_data = pickle.load(f)
        
        # Check if it's the format with feature names or simplified format
        if isinstance(scaler_data, dict):
            scaler = scaler_data['scaler']
            feature_names = scaler_data.get('feature_names', None)
            print(f"✓ {scaler_type} with feature names loaded successfully")
            return scaler, feature_names
        else:
            # Simplified format: only scaler object
            scaler = scaler_data
            feature_names = None
            print(f"✓ {scaler_type} (simplified) loaded successfully")
            return scaler, feature_names
            
    except FileNotFoundError:
        print(f"⚠️  Scaler file not found: {scaler_file}")
        print(f"⚠️  Creating temporary {scaler_type} for demonstration purposes")
        print("🔴 WARNING: In production, you MUST use the actual trained scaler saved during preprocessing!")
        print("🔴 The current temporary scaler is only for demonstration and may not provide accurate results!")
        
        # Create temporary scaler for demonstration
        if decision_type == 'discharge':
            # Create temporary MinMaxScaler for discharge
            temp_scaler = MinMaxScaler()
            # Fit with dummy data representing typical medical parameter ranges
            # For discharge: 35 features (37 total - gender at index 1 - readmission_count at index 36)
            # Features: age + indices 2-35 = 1 + 34 = 35 features
            dummy_data = np.array([
                # Format: [min_values, max_values] for 35 features (excluding gender and readmission_count)
                # age, weight, HR, ArterialO2, Hgb, ArterialCO2, pH_venous, Hct, WBC, Cl, Cr, Glu, Mg, Na, pH_arterial, FiO2, BE, BUN, Ca, Bili, Glu_blood, K, HCO3, Plt, PT, PTT, INR, SBP, DBP, MBP, Temp, SaO2, RR, TV, extra
                [18, 40, 60, 70, 8, 25, 7.0, 15, 3, 90, 0.5, 50, 1.0, 130, 7.2, 21, -15, 5, 0.8, 0.2, 50, 3.0, 15, 50, 8, 20, 0.8, 60, 40, 50, 35, 70, 10, 200, 300],  # min values (35 features)
                [90, 90, 120, 120, 18, 60, 7.8, 50, 15, 120, 8, 300, 4.5, 150, 7.6, 80, 10, 80, 1.4, 8, 350, 5.5, 35, 400, 18, 70, 4.0, 200, 110, 140, 42, 100, 40, 800, 600]   # max values (35 features)
            ])
            temp_scaler.fit(dummy_data)
            print(f"✓ Temporary MinMaxScaler created for discharge decision making (35 features)")
        else:
            # Create temporary StandardScaler for extubation
            temp_scaler = StandardScaler()
            # For extubation: 29 features (30 total - gender at index 1)
            # Features: age + indices 2-29 = 1 + 28 = 29 features
            dummy_data = np.array([
                # Generate some sample data points for fitting (29 features)
                # age, weight, HR, ArterialO2, Hgb, ArterialCO2, Hct, WBC, Cl, Cr, Glu, Mg, Na, pH, FiO2, BE, BUN, K, HCO3, Plt, SBP, DBP, MBP, Temp, SaO2, PEEP, RR, TV
                [65, 75, 80, 95, 12, 40, 38, 8, 102, 1.2, 110, 2.0, 140, 7.4, 40, -2, 25, 4.0, 24, 250, 120, 70, 90, 36.5, 98, 5.0, 18, 450],
                [70, 85, 90, 100, 14, 45, 40, 9, 105, 1.3, 120, 2.2, 142, 7.42, 45, -1, 30, 4.2, 26, 280, 130, 75, 95, 37.0, 99, 6.0, 20, 500],
                [60, 70, 85, 90, 11, 35, 36, 7, 100, 1.1, 100, 1.8, 138, 7.38, 35, -3, 20, 3.8, 22, 220, 110, 65, 85, 36.0, 97, 4.0, 16, 400]
            ])
            temp_scaler.fit(dummy_data)
            print(f"✓ Temporary StandardScaler created for extubation decision making (29 features)")
        
        return temp_scaler, None
        
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        return None, None

# HTML template for the welcome page
WELCOME_HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>ICU Decision Support System</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
        }
        .welcome-card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            padding: 40px;
            text-align: center;
            max-width: 800px;
            margin: 0 auto;
        }
        .logo-container {
            margin-bottom: 30px;
        }
        .logo-container img {
            max-width: 200px;
            height: auto;
        }
        .system-title {
            color: #333;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 15px;
        }
        .system-subtitle {
            color: #666;
            font-size: 1.2rem;
            margin-bottom: 30px;
        }
        .features-list {
            text-align: left;
            margin: 30px 0;
        }
        .feature-item {
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        .feature-item:last-child {
            border-bottom: none;
        }
        .feature-icon {
            color: #667eea;
            margin-right: 15px;
            width: 20px;
        }
        .btn-primary-custom {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            padding: 15px 40px;
            font-size: 1.1rem;
            font-weight: bold;
            border-radius: 50px;
            margin: 10px;
            transition: all 0.3s ease;
        }
        .btn-primary-custom:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .btn-secondary-custom {
            background: #6c757d;
            border: none;
            padding: 15px 40px;
            font-size: 1.1rem;
            font-weight: bold;
            border-radius: 50px;
            margin: 10px;
            transition: all 0.3s ease;
        }
        .btn-secondary-custom:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .disclaimer {
            background: #f8f9fa;
            border-left: 4px solid #dc3545;
            padding: 15px;
            margin-top: 30px;
            border-radius: 5px;
        }
        .disclaimer-title {
            color: #dc3545;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .disclaimer-text {
            color: #6c757d;
            font-size: 0.9rem;
            line-height: 1.4;
        }
    </style>
</head>
<body>
<div class="container">
    <div class="welcome-card">
        <div class="logo-container">
            <img src="data:image/jpeg;base64,{{ img_data }}" alt="ConMED-RL Logo" class="img-fluid">
        </div>
        
        <h1 class="system-title">ICU Decision Support System</h1>
        <p class="system-subtitle">Reinforcement Learning-Based Intensive Care Decision Support Tool</p>
        
        <div class="features-list">
            <div class="feature-item">
                <i class="fas fa-robot feature-icon"></i>
                <strong>AI-Powered Analysis:</strong> Decision evaluation models based on Fitted Q-Evaluation (FQE)
            </div>
            <div class="feature-item">
                <i class="fas fa-chart-line feature-icon"></i>
                <strong>Multi-dimensional Assessment:</strong> Comprehensive evaluation of patient conditions, risk factors, and constraints
            </div>
            <div class="feature-item">
                <i class="fas fa-hospital feature-icon"></i>
                <strong>ICU-Specialized:</strong> Optimized specifically for ICU discharge and extubation decision scenarios
            </div>
            <div class="feature-item">
                <i class="fas fa-shield-alt feature-icon"></i>
                <strong>Decision Support:</strong> Assists physicians in making more accurate clinical decisions
            </div>
        </div>
        
        <div class="disclaimer">
            <div class="disclaimer-title">
                <i class="fas fa-exclamation-triangle"></i> Important Notice
            </div>
            <div class="disclaimer-text">
                This system is solely a clinical decision support tool and cannot replace professional medical judgment. All clinical decisions should be made by qualified healthcare professionals based on specific patient conditions. Please ensure you fully understand the functionality and limitations of this system before use.
            </div>
        </div>
        
        <div class="disclaimer" style="border-left: 4px solid #dc3545; background: #fff3cd; border-color: #ffc107;">
            <div class="disclaimer-title" style="color: #856404;">
                <i class="fas fa-flask"></i> Demo Version Notice
            </div>
            <div class="disclaimer-text" style="color: #856404;">
                <strong>This is a demonstration version</strong> - The current system uses temporary data normalization components for demonstration purposes.<br>
                • Discharge Decision: Uses temporary MinMaxScaler<br>
                • Extubation Decision: Uses temporary StandardScaler<br>
                <strong>For production use, you must use the actual scaler files trained and saved during the data preprocessing phase.</strong>
            </div>
        </div>
        
        <div class="mt-4">
            <p class="mb-3" style="font-size: 1.1rem; color: #333;">
                <strong>Would you like to use this decision support tool?</strong>
            </p>
                         <a href="{{ url_for('decision_type_selection') }}" class="btn btn-primary btn-primary-custom">
                 <i class="fas fa-check"></i> Yes, I want to use it
             </a>
            <a href="{{ url_for('decline') }}" class="btn btn-secondary btn-secondary-custom">
                <i class="fas fa-times"></i> Not now
            </a>
        </div>
        
        <div class="mt-4">
            <small class="text-muted">
                ConMED-RL © 2025 | ICU Decision Making Support System
            </small>
        </div>
    </div>
</div>
</body>
</html>
"""

# HTML template for the index page
INDEX_HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Select Model</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
</head>
<body>
<div class="container mt-5">
    <h1 class="text-center">FQE Estimation of {{ decision_type.title() }} Outcome</h1>
    <p class="text-center text-muted">Fitted Q-Evaluation for ICU {{ decision_type.title() }} Decision Making</p>
    <div class="text-center mb-3">
        <span class="badge badge-primary">{{ decision_type.title() }} Decision</span>
        <span class="badge badge-info">{{ threshold_set.replace('_', ' ').title() }}</span>
    </div>
    <div class="text-center mt-4">
        <img src="data:image/jpeg;base64,{{ img_data }}" alt="Illustration" class="img-fluid">
    </div>
    
    <div class="card mt-4">
        <div class="card-body">
            <h5 class="card-title">Model Information</h5>
            {% if decision_type == 'discharge' %}
            <div class="row">
                <div class="col-md-4">
                    <h6 class="text-primary">Model 1 - Objective (OBJ)</h6>
                    <p class="small"><strong>Mortality Risk Estimation</strong><br>
                    <i class="fas fa-heartbeat text-danger"></i> Evaluates the risk of mortality following discharge decision</p>
                </div>
                <div class="col-md-4">
                    <h6 class="text-success">Model 2 - Constraint (CON_RR)</h6>
                    <p class="small"><strong>Readmission Risk Estimation</strong><br>
                    <i class="fas fa-hospital text-warning"></i> Evaluates the risk of patient readmission to ICU</p>
                </div>
                <div class="col-md-4">
                    <h6 class="text-info">Model 3 - Constraint (CON_LOS)</h6>
                    <p class="small"><strong>Length-of-Stay Estimation</strong><br>
                    <i class="fas fa-calendar-alt text-info"></i> Evaluates the expected length of stay in hospital</p>
                </div>
            </div>
            {% else %}
            <div class="row">
                <div class="col-md-6">
                    <h6 class="text-primary">Model 1 - Objective (OBJ)</h6>
                    <p class="small"><strong>Extubation Failure Risk Estimation</strong><br>
                    <i class="fas fa-lungs text-danger"></i> Evaluates the risk of extubation failure and reintubation</p>
                </div>
                <div class="col-md-6">
                    <h6 class="text-success">Model 2 - Constraint (CON)</h6>
                    <p class="small"><strong>Length-of-Stay Estimation</strong><br>
                    <i class="fas fa-calendar-alt text-info"></i> Evaluates the expected length of stay in ICU</p>
                </div>
            </div>
            {% endif %}
        </div>
    </div>
    
    <form method="post" class="mt-4">
        <div class="form-group">
            <label for="model_selection"><strong>Select FQE Model:</strong></label>
            <select class="form-control" id="model_selection" name="model_selection">
                {% if decision_type == 'discharge' %}
                <option value="1">Model 1 (OBJ) - Mortality Risk Estimation</option>
                <option value="2">Model 2 (CON_RR) - Readmission Risk Estimation</option>
                <option value="3">Model 3 (CON_LOS) - Length-of-Stay Estimation</option>
                {% else %}
                <option value="1">Model 1 (OBJ) - Extubation Failure Risk Estimation</option>
                <option value="2">Model 2 (CON) - Length-of-Stay Estimation</option>
                {% endif %}
            </select>
            <small class="form-text text-muted">
                Choose the appropriate model based on your analysis requirements:
                {% if decision_type == 'discharge' %}
                <br>• <strong>OBJ:</strong> Estimates mortality risk following discharge
                <br>• <strong>CON_RR:</strong> Estimates readmission risk to ICU
                <br>• <strong>CON_LOS:</strong> Estimates expected length of stay in hospital
                {% else %}
                <br>• <strong>OBJ:</strong> Estimates extubation failure and reintubation risk
                <br>• <strong>CON:</strong> Estimates expected length of stay in ICU
                {% endif %}
            </small>
        </div>
        <div class="text-center">
            <button type="submit" class="btn btn-primary btn-lg mr-3">Confirm Selection</button>
            <a href="{{ url_for('threshold_selection', decision_type=decision_type) }}" class="btn btn-secondary btn-lg">
                <i class="fas fa-arrow-left"></i> Back to Threshold Selection
            </a>
        </div>
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
    <p class="text-center text-muted mb-2">{{ decision_type.title() }} Decision Making - Model {{ model_index + 1 }} Selected ({{ labels|length }} variables)</p>
    <div class="text-center mb-4">
        <span class="badge badge-primary">{{ decision_type.title() }} Decision</span>
        <span class="badge badge-info">{{ threshold_set.replace('_', ' ').title() }}</span>
        {% if decision_type == 'discharge' %}
            {% if model_index == 0 %}
            <span class="badge badge-danger">Mortality Risk (OBJ)</span>
            {% elif model_index == 1 %}
            <span class="badge badge-warning">Readmission Risk (CON_RR)</span>
            {% elif model_index == 2 %}
            <span class="badge badge-info">Length-of-Stay (CON_LOS)</span>
            {% endif %}
        {% else %}
            {% if model_index == 0 %}
            <span class="badge badge-danger">Extubation Failure Risk (OBJ)</span>
            {% elif model_index == 1 %}
            <span class="badge badge-info">Length-of-Stay (CON)</span>
            {% endif %}
        {% endif %}
    </div>
    
    <!-- Progress Bar -->
    <div class="progress">
        <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%" id="progress-bar">
            0/{{ labels|length }} completed
        </div>
    </div>
    
    <form method="post" class="mt-4" onsubmit="return validateForm()">
        <!-- Dynamic form sections based on decision type -->
        {% for section in form_sections %}
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
                                    {% elif (decision_type == 'discharge' and idx == 33) or (decision_type == 'extubation' and idx == 26) %}
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

# HTML template for decision type selection
DECISION_TYPE_HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Select Decision Type</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        .decision-card {
            border: 2px solid #e9ecef;
            border-radius: 12px;
            padding: 30px;
            margin: 20px 0;
            transition: all 0.3s ease;
            cursor: pointer;
            background: white;
        }
        .decision-card:hover {
            border-color: #667eea;
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }
        .decision-card.discharge {
            border-left: 5px solid #28a745;
        }
        .decision-card.extubation {
            border-left: 5px solid #007bff;
        }
        .decision-icon {
            font-size: 3rem;
            margin-bottom: 20px;
        }
        .decision-icon.discharge {
            color: #28a745;
        }
        .decision-icon.extubation {
            color: #007bff;
        }
        .decision-title {
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        .decision-description {
            color: #666;
            line-height: 1.6;
        }
        .btn-select {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            padding: 12px 30px;
            font-weight: bold;
            border-radius: 25px;
            transition: all 0.3s ease;
        }
        .btn-select:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
    </style>
</head>
<body>
<div class="container mt-5">
    <h1 class="text-center mb-2">Select Decision Type</h1>
    <p class="text-center text-muted mb-5">Choose the type of ICU decision you want to analyze</p>
    
    <div class="row">
        <div class="col-md-6">
            <div class="decision-card discharge" onclick="selectDecision('discharge')">
                <div class="text-center">
                    <div class="decision-icon discharge">
                        <i class="fas fa-hospital-user"></i>
                    </div>
                    <h3 class="decision-title">Discharge Decision Making</h3>
                    <p class="decision-description">
                        Evaluate patient readiness for ICU discharge based on comprehensive clinical indicators. 
                        This model helps assess the optimal timing for transferring patients from intensive care to general ward.
                    </p>
                    <div class="mt-3">
                        <span class="badge badge-success">Available Models: 3</span>
                        <span class="badge badge-info">Threshold Sets: 2</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="col-md-6">
            <div class="decision-card extubation" onclick="selectDecision('extubation')">
                <div class="text-center">
                    <div class="decision-icon extubation">
                        <i class="fas fa-lungs"></i>
                    </div>
                    <h3 class="decision-title">Extubation Decision Making</h3>
                    <p class="decision-description">
                        Assess patient readiness for mechanical ventilation discontinuation. 
                        This model evaluates respiratory function and overall patient stability for safe extubation.
                    </p>
                    <div class="mt-3">
                        <span class="badge badge-primary">Available Models: 3</span>
                        <span class="badge badge-info">Threshold Sets: 2</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="text-center mt-4">
        <a href="{{ url_for('index') }}" class="btn btn-secondary btn-lg">
            <i class="fas fa-arrow-left"></i> Back to Welcome
        </a>
    </div>
</div>

<script>
function selectDecision(decisionType) {
    window.location.href = '/threshold_selection/' + decisionType;
}
</script>
</body>
</html>
"""

# HTML template for prediction results
RESULTS_HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Prediction Results</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px 0;
        }
        .results-container {
            max-width: 800px;
            margin: 0 auto;
        }
        .results-card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            padding: 40px;
            margin-bottom: 20px;
        }
        .prediction-value {
            font-size: 4rem;
            font-weight: bold;
            text-align: center;
            margin: 30px 0;
            padding: 30px;
            border-radius: 15px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        }
        .prediction-value.low-risk {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            color: #155724;
        }
        .prediction-value.moderate-risk {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            color: #856404;
        }
        .prediction-value.high-risk {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            color: #721c24;
        }
        .prediction-value.length-stay {
            background: linear-gradient(135deg, #cce7ff 0%, #b3d9ff 100%);
            color: #004085;
        }
        .model-info {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .interpretation-box {
            background: #e7f3ff;
            border: 1px solid #b3d9ff;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
        }
        .warning-box {
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .btn-action {
            margin: 10px;
            padding: 12px 25px;
            font-weight: bold;
            border-radius: 25px;
            transition: all 0.3s ease;
        }
        .btn-action:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .decision-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: bold;
            display: inline-block;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
<div class="container results-container">
    <div class="results-card">
        <div class="text-center">
            <div class="decision-badge">
                <i class="fas fa-{{ 'hospital-user' if decision_type == 'discharge' else 'lungs' }}"></i>
                {{ decision_type.title() }} Decision Making Results
            </div>
        </div>
        
        <h1 class="text-center mb-4">FQE Model Prediction Results</h1>
        
        <div class="model-info">
            <h5><i class="fas fa-cog"></i> Model Information</h5>
            <p><strong>Decision Type:</strong> {{ decision_type.title() }}</p>
            <p><strong>Threshold Setting:</strong> {{ threshold_set.replace('_', ' ').title() }}</p>
            <p><strong>Selected Model:</strong> Model {{ model_index + 1 }} - {{ model_description }}</p>
        </div>
        
        <div class="prediction-value {% if 'Low Risk' in risk_level %}low-risk{% elif 'Moderate Risk' in risk_level %}moderate-risk{% elif 'High Risk' in risk_level %}high-risk{% elif 'Expected:' in risk_level %}length-stay{% endif %}">
            <div class="mb-2" style="font-size: 1.2rem; font-weight: normal;">Prediction Result</div>
            <div>{{ "%.4f"|format(prediction_value) }}</div>
            <div class="mt-2" style="font-size: 1.5rem;">{{ risk_level }}</div>
        </div>
        
        <div class="interpretation-box">
            <h5><i class="fas fa-info-circle"></i> Interpretation</h5>
            <p class="mb-0">{{ interpretation }}</p>
        </div>
        
        {% if decision_type == 'discharge' %}
        <div class="mt-4">
            <h6>Risk Assessment Guidelines:</h6>
            <ul class="small">
                {% if model_index == 0 %}
                <li><strong>Low Risk (< 0.3):</strong> Patient shows good indicators for discharge</li>
                <li><strong>Moderate Risk (0.3-0.7):</strong> Careful monitoring recommended</li>
                <li><strong>High Risk (> 0.7):</strong> Consider delaying discharge</li>
                {% elif model_index == 1 %}
                <li><strong>Low Risk (< 0.2):</strong> Low probability of readmission</li>
                <li><strong>Moderate Risk (0.2-0.5):</strong> Monitor patient post-discharge</li>
                <li><strong>High Risk (> 0.5):</strong> High probability of readmission</li>
                {% else %}
                <li>Values represent expected hospital length of stay in days</li>
                {% endif %}
            </ul>
        </div>
        {% else %}
        <div class="mt-4">
            <h6>Risk Assessment Guidelines:</h6>
            <ul class="small">
                {% if model_index == 0 %}
                <li><strong>Low Risk (< 0.2):</strong> Good candidate for extubation</li>
                <li><strong>Moderate Risk (0.2-0.5):</strong> Proceed with caution</li>
                <li><strong>High Risk (> 0.5):</strong> Consider delaying extubation</li>
                {% else %}
                <li>Values represent expected ICU length of stay in days</li>
                {% endif %}
            </ul>
        </div>
        {% endif %}
        
        <div class="warning-box">
            <h6><i class="fas fa-exclamation-triangle"></i> Demo Version Warning</h6>
            <p class="mb-0">
                <strong>This result uses a temporary {{ scaler_type }} for demonstration purposes.</strong><br>
                For production use, replace with trained scaler from the data preprocessing phase.
                This demo result may not reflect actual clinical outcomes.
            </p>
        </div>
        
        <div class="text-center mt-4">
            <a href="{{ url_for('predict', decision_type=decision_type, threshold_set=threshold_set, model_index=model_index) }}" 
               class="btn btn-primary btn-action">
                <i class="fas fa-edit"></i> Analyze Another Patient
            </a>
            <a href="{{ url_for('model_selection', decision_type=decision_type, threshold_set=threshold_set) }}" 
               class="btn btn-secondary btn-action">
                <i class="fas fa-cog"></i> Change Model
            </a>
            <a href="{{ url_for('decision_type_selection') }}" 
               class="btn btn-info btn-action">
                <i class="fas fa-home"></i> New Analysis
            </a>
        </div>
        
        <div class="mt-4 text-center">
            <small class="text-muted">
                ConMED-RL © 2025 | FQE-Based ICU Decision Support System
            </small>
        </div>
    </div>
</div>
</body>
</html>
"""

# HTML template for threshold setting selection
THRESHOLD_SELECTION_HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Select Threshold Setting</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        .threshold-card {
            border: 2px solid #e9ecef;
            border-radius: 12px;
            padding: 25px;
            margin: 15px 0;
            transition: all 0.3s ease;
            cursor: pointer;
            background: white;
        }
        .threshold-card:hover {
            border-color: #667eea;
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        }
        .threshold-card.set1 {
            border-left: 5px solid #17a2b8;
        }
        .threshold-card.set2 {
            border-left: 5px solid #fd7e14;
        }
        .threshold-icon {
            font-size: 2.5rem;
            margin-bottom: 15px;
        }
        .threshold-icon.set1 {
            color: #17a2b8;
        }
        .threshold-icon.set2 {
            color: #fd7e14;
        }
        .threshold-title {
            font-size: 1.3rem;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        .threshold-description {
            color: #666;
            line-height: 1.6;
            font-size: 0.95rem;
        }
        .decision-type-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: bold;
        }
    </style>
</head>
<body>
<div class="container mt-5">
    <div class="text-center mb-4">
        <div class="decision-type-badge">
            <i class="fas fa-{{ 'hospital-user' if decision_type == 'discharge' else 'lungs' }}"></i>
            {{ decision_type.title() }} Decision Making
        </div>
    </div>
    
    <h1 class="text-center mb-2">Select Threshold Setting</h1>
    <p class="text-center text-muted mb-5">Choose the threshold configuration for your analysis</p>
    
    <div class="row">
        <div class="col-md-6">
            <div class="threshold-card set1" onclick="selectThreshold('threshold_set_1')">
                <div class="text-center">
                    <div class="threshold-icon set1">
                        <i class="fas fa-cog"></i>
                    </div>
                    <h3 class="threshold-title">Threshold Set 1</h3>
                    <p class="threshold-description">
                        Conservative threshold settings optimized for safety-first approach. 
                        This configuration prioritizes patient safety with stricter criteria for decision making.
                    </p>
                    <div class="mt-3">
                        <span class="badge badge-info">Conservative</span>
                        <span class="badge badge-success">Safety-First</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="col-md-6">
            <div class="threshold-card set2" onclick="selectThreshold('threshold_set_2')">
                <div class="text-center">
                    <div class="threshold-icon set2">
                        <i class="fas fa-sliders-h"></i>
                    </div>
                    <h3 class="threshold-title">Threshold Set 2</h3>
                    <p class="threshold-description">
                        Balanced threshold settings that optimize both safety and efficiency. 
                        This configuration balances patient safety with resource utilization considerations.
                    </p>
                    <div class="mt-3">
                        <span class="badge badge-warning">Balanced</span>
                        <span class="badge badge-primary">Efficient</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="text-center mt-4">
        <a href="{{ url_for('decision_type_selection') }}" class="btn btn-secondary btn-lg">
            <i class="fas fa-arrow-left"></i> Back to Decision Type
        </a>
    </div>
</div>

<script>
function selectThreshold(thresholdSet) {
    window.location.href = '/model_selection/{{ decision_type }}/' + thresholdSet;
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    """Welcome page where doctors choose whether to use the decision support tool"""
    # Load image and convert to base64 for HTML rendering
    img = Image.open(os.path.join(PROJECT_ROOT, 'image', 'ConMED-RL Logo.png'))
    
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

    return render_template_string(WELCOME_HTML_TEMPLATE, img_data = img_str)

@app.route('/decision_type_selection')
def decision_type_selection():
    """Decision type selection page"""
    return render_template_string(DECISION_TYPE_HTML_TEMPLATE)

@app.route('/threshold_selection/<decision_type>')
def threshold_selection(decision_type):
    """Threshold setting selection page"""
    if decision_type not in ['discharge', 'extubation']:
        flash("Invalid decision type", "error")
        return redirect(url_for('decision_type_selection'))
    
    return render_template_string(THRESHOLD_SELECTION_HTML_TEMPLATE, decision_type=decision_type)

@app.route('/model_selection/<decision_type>/<threshold_set>', methods = ['GET', 'POST'])
def model_selection(decision_type, threshold_set):
    """Model selection page with decision type and threshold setting"""
    if decision_type not in ['discharge', 'extubation']:
        flash("Invalid decision type", "error")
        return redirect(url_for('decision_type_selection'))
    
    if threshold_set not in ['threshold_set_1', 'threshold_set_2']:
        flash("Invalid threshold setting", "error")
        return redirect(url_for('threshold_selection', decision_type=decision_type))
    
    model_files = get_model_files(decision_type, threshold_set)
    
    if request.method == 'POST':
        model_index = int(request.form.get('model_selection')) - 1
        if 0 <= model_index < len(model_files):
            return redirect(url_for('predict', decision_type=decision_type, threshold_set=threshold_set, model_index=model_index))
        else:
            flash("Invalid model selection", "error")

    # Load image and convert to base64 for HTML rendering
    img = Image.open(os.path.join(PROJECT_ROOT, 'image', 'ConMED-RL Logo.png'))
    
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

    # Update INDEX_HTML_TEMPLATE context
    return render_template_string(INDEX_HTML_TEMPLATE, 
                                model_files=len(model_files), 
                                img_data=img_str, 
                                decision_type=decision_type,
                                threshold_set=threshold_set)

@app.route('/decline')
def decline():
    """Page shown when doctor declines to use the tool"""
    decline_html = """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <title>Thank You for Your Consideration</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
        <style>
            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
            }
            .decline-card {
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                padding: 40px;
                text-align: center;
                max-width: 600px;
                margin: 0 auto;
            }
            .decline-icon {
                font-size: 4rem;
                color: #6c757d;
                margin-bottom: 20px;
            }
            .decline-title {
                color: #333;
                font-size: 2rem;
                font-weight: bold;
                margin-bottom: 20px;
            }
            .decline-message {
                color: #666;
                font-size: 1.1rem;
                line-height: 1.6;
                margin-bottom: 30px;
            }
            .btn-home {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                padding: 12px 30px;
                font-size: 1.1rem;
                font-weight: bold;
                border-radius: 50px;
                transition: all 0.3s ease;
            }
            .btn-home:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
        </style>
    </head>
    <body>
    <div class="container">
        <div class="decline-card">
            <div class="decline-icon">
                <i class="fas fa-hand-paper"></i>
            </div>
            <h1 class="decline-title">Thank You for Your Consideration</h1>
            <div class="decline-message">
                <p>We understand your decision. This decision support system will always be available for your use.</p>
                <p>If you need to use this tool in the future, please feel free to access our system anytime.</p>
                <p>We wish you well in your work!</p>
            </div>
            <a href="{{ url_for('index') }}" class="btn btn-primary btn-home">
                <i class="fas fa-home"></i> Return to Home
            </a>
        </div>
    </div>
    </body>
    </html>
    """
    return render_template_string(decline_html)

@app.route('/predict/<decision_type>/<threshold_set>/<int:model_index>', methods = ['GET', 'POST'])
def predict(decision_type, threshold_set, model_index):
    model_files = get_model_files(decision_type, threshold_set)
    if model_index < 0 or model_index >= len(model_files):
        flash("Invalid model selection", "error")
        return redirect(url_for('model_selection', decision_type = decision_type, threshold_set = threshold_set))
    
    model = load_model(model_files[model_index])
    
    # Load the pre-trained scaler from training phase
    # Note: This loads the appropriate scaler type based on decision type
    # - Discharge: MinMaxScaler
    # - Extubation: StandardScaler
    scaler, feature_names = load_scaler(decision_type)
    if scaler is None:
        flash("Error: Could not load the trained scaler. Please check if the scaler file exists.", "error")
        return redirect(url_for('model_selection', decision_type = decision_type, threshold_set = threshold_set))
    
    # Get patient data labels based on decision type
    labels = get_patient_labels(decision_type)

    if request.method == 'POST':
        try:
            expected_num_vars = len(labels)
            input_values = [float(request.form.get(f"input_{i}")) for i in range(expected_num_vars)]
            if len(input_values) != expected_num_vars:
                flash(f"Please enter all {expected_num_vars} variables for {decision_type} decision making.", "error")
                return redirect(url_for('predict', decision_type = decision_type, threshold_set = threshold_set, model_index = model_index))

            # Process features based on decision type
            # Important: Gender (index 1) should not be scaled for both decision types
            # For discharge: also exclude Readmission Count (index 36)
            # For extubation: no readmission count field
            all_features = np.array(input_values).reshape(1, -1)
            
            # Apply scaler to features except categorical variables
            # The scaler type depends on the decision type:
            # - Discharge: MinMaxScaler (scales to [0,1] range)
            # - Extubation: StandardScaler (standardizes to mean=0, std=1)
            print('📊 Starting data processing...')
            print(f"📊 Input values length: {len(input_values)}")
            print(f"📊 All features shape: {all_features.shape}")
            
            if decision_type == 'discharge':
                print('🏥 Processing DISCHARGE decision...')
                # For discharge: exclude gender (index 1) and readmission count (index 36)
                features_to_scale = np.concatenate([
                    all_features[:, :1],      # age (index 0)
                    all_features[:, 2:36]     # indices 2-35 (medical indicators)
                ], axis = 1)
                print(f"📊 Features to scale shape: {features_to_scale.shape}")
                
                scaled_features = scaler.transform(features_to_scale)
                print(f"📊 Scaled features shape: {scaled_features.shape}")
                
                # Reconstruct the final input with gender and readmission count in correct positions
                # Insert gender at position 1 (unscaled)
                temp_input = np.insert(scaled_features, 1, input_values[1], axis = 1)
                print(f"📊 After inserting gender shape: {temp_input.shape}")
                
                # Insert readmission count at position 36 (unscaled)
                final_input = np.insert(temp_input, 36, input_values[36], axis = 1)
                print(f"📊 Final input shape: {final_input.shape}")
                
            elif decision_type == 'extubation':
                print('🫁 Processing EXTUBATION decision...')
                # For extubation: exclude only gender (index 1), no readmission count
                features_to_scale = np.concatenate([
                    all_features[:, :1],      # age (index 0)
                    all_features[:, 2:]       # indices 2-29 (medical indicators)
                ], axis = 1)
                print(f"📊 Features to scale shape: {features_to_scale.shape}")
                
                scaled_features = scaler.transform(features_to_scale)
                print(f"📊 Scaled features shape: {scaled_features.shape}")
                
                # Reconstruct the final input with gender in correct position
                # Insert gender at position 1 (unscaled)
                final_input = np.insert(scaled_features, 1, input_values[1], axis = 1)
                print(f"📊 Final input shape: {final_input.shape}")
            else:
                print("Here is the error")
                raise ValueError(f"Invalid decision type: {decision_type}")

            # Convert input to torch tensor
            print('🔢 Converting to PyTorch tensor...')
            final_input_tensor = torch.tensor(final_input, dtype = torch.float).to(device)
            print(f"🔢 Input tensor shape: {final_input_tensor.shape}")
            print(f"🔢 Input tensor device: {final_input_tensor.device}")

            # Use the model to make predictions and return the result
            print('🧠 Starting model prediction...')
            with torch.no_grad():
                response = model.avg_Q_value_est(final_input_tensor)
                print(f"🧠 Raw model response: {response}")
                print(f"🧠 Response type: {type(response)}")
                
                # Convert tensor to float if needed
                if torch.is_tensor(response):
                    response = response.item()
                    print(f"🧠 Converted response: {response}")
            print('✅ Model prediction completed!')

            print(f"DEBUG: Prediction response: {response}")
            
            # Get model description based on decision type and model index
            if decision_type == 'discharge':
                model_descriptions = [
                    "Mortality Risk (OBJ)",
                    "Readmission Risk (CON_RR)", 
                    "Length-of-Stay (CON_LOS)"
                ]
            else:  # extubation
                model_descriptions = [
                    "Extubation Failure Risk (OBJ)",
                    "Length-of-Stay (CON)"
                ]
            
            model_desc = model_descriptions[model_index]
            
            # Add interpretation context
            if decision_type == 'discharge':
                if model_index == 0:  # OBJ - Mortality Risk
                    interpretation = "Lower values indicate lower mortality risk following discharge."
                    risk_level = "Low Risk" if response < 0.3 else "Moderate Risk" if response < 0.7 else "High Risk"
                elif model_index == 1:  # CON_RR - Readmission Risk
                    interpretation = "Lower values indicate lower risk of readmission to ICU."
                    risk_level = "Low Risk" if response < 0.2 else "Moderate Risk" if response < 0.5 else "High Risk"
                elif model_index == 2:  # CON_LOS - Length-of-Stay
                    interpretation = "Values represent expected length of stay in hospital (days)."
                    risk_level = f"Expected: {response:.1f} days"
            else:  # extubation
                if model_index == 0:  # OBJ - Extubation Failure Risk
                    interpretation = "Lower values indicate lower risk of extubation failure."
                    risk_level = "Low Risk" if response < 0.2 else "Moderate Risk" if response < 0.5 else "High Risk"
                elif model_index == 1:  # CON - Length-of-Stay
                    interpretation = "Values represent expected length of stay in ICU (days)."
                    risk_level = f"Expected: {response:.1f} days"
            
            print(f"DEBUG: Model description: {model_desc}")
            print(f"DEBUG: Risk level: {risk_level}")
            
            # Store results in session and redirect to results page
            session['prediction_results'] = {
                'prediction_value': response,
                'model_description': model_desc,
                'interpretation': interpretation,
                'risk_level': risk_level,
                'scaler_type': "MinMaxScaler" if decision_type == 'discharge' else "StandardScaler"
            }
            
            print(f"DEBUG: Session data stored: {session.get('prediction_results')}")
            
            return redirect(url_for('show_results', 
                                  decision_type=decision_type, 
                                  threshold_set=threshold_set, 
                                  model_index=model_index))
        except ValueError as ve:
            print(f"❌ ValueError: {ve}")
            flash(f"Invalid input: {ve}", "error")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            flash(f"An error occurred during prediction: {e}", "error")

    # Get field configurations and form sections based on decision type
    field_configs = get_field_configs(decision_type)
    form_sections = get_form_sections(decision_type)
    
    return render_template_string(PREDICT_HTML_TEMPLATE, 
                                  labels = labels, 
                                  model_index = model_index, 
                                  decision_type = decision_type,
                                  threshold_set = threshold_set,
                                  field_configs = field_configs,
                                  form_sections = form_sections)

@app.route('/results/<decision_type>/<threshold_set>/<int:model_index>')
def show_results(decision_type, threshold_set, model_index):
    """Show prediction results page"""
    # Get results from session
    results = session.get('prediction_results', {})
    
    print(f"DEBUG: Retrieved session data: {results}")
    print(f"DEBUG: Session keys: {list(session.keys())}")
    
    if not results:
        print("DEBUG: No results found in session!")
        flash("No prediction results found. Please run the analysis again.", "error")
        return redirect(url_for('predict', 
                              decision_type=decision_type, 
                              threshold_set=threshold_set, 
                              model_index=model_index))
    
    # Don't clear the session data immediately - keep it for debugging
    # session.pop('prediction_results', None)
    
    return render_template_string(RESULTS_HTML_TEMPLATE,
                                decision_type=decision_type,
                                threshold_set=threshold_set,
                                model_index=model_index,
                                prediction_value=results.get('prediction_value', 0),
                                model_description=results.get('model_description', ''),
                                interpretation=results.get('interpretation', ''),
                                risk_level=results.get('risk_level', ''),
                                scaler_type=results.get('scaler_type', ''))

if __name__ == '__main__':
    # Development environment run
    app.run(host = '0.0.0.0', port = 5000, debug = False)
    
# Gunicorn entry point
def create_app():
    return app
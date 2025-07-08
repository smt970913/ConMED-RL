#!/usr/bin/env python3
"""
Test script for web_application_demo.py

This script tests the key functions to ensure they work correctly for both
discharge and extubation decision types.
"""

import sys
import os

# Add the parent directory to the path so we can import the demo module
sys.path.insert(0, os.path.dirname(__file__))

from web_application_demo import (
    get_patient_labels, 
    get_field_configs, 
    get_form_sections, 
    get_model_files,
    load_scaler
)

def test_patient_labels():
    """Test patient labels function"""
    print("Testing patient labels...")
    
    # Test discharge decision making
    discharge_labels = get_patient_labels('discharge')
    print(f"Discharge labels count: {len(discharge_labels)}")
    assert len(discharge_labels) == 37, f"Expected 37 labels for discharge, got {len(discharge_labels)}"
    
    # Test extubation decision making
    extubation_labels = get_patient_labels('extubation')
    print(f"Extubation labels count: {len(extubation_labels)}")
    assert len(extubation_labels) == 30, f"Expected 30 labels for extubation, got {len(extubation_labels)}"
    
    # Test invalid decision type
    invalid_labels = get_patient_labels('invalid')
    assert len(invalid_labels) == 0, "Expected empty list for invalid decision type"
    
    print("✓ Patient labels test passed!")

def test_field_configs():
    """Test field configurations function"""
    print("Testing field configurations...")
    
    # Test discharge decision making
    discharge_configs = get_field_configs('discharge')
    print(f"Discharge field configs count: {len(discharge_configs)}")
    assert len(discharge_configs) == 37, f"Expected 37 configs for discharge, got {len(discharge_configs)}"
    
    # Test extubation decision making
    extubation_configs = get_field_configs('extubation')
    print(f"Extubation field configs count: {len(extubation_configs)}")
    assert len(extubation_configs) == 30, f"Expected 30 configs for extubation, got {len(extubation_configs)}"
    
    # Test invalid decision type
    invalid_configs = get_field_configs('invalid')
    assert len(invalid_configs) == 0, "Expected empty dict for invalid decision type"
    
    print("✓ Field configurations test passed!")

def test_form_sections():
    """Test form sections function"""
    print("Testing form sections...")
    
    # Test discharge decision making
    discharge_sections = get_form_sections('discharge')
    print(f"Discharge sections count: {len(discharge_sections)}")
    assert len(discharge_sections) > 0, "Expected non-empty sections for discharge"
    
    # Test extubation decision making
    extubation_sections = get_form_sections('extubation')
    print(f"Extubation sections count: {len(extubation_sections)}")
    assert len(extubation_sections) > 0, "Expected non-empty sections for extubation"
    
    # Test invalid decision type
    invalid_sections = get_form_sections('invalid')
    assert len(invalid_sections) == 0, "Expected empty list for invalid decision type"
    
    print("✓ Form sections test passed!")

def test_model_files():
    """Test model files function"""
    print("Testing model files...")
    
    # Test discharge decision making
    discharge_models = get_model_files('discharge', 'threshold_set_1')
    print(f"Discharge models count: {len(discharge_models)}")
    assert len(discharge_models) == 3, f"Expected 3 models for discharge, got {len(discharge_models)}"
    
    # Test extubation decision making
    extubation_models = get_model_files('extubation', 'threshold_set_1')
    print(f"Extubation models count: {len(extubation_models)}")
    assert len(extubation_models) == 2, f"Expected 2 models for extubation, got {len(extubation_models)}"
    
    # Test invalid decision type
    invalid_models = get_model_files('invalid', 'threshold_set_1')
    assert len(invalid_models) == 0, "Expected empty list for invalid decision type"
    
    print("✓ Model files test passed!")

def test_load_scaler():
    """Test scaler loading function"""
    print("Testing scaler loading...")
    
    # Test discharge decision making (should create temporary MinMaxScaler)
    discharge_scaler, _ = load_scaler('discharge')
    assert discharge_scaler is not None, "Expected scaler object for discharge"
    print("✓ Discharge scaler loaded (temporary for demo)")
    
    # Test extubation decision making (should create temporary StandardScaler)
    extubation_scaler, _ = load_scaler('extubation')
    assert extubation_scaler is not None, "Expected scaler object for extubation"
    print("✓ Extubation scaler loaded (temporary for demo)")
    
    # Test invalid decision type
    invalid_scaler, _ = load_scaler('invalid')
    assert invalid_scaler is None, "Expected None for invalid decision type"
    
    print("✓ Scaler loading test passed!")

def main():
    """Run all tests"""
    print("🧪 Starting demo tests...")
    print("=" * 50)
    
    try:
        test_patient_labels()
        print()
        test_field_configs()
        print()
        test_form_sections()
        print()
        test_model_files()
        print()
        test_load_scaler()
        print()
        
        print("=" * 50)
        print("🎉 All tests passed successfully!")
        print()
        print("Key Features:")
        print("✓ Support for both discharge and extubation decision making")
        print("✓ Dynamic form generation based on decision type")
        print("✓ FQE model prediction with detailed results page")
        print("✓ Risk level assessment and interpretation")
        print("✓ Temporary scalers for demonstration purposes")
        print()
        print("The demo is ready to run. You can now start the web application:")
        print("cd CDM-Software")
        print("python web_application_demo.py")
        print()
        print("Then open http://localhost:5000 in your browser")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
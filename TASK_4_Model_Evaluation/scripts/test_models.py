"""
Test script to check if model files can be opened
"""
import pickle
import os

model_files = [
    'models/random_forest_fifa_2026.pkl',
    'models/logistic_regression_fifa_2026.pkl',
    'models/model_results.pkl'
]

print(" Testing Model Files...\n")
print("=" * 60)

for model_path in model_files:
    print(f"\n Testing: {model_path}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"    File not found!")
        continue
    
    # Check file size
    file_size = os.path.getsize(model_path)
    print(f"    File size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
    
    # Try to load the model
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"    Successfully loaded!")
        print(f"    Type: {type(model_data)}")
        
        # Show additional info based on type
        if hasattr(model_data, '__dict__'):
            print(f"    Attributes: {list(model_data.__dict__.keys())[:5]}")
        elif isinstance(model_data, dict):
            print(f"    Keys: {list(model_data.keys())[:5]}")
            
    except Exception as e:
        print(f"    Failed to load: {e}")
        print(f"    Error type: {type(e).__name__}")

print("\n" + "=" * 60)
print("\n Model test complete!")

import pickle
import sklearn

# Print installed scikit-learn version
print("Installed scikit-learn version:", sklearn.__version__)

# Path to your model file
file_path = "Experience_Category.pkl"  # Update with the correct path

try:
    # Load the model
    with open(file_path, "rb") as file:
        model = pickle.load(file)

    # Print model details
    print("Model Type:", type(model))

    # Check if feature names are stored
    if hasattr(model, "feature_names_in_"):
        print("Features used for prediction:", model.feature_names_in_)
    else:
        print("Feature names not stored in the model.")

    # Check expected input shape
    if hasattr(model, "n_features_in_"):
        print("Expected number of input features:", model.n_features_in_)

    # Check model parameters
    if hasattr(model, "get_params"):
        print("Model Parameters:", model.get_params())

except Exception as e:
    print("Error while loading the model:", str(e))

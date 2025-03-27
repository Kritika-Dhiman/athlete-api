import pickle
import catboost

# Load the model
with open("HealthScorePredictor.pkl", "rb") as file:
    model = pickle.load(file)

# Check model type
print("Model Type:", type(model))

# Check if feature names are stored
if hasattr(model, "feature_names_"):
    print("Features used for prediction:", model.feature_names_)
else:
    print("Feature names not stored in the model.")

# Check expected input shape
if hasattr(model, "n_features_in_"):
    print("Expected number of input features:", model.n_features_in_)

# Check model parameters
print("Model Parameters:", model.get_params())

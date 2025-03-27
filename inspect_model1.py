import pickle

# Load the model
with open("WeightPredictor.pkl", "rb") as file:
    model = pickle.load(file)

# Check the type of model
print("Model Type:", type(model))

# Check if feature names are stored
if hasattr(model, "feature_names_in_"):
    print("Features used for prediction:", model.feature_names_in_)
else:
    print("Feature names not stored in the model.")

# Check expected input shape
if hasattr(model, "n_features_in_"):
    print("Expected number of input features:", model.n_features_in_)

# Check model parameters (for DecisionTreeRegressor, RandomForest, etc.)
if hasattr(model, "get_params"):
    print("Model Parameters:", model.get_params())

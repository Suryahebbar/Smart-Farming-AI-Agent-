import pandas as pd
import lightgbm as lgb
import pickle

# Load the data
data = pd.read_csv('Crop_recommendation.csv')  # Ensure this matches your downloaded dataset
X = data.drop('label', axis=1)
y = data['label']

# Train the model
model = lgb.LGBMClassifier()
model.fit(X, y)

# Save the model
with open('crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model trained and saved as crop_model.pkl")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset (facial features + psychological labels)
data = pd.read_csv("facial_features.csv")  # Contains extracted features + labels

X = data.drop(columns=["personality_label"])  # Features
y = data["personality_label"]  # Labels (e.g., Extrovert = 1, Introvert = 0)

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

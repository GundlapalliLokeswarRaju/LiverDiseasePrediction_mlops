import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# load the data set
data = pd.read_csv("data/liver.csv")


# Convert Gender to numeric
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

# Handling missing values if any
data.fillna(data.mean(), inplace= True)


# Split into features and target 
X = data.drop(columns=['Dataset'])

y = data["Dataset"]


# Train-test split

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Scale the features 

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)


# Save the processed data

joblib.dump((X_train, X_test, y_train, y_test, scalar), 'main/models/data.pkl')

print("preprocessing complete. Data saved")
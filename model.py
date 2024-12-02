import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib  # To save and load the model
from sklearn.preprocessing import StandardScaler
# Load dataset
df = pd.read_csv("heart.csv")
features = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 
    'oldpeak', 'slope', 'ca', 'thal'
]
target = 'target'

# Selecting X and y
X = df[features]
y = df[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Initialize the model and train it
dt_classifier = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=42)
dt_classifier.fit(X_train, y_train)

ig = plt.figure(figsize=(15,7))
plot_tree(dt_classifier, feature_names=features, class_names=['No Risk', 'High Risk'], filled=True)
plt.title("Decision Tree Visualisation")
plt.show()
# Save the model
joblib.dump(dt_classifier, 'model.pkl')

# Evaluate the model
y_pred = dt_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Le taux d'erreur est:", 1 - accuracy)  
print(f"Model Accuracy: {accuracy:.2f}")
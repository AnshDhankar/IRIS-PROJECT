
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import joblib


# Load dataset
df = pd.read_csv("IRIS.csv")

# Show first 5 rows
df.head()

# Info
df.info()

# Description
df.describe()

# Check null values
df.isnull().sum()

# Class distribution
print(df.columns)
df['species'].value_counts()

# Pairplot
sns.pairplot(df, hue='species')
plt.show()

# Heatmap
sns.heatmap(df.drop("species", axis=1).corr(), annot=True)
plt.show()

# Features and Target
X = df.drop(['species', 'species'], axis=1)
y = df['species']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

models = {
    "KNN": knn,
    "Logistic Regression": lr,
    "Decision Tree": dt
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    
    print("Model:", name)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("-" * 50)

    joblib.dump(knn, "best_model.pkl")
print("Model saved successfully!")

# Load model
model = joblib.load("best_model.pkl")

# Example input: [sepal_length, sepal_width, petal_length, petal_width]
sample = [[5.1, 3.5, 1.4, 0.2]]

# Scale input (important!)
sample = scaler.transform(sample)

prediction = model.predict(sample)

print("Predicted Species:", prediction[0])



# Accuracy comparison
import matplotlib.pyplot as plt

accuracies = []

for model in models.values():
    y_pred = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

plt.bar(models.keys(), accuracies)
plt.title("Model Accuracy Comparison")
plt.show()



# to save  this project
joblib.dump(knn,"best_model.pkl")
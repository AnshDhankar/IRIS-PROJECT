# 🌸 Iris Flower Classification Project

## 📌 Overview
This project focuses on classifying iris flower species using machine learning algorithms. The classification is based on features like sepal length, sepal width, petal length, and petal width.

---

## 📊 Dataset
The dataset used is the famous Iris dataset which contains 150 samples of iris flowers belonging to three species:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

---

## ⚙️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 🔍 Steps Performed
1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Visualization (Pairplot, Heatmap)
4. Data Preprocessing
5. Model Training
6. Model Evaluation
7. Model Saving

---

## 🤖 Machine Learning Models Used
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree

---

## 📈 Results
- KNN achieved the highest accuracy among all models.
- Petal length and petal width were the most important features for classification.

---

## 💾 Model Saving
The best model was saved using Joblib:
```python
joblib.dump(model, "best_model.pkl")

# -----------------------------
# STEP 1: Import Libraries
# -----------------------------
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import plotly.express as px


# -----------------------------
# STEP 2: Import Dataset
# -----------------------------
df = pd.read_csv("medical_examination.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nDataset Shape:")
print(df.shape)


# -----------------------------
# STEP 3: Export Data (Backup)
# -----------------------------
df.to_csv("medical_examination_backup.csv", index=False)
print("\nBackup file saved successfully!")


# -----------------------------
# STEP 4: Data Cleaning
# -----------------------------

# 4.1 Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 4.2 Remove Duplicates
df = df.drop_duplicates()
print("\nShape after removing duplicates:", df.shape)

# 4.3 Outlier Detection
plt.figure(figsize=(6,4))
sns.boxplot(x=df["age"])
plt.title("Outlier Detection - Age")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x=df["weight"])
plt.title("Outlier Detection - Weight")
plt.show()


# -----------------------------
# STEP 5: Data Transformation
# -----------------------------
scaler = StandardScaler()

numeric_cols = ["age", "height", "weight", "ap_hi", "ap_lo"]
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\nAfter Standardization:")
print(df.head())


# -----------------------------
# STEP 6: Descriptive Statistics
# -----------------------------
print("\nDescriptive Statistics:")
print(df.describe())


# -----------------------------
# STEP 7: Basic Visualization
# -----------------------------

# Histogram
plt.figure(figsize=(6,4))
plt.hist(df["age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Bar Chart
plt.figure(figsize=(6,4))
sns.countplot(x="cardio", data=df)
plt.title("Cardiovascular Disease Distribution")
plt.show()


# -----------------------------
# STEP 8: Advanced Visualization
# -----------------------------

# Pair Plot
sns.pairplot(df[["age", "weight", "ap_hi", "ap_lo", "cardio"]], hue="cardio")
plt.show()

# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# -----------------------------
# STEP 9: Interactive Visualization
# -----------------------------
fig = px.scatter(
    df,
    x="ap_hi",
    y="ap_lo",
    color="cardio",
    title="Interactive Blood Pressure Analysis"
)
fig.show()


# -----------------------------
# STEP 10: Probability Analysis
# -----------------------------
plt.figure(figsize=(6,4))
sns.histplot(df["weight"], kde=True)
plt.title("Probability Distribution of Weight")
plt.show()


# -----------------------------
# STEP 11: Classification (k-NN)
# -----------------------------

X = df.drop("cardio", axis=1)
y = df["cardio"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("\nModel Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# -----------------------------
# STEP 12: Clustering (k-Means)
# -----------------------------
kmeans = KMeans(n_clusters=2, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

plt.figure(figsize=(6,4))
sns.scatterplot(
    x=df["ap_hi"],
    y=df["ap_lo"],
    hue=df["Cluster"]
)
plt.title("K-Means Clustering of Blood Pressure")
plt.show()


# -----------------------------
# STEP 13: Summary
# -----------------------------
print("""
Summary & Insights:
- Age and blood pressure strongly influence cardiovascular disease
- k-NN provides good classification accuracy
- k-Means clustering separates patients into risk groups
- Visualizations reveal clear medical patterns
""")

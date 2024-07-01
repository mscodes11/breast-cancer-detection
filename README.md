# **Breast Cancer Detection**

#Breast cancer detection is crucial for early intervention and improving survival rates. The provided code utilizes machine learning techniques to analyze a breast cancer dataset, aiming to predict diagnosis based on various features. It facilitates comprehensive data preprocessing, model training, evaluation, and result interpretation to support effective healthcare decision-making.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
sns.set_style("white")

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier



from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# **Load the dataset**

We can also download the dataset used in the ML model from the following link:
https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset


df = pd.read_csv('/content/breast-cancer.csv')

# Display the first few rows of the DataFrame
print(df.head())


## **Data Preparation**

df.isna().sum()

X = df.drop('diagnosis', axis='columns')
y = df.diagnosis

print(X.shape)
print(y.shape)

X.info()

X.describe()

# **EDA**

sns.countplot(df['diagnosis'],label='count')

y.value_counts()

y_labels = y.map({'M': 'Malignant', 'B': 'Benign'})

# Plot KDE for each feature
for column in X.columns:
    plt.figure(figsize=(8, 6))
    sns.kdeplot(X.loc[y == 'M', column], label='Malignant', shade=True)
    sns.kdeplot(X.loc[y == 'B', column], label='Benign', shade=True)
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.title(f'KDE Plot for {column}')
    plt.legend()
    plt.show()

plt.figure(figsize=(20, 16))
sns.heatmap(X.corr(), annot=True, cmap="YlGnBu")

# **Data Set Training**


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)

# **PCA**
Principal Component Analysis (PCA) is a dimensionality reduction technique used to transform high-dimensional data into a lower-dimensional space while retaining as much variance as possible. 

y_numeric = y.map({'M': 0, 'B': 1})  # Map 'M' (Malignant) to 0 and 'B' (Benign) to 1

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Components')
plt.grid()
plt.show()

# Visualize principal components
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_numeric, cmap='viridis', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Breast Cancer Dataset')
plt.colorbar(scatter, label='Diagnosis', ticks=[0, 1])
plt.grid()
plt.show()

# **KNN**
K-Nearest Neighbors (KNN) is a simple yet effective supervised learning algorithm used for classification and regression tasks. It predicts the classification of a new data point based on majority voting among its k nearest neighbors in the feature space.

# Initialize KNN model
y_numeric = y.map({'M': 0, 'B': 1})  # Malignant (M) -> 0, Benign (B) -> 1

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize k-NN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # Initialize with k=5, you can tune this hyperparameter

# Train the k-NN classifier
knn.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = knn.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification report (precision, recall, F1-score)
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

# **Logistic Regression**
Logistic Regression is a statistical model used for binary classification tasks, predicting the probability of an outcome based on input variables. It estimates the probability using a logistic function, making it suitable for interpreting relationships between independent and dependent variables in healthcare and social sciences.

# Initialize logistic regression model
logreg = LogisticRegression(max_iter=1000)

# Train the logistic regression model
logreg.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = logreg.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification report (precision, recall, F1-score)
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

# Training score
train_score = logreg.score(X_train_scaled, y_train)
print(f'Training Score: {train_score:.2f}')

# Cross-validation score
cv_scores = cross_val_score(logreg, X_train_scaled, y_train, cv=5)
print(f'Cross-Validation Scores: {cv_scores}')
print(f'CV Mean Score: {np.mean(cv_scores):.2f}')


# **Random Forest Classifier**
Random Forest Classifier is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees.

# Initialize Random Forest Classifier model
rf_clf = RandomForestClassifier(random_state=42)

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
rf_grid = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)

# Fit GridSearchCV
rf_grid.fit(X_train, y_train)

# Print best parameters found by GridSearchCV
print('Best Parameters:', rf_grid.best_params_)

# Get the best model
best_rf_model = rf_grid.best_estimator_

# Training Score
train_score = best_rf_model.score(X_train, y_train)
print(f'Training Score: {train_score:.4f}')

# Cross-validation scores
cv_scores = rf_grid.best_score_
print(f'CV Score: {cv_scores:.4f}')

# Test score
test_score = best_rf_model.score(X_test, y_test)
print(f'Test Score: {test_score:.4f}')

# Predict on the test set
y_pred = best_rf_model.predict(X_test)

# Classification report (precision, recall, F1-score)
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Score: {accuracy:.4f}')


# **SVC**
Support Vector Classifier (SVC) is a supervised machine learning algorithm used for both classification and regression tasks. It finds an optimal hyperplane that best separates different classes in the feature space by maximizing the margin between them, making it effective for complex datasets with non-linear boundaries.

# Initialize SVC model
svc = SVC(kernel='linear', C=0.001)  
svc.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = svc.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification report (precision, recall, F1-score)
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

# Cross-validation score
cv_scores = cross_val_score(svc, X_train_scaled, y_train, cv=5)
print(f'Cross-Validation Scores: {cv_scores}')
print(f'CV Mean Score: {np.mean(cv_scores):.2f}')

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Plot the decision boundary
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Breast Cancer Dataset with SVM Decision Boundary')
plt.colorbar(scatter, label='Diagnosis', ticks=[0, 1])
plt.grid()

# Train SVC on PCA-transformed data for visualization purposes
svc_pca = SVC(kernel='linear', C=0.01)
svc_pca.fit(X_pca, y_train)

# Plot the hyperplane
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svc_pca.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.show()

# **Evaluation**

# Initialize and train models
models = {
    'knn': KNeighborsClassifier(n_neighbors=5),
    'logreg': LogisticRegression(max_iter=1000),
    'rf': RandomForestClassifier(random_state=42),
    'svc': SVC(kernel='linear', C=1, probability=True)
}

# Dictionary to store evaluation metrics
evaluation_results = {}

# Train each model and collect evaluation metrics
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'], output_dict=True)
    precision = class_report['Benign']['precision']
    recall = class_report['Benign']['recall']
    f1_score = class_report['Benign']['f1-score']
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = model.decision_function(X_test_scaled)
        
    roc_auc = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    evaluation_results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'roc_auc': roc_auc,
        'cv_mean_score': np.mean(cv_scores)
    }

# Print evaluation results for individual models
for model_name, metrics in evaluation_results.items():
    print(f"Model: {model_name.upper()}")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print("\n")

# Find the best model based on a chosen metric (e.g., accuracy)
best_model = max(evaluation_results, key=lambda x: evaluation_results[x]['accuracy'])
print(f"The best model based on accuracy is: {best_model.upper()}")

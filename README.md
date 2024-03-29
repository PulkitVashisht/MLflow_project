import mlflow
import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

# Load and preprocess data
def load_data(file_path):
    return pd.read_csv(file_path)

# Define MLflow experiment name
mlflow.set_experiment("Sentiment_Analysis")

# Split data into features and target
X = df['Review text']
y = df['Sentiment']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models and their pipelines
pipelines = {
     'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, lowercase=True)),
        ('scaler', MaxAbsScaler()),  # Use MaxAbsScaler instead of StandardScaler
        ('classifier', LogisticRegression())
    ]),
    'Multinomial Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, lowercase=True)),
        ('classifier', MultinomialNB())
    ]),
    'Support Vector Machine': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, lowercase=True)),
        ('scaler', MinMaxScaler()),
        ('classifier', SVC())
    ]),
    'Random Forest': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, lowercase=True)),
        ('classifier', RandomForestClassifier())
    ]),
    'K-Nearest Neighbors': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, lowercase=True)),
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier())
    ])
}

from sklearn.preprocessing import MaxAbsScaler

# Define hyperparameters grids for each model
hyperparameter_grids = {
    'Logistic Regression': {
        'scaler': [None, MaxAbsScaler()],  # Change to MaxAbsScaler
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__solver': ['lbfgs', 'liblinear'],
        'classifier__max_iter': [100, 200, 300]
    },
    'Multinomial Naive Bayes': {
        'classifier__alpha': [0.1, 1.0, 10.0]
    },
    'Support Vector Machine': {
        'scaler': [None, MaxAbsScaler()],  # Change to MaxAbsScaler
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__kernel': ['linear', 'rbf']
    },
    'Random Forest': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20]
    },
    'K-Nearest Neighbors': {
        'scaler': [None, MaxAbsScaler()],  # Change to MaxAbsScaler
        'classifier__n_neighbors': [3, 5, 10],
        'classifier__weights': ['uniform', 'distance']
    }
}

# Check if there's an active MLflow run, and end it if it exists
if mlflow.active_run():
    mlflow.end_run()

# Start MLflow run
mlflow.start_run()

# Iterate over each model and its pipeline
for model_name, pipeline in pipelines.items():
    # Start MLflow run for the current model
    with mlflow.start_run(run_name=model_name, nested=True):  # Start a nested run
        # Set up grid search with cross-validation
        grid_search = GridSearchCV(pipeline, hyperparameter_grids[model_name], cv=3, scoring='f1_weighted', error_score='raise')
        
        # Fit the grid search to find the best parameters
        try:
            grid_search.fit(X_train, y_train)
        except ValueError as e:
            print(f"Fitting failed for {model_name}: {e}")
            continue
        
        # Get the best estimator from the grid search
        best_estimator = grid_search.best_estimator_
        
        # Log the best hyperparameters
        for key, value in grid_search.best_params_.items():
            mlflow.log_param(key, value)
        
        # Fit the best estimator on the entire training set
        start_time = time.time()
        best_estimator.fit(X_train, y_train)
        fit_time = time.time() - start_time
        
        # Log the fitting time
        mlflow.log_metric("fit_time", fit_time)
        
        # Evaluate the best estimator on the test set
        y_pred = best_estimator.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log evaluation metrics
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", accuracy)

# End MLflow run
mlflow.end_run()

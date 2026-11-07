
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the lapse prediction dataset
df = pd.read_csv("outputs/lapse_dataset.csv")

# Separate input features and target
X = df.drop(columns=["customer_id", "lapsed"])
y = df["lapsed"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Identify numerical and categorical features
numerical_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days"
]

categorical_features = [
    "city",
    "gender",
    "favorite_category"
]

# Create numerical preprocessing pipeline
numerical_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

# Create categorical preprocessing pipeline
categorical_pipeline = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(
            drop="first",
            handle_unknown="ignore"
        ))
    ]
)

# Combine numerical and categorical preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

print("\nPreprocessing pipeline created successfully!")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Baseline model: predict every customer as not lapsed
y_baseline = [0] * len(y_test)

print("\nBASELINE MODEL PERFORMANCE")
print("Accuracy:", accuracy_score(y_test, y_baseline))
print("Precision:", precision_score(y_test, y_baseline, zero_division=0))
print("Recall:", recall_score(y_test, y_baseline, zero_division=0))
print("F1 Score:", f1_score(y_test, y_baseline, zero_division=0))

# -------------------------------
# LOGISTIC REGRESSION
# -------------------------------

# Create Logistic Regression pipeline
lr_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# Define Logistic Regression hyperparameters
lr_param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100]
}

# Configure GridSearchCV
lr_grid = GridSearchCV(
    estimator=lr_pipeline,
    param_grid=lr_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("\nLogistic Regression pipeline created!")

# Train and tune Logistic Regression
lr_grid.fit(X_train, y_train)

print("\nLOGISTIC REGRESSION RESULTS")
print("Best Parameters:", lr_grid.best_params_)
print("Best Cross-Validation F1 Score:", lr_grid.best_score_)

# -------------------------------
# RANDOM FOREST
# -------------------------------

# Create Random Forest pipeline
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]
)

# Define Random Forest hyperparameters
rf_param_grid = {
    "classifier__max_depth": [3, 5, 8],
    "classifier__min_samples_leaf": [5, 10, 20]
}

# Configure GridSearchCV
rf_grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("\nRandom Forest pipeline created!")

# Train and tune Random Forest
rf_grid.fit(X_train, y_train)

print("\nRANDOM FOREST RESULTS")
print("Best Parameters:", rf_grid.best_params_)
print("Best Cross-Validation F1 Score:", rf_grid.best_score_)

# -------------------------------
# TEST SET PREDICTIONS
# -------------------------------

# Get predictions from the best Logistic Regression pipeline
lr_best_model = lr_grid.best_estimator_
lr_predictions = lr_best_model.predict(X_test)

# Get predictions from the best Random Forest pipeline
rf_best_model = rf_grid.best_estimator_
rf_predictions = rf_best_model.predict(X_test)

print("\nPredictions completed for both models!")

# -------------------------------
# MODEL EVALUATION
# -------------------------------

print("\nLOGISTIC REGRESSION TEST PERFORMANCE")
print("Accuracy:", accuracy_score(y_test, lr_predictions))
print("Precision:", precision_score(y_test, lr_predictions, zero_division=0))
print("Recall:", recall_score(y_test, lr_predictions, zero_division=0))
print("F1 Score:", f1_score(y_test, lr_predictions, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_predictions))

print("\nRANDOM FOREST TEST PERFORMANCE")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Precision:", precision_score(y_test, rf_predictions, zero_division=0))
print("Recall:", recall_score(y_test, rf_predictions, zero_division=0))
print("F1 Score:", f1_score(y_test, rf_predictions, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

# -------------------------------
# MODEL COMPARISON
# -------------------------------

comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, lr_predictions),
        accuracy_score(y_test, rf_predictions)
    ],
    "Precision": [
        precision_score(y_test, lr_predictions, zero_division=0),
        precision_score(y_test, rf_predictions, zero_division=0)
    ],
    "Recall": [
        recall_score(y_test, lr_predictions, zero_division=0),
        recall_score(y_test, rf_predictions, zero_division=0)
    ],
    "F1 Score": [
        f1_score(y_test, lr_predictions, zero_division=0),
        f1_score(y_test, rf_predictions, zero_division=0)
    ]
})

print("\nMODEL COMPARISON")
print(comparison.round(3))

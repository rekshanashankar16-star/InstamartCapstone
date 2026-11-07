import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import resample
from scipy.sparse import csr_matrix, vstack
import numpy as np

# Load the lapse prediction dataset
df = pd.read_csv("outputs/lapse_dataset.csv")

# Separate input features and target
X = df.drop(columns=["customer_id", "lapsed"])
y = df["lapsed"]

print("Dataset shape:", df.shape)
print("Input features shape:", X.shape)
print("Target shape:", y.shape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nOriginal training target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# Identify numerical features
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

# Identify categorical features
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

# Combine both preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform testing data using the same fitted preprocessor
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

# Separate non-lapsed customers
X_not_lapsed = X_train_processed[y_train == 0]

# Separate lapsed customers
X_lapsed = X_train_processed[y_train == 1]

print("\nBefore Oversampling:")
print("Not lapsed customers:", X_not_lapsed.shape[0])
print("Lapsed customers:", X_lapsed.shape[0])

# Oversample lapsed customers with replacement
X_lapsed_oversampled = resample(
    X_lapsed,
    replace=True,
    n_samples=X_not_lapsed.shape[0] - X_lapsed.shape[0],
    random_state=42
)

print("\nOversampled lapsed customers:", X_lapsed_oversampled.shape[0])


# Convert datasets into sparse matrices
X_not_lapsed = csr_matrix(X_not_lapsed)
X_lapsed = csr_matrix(X_lapsed)
X_lapsed_oversampled = csr_matrix(X_lapsed_oversampled)

# Combine original records and additional duplicate records
X_balanced = vstack([
    X_not_lapsed,
    X_lapsed,
    X_lapsed_oversampled
])

# Create matching target labels
y_balanced = np.concatenate([
    np.zeros(X_not_lapsed.shape[0]),
    np.ones(X_lapsed.shape[0]),
    np.ones(X_lapsed_oversampled.shape[0])
])

print("\nAfter Oversampling:")
print("Balanced training shape:", X_balanced.shape)
print("Balanced target distribution:")
print(pd.Series(y_balanced).value_counts())

# Train original Logistic Regression on imbalanced data
original_lr_model = LogisticRegression(
    C=0.1,
    max_iter=1000
)

original_lr_model.fit(X_train_processed, y_train)

# Predict using original model
original_predictions = original_lr_model.predict(X_test_processed)

print("\nOriginal Logistic Regression trained successfully!")

# Create Logistic Regression model
balanced_lr_model = LogisticRegression(
    C=0.1,
    max_iter=1000
)

# Train the model using balanced training data
balanced_lr_model.fit(X_balanced, y_balanced)

print("\nLogistic Regression trained on balanced data successfully!")

# Predict using the balanced Logistic Regression model
balanced_predictions = balanced_lr_model.predict(X_test_processed)

print("\nPredictions on original test data completed!")

# Evaluate the balanced Logistic Regression model
print("\nBALANCED LOGISTIC REGRESSION PERFORMANCE")

print("Precision:", precision_score(
    y_test, balanced_predictions, zero_division=0
))

print("Recall:", recall_score(
    y_test, balanced_predictions, zero_division=0
))

print("F1 Score:", f1_score(
    y_test, balanced_predictions, zero_division=0
))

# Compare Logistic Regression before and after oversampling
comparison = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1 Score"],
    "Before Oversampling": [
        precision_score(y_test, original_predictions, zero_division=0),
        recall_score(y_test, original_predictions, zero_division=0),
        f1_score(y_test, original_predictions, zero_division=0)
    ],
    "After Oversampling": [
        precision_score(y_test, balanced_predictions, zero_division=0),
        recall_score(y_test, balanced_predictions, zero_division=0),
        f1_score(y_test, balanced_predictions, zero_division=0)
    ]
})

print("\nBEFORE VS AFTER OVERSAMPLING")
print(comparison.round(3))
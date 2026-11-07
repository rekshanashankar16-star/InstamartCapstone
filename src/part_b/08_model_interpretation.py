import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import resample
from scipy.sparse import csr_matrix, vstack

import joblib

# Load the lapse prediction dataset
df = pd.read_csv("outputs/lapse_dataset.csv")

# Separate input features and target
X = df.drop(columns=["customer_id", "lapsed"])
y = df["lapsed"]

print("Dataset loaded successfully!")
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

# Store customer IDs belonging to the test dataset
test_customer_ids = df.loc[
    X_test.index, "customer_id"
].reset_index(drop=True)

print("\nTest customer IDs stored:", len(test_customer_ids))

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# Define numerical and categorical features
numeric_features = [
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

print("\nNumerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

# Create numerical preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

# Create categorical preprocessing pipeline
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    ))
])

# Combine both preprocessing pipelines
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Fit on training data and transform both datasets
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing completed!")
print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

# Train the best Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

rf_model.fit(X_train_processed, y_train)

print("\nRandom Forest model trained successfully!")

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Find the five most important features
rf_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_model.feature_importances_
})

rf_importance = rf_importance.sort_values(
    by="importance",
    ascending=False
)

print("\nTOP 5 IMPORTANT FEATURES - RANDOM FOREST")
print(rf_importance.head(5).to_string(index=False))

# Business interpretation of Random Forest features
print("\nBUSINESS INTERPRETATION - RANDOM FOREST")

print("1. Frequency: Customers who order more frequently show different lapse patterns; purchase frequency is the strongest predictor.")

print("2. Monetary: Customer spending contributes strongly to lapse prediction and helps identify valuable customers.")

print("3. Recency: The time since the last purchase helps identify customers who may be becoming inactive.")

print("4. Cancellation rate: Frequent order cancellations provide an additional signal of possible customer disengagement.")

print("5. Average discount percentage: Discount usage contributes to identifying different customer lapse patterns.")

# Train the best Logistic Regression model
lr_model = LogisticRegression(
    C=0.1,
    max_iter=1000
)

lr_model.fit(X_train_processed, y_train)

print("\nLogistic Regression model trained successfully!")

# Get Logistic Regression coefficients
lr_coefficients = pd.DataFrame({
    "feature": feature_names,
    "coefficient": lr_model.coef_[0]
})

# Select the three strongest coefficients by absolute value
lr_coefficients["absolute_coefficient"] = (
    lr_coefficients["coefficient"].abs()
)

top_3_coefficients = lr_coefficients.sort_values(
    by="absolute_coefficient",
    ascending=False
).head(3)

print("\nTOP 3 LOGISTIC REGRESSION COEFFICIENTS")
print(
    top_3_coefficients[
        ["feature", "coefficient"]
    ].to_string(index=False)
)

# Business interpretation of Logistic Regression coefficients
print("\nBUSINESS INTERPRETATION - LOGISTIC REGRESSION")

print("1. Frequency (Negative): Higher purchase frequency is associated with lower customer lapse risk.")

print("2. Electronics (Negative): Customers in the Electronics category have lower predicted lapse odds than customers in the reference category.")

print("3. Snacks (Negative): Customers in the Snacks category have lower predicted lapse odds than customers in the reference category.")

# Separate the two classes from training data
X_not_lapsed = X_train_processed[y_train == 0]
X_lapsed = X_train_processed[y_train == 1]

# Add duplicate lapsed records to balance the classes
X_lapsed_oversampled = resample(
    X_lapsed,
    replace=True,
    n_samples=X_not_lapsed.shape[0] - X_lapsed.shape[0],
    random_state=42
)

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

print("\nBalanced training shape:", X_balanced.shape)
print("Balanced target distribution:", np.unique(
    y_balanced, return_counts=True
))

# Train Logistic Regression on balanced training data
balanced_lr_model = LogisticRegression(
    C=0.1,
    max_iter=1000
)

balanced_lr_model.fit(X_balanced, y_balanced)

print("\nBalanced Logistic Regression trained successfully!")

# Save the trained model
joblib.dump(
    balanced_lr_model,
    "outputs/lapse_model.joblib"
)

print("\nModel saved successfully!")

# Reload the saved model
loaded_model = joblib.load(
    "outputs/lapse_model.joblib"
)

print("Model reloaded successfully!")

# Generate predictions using the original model
original_predictions = balanced_lr_model.predict(
    X_test_processed
)

# Generate predictions using the reloaded model
reloaded_predictions = loaded_model.predict(
    X_test_processed
)

# Compare both predictions
if np.array_equal(original_predictions, reloaded_predictions):
    print("\nSUCCESS: Both models produce exactly the same predictions!")
else:
    print("\nERROR: Predictions do not match!")

# Load customer segment information
segments = pd.read_csv("outputs/customer_segments.csv")

# Identify Premium customers
premium_ids = set(
    segments.loc[
        segments["is_premium"] == True,
        "customer_id"
    ].astype(str)
)

print("\nNumber of Premium customers:", len(premium_ids))

# Predict lapse probabilities for test customers
lapse_probabilities = loaded_model.predict_proba(
    X_test_processed
)[:, 1]

print("\nLapse probabilities calculated successfully!")
print("Number of predictions:", len(lapse_probabilities))

# Create test customer prediction results
test_results = pd.DataFrame({
    "customer_id": test_customer_ids.astype(str).to_numpy(),
    "lapse_probability": lapse_probabilities,
    "actually_lapsed": y_test.to_numpy()
})

print("\nTest customer prediction results:")
print(test_results.head())

# Filter only Premium customers
premium_test_results = test_results[
    test_results["customer_id"].isin(premium_ids)
]

# Rank customers by lapse probability (highest first)
premium_test_results = premium_test_results.sort_values(
    by="lapse_probability",
    ascending=False
)

# Select the top 10 customers
retention_list = premium_test_results.head(10)

print("\nTOP 10 PREMIUM CUSTOMERS AT RISK OF LAPSING")
print(retention_list.to_string(index=False))

# Save the retention list
retention_list.to_csv(
    "outputs/retention_list.csv",
    index=False
)

print("\nRetention list saved successfully!")
print("File location: outputs/retention_list.csv")

# Count customers who actually lapsed
actual_lapsed_count = retention_list["actually_lapsed"].sum()

print(
    "\nNumber of customers who actually lapsed:",
    actual_lapsed_count
)

# Project limitation
print("\nLIMITATION:")
print(
    "Premium customer segments were identified using full-period "
    "RFM data, including September. Therefore, the Premium retention "
    "analysis is post-hoc and is not a leakage-free evaluation."
)
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the lapse prediction dataset
df = pd.read_csv("outputs/lapse_dataset.csv")

# Display dataset information
print("Dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nTarget distribution:")
print(df["lapsed"].value_counts())

# Separate target variable
y = df["lapsed"]

# Remove target and customer ID from input features
X = df.drop(columns=["customer_id", "lapsed"])

print("\nInput Features (X):")
print(X.head())

print("\nTarget Variable (y):")
print(y.head())

print("\nX shape:", X.shape)
print("y shape:", y.shape)

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

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

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

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# Fit preprocessing on training data and transform it
X_train_processed = preprocessor.fit_transform(X_train)

# Transform testing data using the same fitted preprocessor
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training data shape:", X_train_processed.shape)
print("Processed testing data shape:", X_test_processed.shape)

# Get the frequency mean learned from training data
frequency_train_mean = preprocessor.named_transformers_["num"].named_steps["scaler"].mean_[1]

# Calculate frequency mean from the full cohort
frequency_full_mean = X["frequency"].mean()

print("\nFrequency Mean Comparison:")
print("Training data frequency mean:", frequency_train_mean)
print("Full cohort frequency mean:", frequency_full_mean)

# Check for missing values in training and testing data
print("\nMissing values in training data:")
print(X_train.isnull().sum())

print("\nMissing values in testing data:")
print(X_test.isnull().sum())

print("\nTotal missing values in training data:", X_train.isnull().sum().sum())
print("Total missing values in testing data:", X_test.isnull().sum().sum())
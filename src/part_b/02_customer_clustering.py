import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load RFM data
rfm = pd.read_csv("outputs/customer_rfm.csv")

# Select RFM features
features = rfm[["recency", "frequency", "monetary"]]

# Standardise RFM features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

print("Original RFM shape:", features.shape)
print("Scaled RFM shape:", scaled_features.shape)

# Store clustering evaluation results
results = []

# Test K values from 2 to 8
for k in range(2, 9):

    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = kmeans.fit_predict(scaled_features)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(scaled_features, labels)

    results.append({
        "K": k,
        "Inertia": inertia,
        "Silhouette": silhouette
    })

# Convert results into a DataFrame
results_df = pd.DataFrame(results)

print("\nClustering Evaluation Results:")
print(results_df.round(3))

# Save clustering evaluation results
results_df.to_csv("outputs/clustering_evaluation.csv", index=False)

print("\nClustering evaluation saved successfully!")
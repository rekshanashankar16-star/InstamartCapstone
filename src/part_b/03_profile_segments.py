import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load RFM data
rfm = pd.read_csv("outputs/customer_rfm.csv")

# Select RFM features
features = rfm[["recency", "frequency", "monetary"]]

# Standardise features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)



# Apply K-means clustering with K=4
kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = kmeans.fit_predict(scaled_features)

# Display cluster counts
print("\nCustomer count in each segment:")
print(rfm["segment_id"].value_counts().sort_index())

# Calculate segment size and average RFM values
segment_profile = rfm.groupby("segment_id").agg(
    customer_count=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")
).reset_index()

print("\nCustomer Segment Profiles:")
print(segment_profile.round(2))

# Identify the segment with the highest average spending
premium_segment_id = segment_profile.loc[
    segment_profile["mean_monetary"].idxmax(),
    "segment_id"
]

# Create premium indicator
rfm["is_premium"] = (
    rfm["segment_id"] == premium_segment_id
).astype(int)

print("\nPremium Segment ID:", premium_segment_id)

print(
    "Premium Customers:",
    rfm["is_premium"].sum()
)

# Assign business names to customer segments
segment_names = {
    0: "Regular Low-Spend",
    1: "Premium",
    2: "Frequent Active",
    3: "At-Risk"
}

rfm["segment_name"] = rfm["segment_id"].map(segment_names)

# Save the final customer segments file
customer_segments = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]
]

customer_segments.to_csv(
    "outputs/customer_segments.csv",
    index=False
)

print("\nFinal customer segments saved successfully!")
print(customer_segments.head())

# Apply PCA to reduce RFM features to 2 dimensions
pca = PCA(n_components=2)

pca_features = pca.fit_transform(scaled_features)

print("\nPCA Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Explained Variance:")
print(pca.explained_variance_ratio_.sum())

# Save segment profile
segment_profile.to_csv(
    "outputs/segment_profile.csv",
    index=False
)

# Save PCA results
pca_df = pd.DataFrame(
    pca_features,
    columns=["PC1", "PC2"]
)

pca_df["customer_id"] = rfm["customer_id"]
pca_df["segment_id"] = rfm["segment_id"]

pca_df.to_csv("outputs/customer_pca.csv", index=False)

print("\nSegment profile and PCA results saved successfully!")
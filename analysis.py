import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

# Load the CSV file
df = pd.read_csv('cat_colors.csv')

# Select the features for clustering (first 4 columns)
features = ['dominant_color_b', 
                    'dominant_color_g', 'dominant_color_r',
                    'secondary_color_b', 
                    'secondary_color_g', 'secondary_color_r',
                    'tertiary_color_b', 
                    'tertiary_color_g', 'tertiary_color_r', 
                    'color_variation']
X = df[features].copy()

# Handle any missing values
X = X.dropna()

# Standardize the features (important for GMM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)

# Perform GMM clustering with 3 components
gmm = GaussianMixture(n_components=10, random_state=42, covariance_type='full')
gmm.fit(X_scaled)

# Get cluster assignments and probabilities
clusters = gmm.predict(X_scaled)
cluster_probs = gmm.predict_proba(X_scaled)

# Add cluster labels and probabilities to the original dataframe
df_clustered = df.loc[X.index].copy()
df_clustered['cluster'] = clusters
df_clustered['cluster_confidence'] = np.max(cluster_probs, axis=1)

# Add individual cluster probabilities
for i in range(3):
    df_clustered[f'prob_cluster_{i}'] = cluster_probs[:, i]

# Print cluster centers (means) in original scale
cluster_means_original = scaler.inverse_transform(gmm.means_)
print("Cluster Means (original scale):")
for i, mean in enumerate(cluster_means_original):
    print(f"Cluster {i}: B={mean[0]:.2f}, G={mean[1]:.2f}, R={mean[2]:.2f}, Variation={mean[3]:.2f}")

# Print cluster sizes
print("\nCluster sizes:")
print(df_clustered['cluster'].value_counts().sort_index())

# Print average confidence by cluster
print("\nAverage cluster confidence:")
avg_confidence = df_clustered.groupby('cluster')['cluster_confidence'].mean()
for cluster, conf in avg_confidence.items():
    print(f"Cluster {cluster}: {conf:.3f}")

# Display first few rows with cluster assignments and probabilities
print("\nFirst 10 rows with cluster assignments:")
display_cols = features + ['cluster', 'cluster_confidence'] + [f'prob_cluster_{i}' for i in range(3)]
print(df_clustered[display_cols].head(10))

# Create visualizations
plt.figure(figsize=(20, 10))

# Plot 1: RGB color space (R-G)
plt.subplot(2, 4, 1)
scatter = plt.scatter(df_clustered['dominant_color_r'], df_clustered['dominant_color_g'], 
                     c=df_clustered['cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Dominant Color Red')
plt.ylabel('Dominant Color Green')
plt.title('GMM Clusters in R-G Color Space')
plt.colorbar(scatter)

# Plot 2: Color variation vs Blue component
plt.subplot(2, 4, 2)
scatter = plt.scatter(df_clustered['dominant_color_b'], df_clustered['color_variation'], 
                     c=df_clustered['cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Dominant Color Blue')
plt.ylabel('Color Variation')
plt.title('GMM Clusters: Blue vs Color Variation')
plt.colorbar(scatter)

# Plot 3: Cluster distribution
plt.subplot(2, 4, 3)
cluster_counts = df_clustered['cluster'].value_counts().sort_index()
plt.bar(cluster_counts.index, cluster_counts.values, color=['#440154', '#31688e', '#fde725'])
plt.xlabel('Cluster')
plt.ylabel('Number of Data Points')
plt.title('Distribution of Data Points Across Clusters')

# Plot 4: Red vs Color Variation
plt.subplot(2, 4, 4)
scatter = plt.scatter(df_clustered['dominant_color_r'], df_clustered['color_variation'], 
                     c=df_clustered['cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Dominant Color Red')
plt.ylabel('Color Variation')
plt.title('GMM Clusters: Red vs Color Variation')
plt.colorbar(scatter)

# Plot 5: Cluster confidence distribution
plt.subplot(2, 4, 5)
plt.hist(df_clustered['cluster_confidence'], bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('Cluster Confidence')
plt.ylabel('Frequency')
plt.title('Distribution of Cluster Confidence')

# Plot 6: Confidence by cluster
plt.subplot(2, 4, 6)
box_data = [df_clustered[df_clustered['cluster'] == i]['cluster_confidence'] for i in range(3)]
plt.boxplot(box_data, labels=[f'Cluster {i}' for i in range(3)])
plt.ylabel('Cluster Confidence')
plt.title('Cluster Confidence by Cluster')

# Plot 7: 3D RGB space (projected to 2D: B vs G)
plt.subplot(2, 4, 7)
scatter = plt.scatter(df_clustered['dominant_color_b'], df_clustered['dominant_color_g'], 
                     c=df_clustered['cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Dominant Color Blue')
plt.ylabel('Dominant Color Green')
plt.title('GMM Clusters in B-G Color Space')
plt.colorbar(scatter)

# Plot 8: Uncertainty visualization (low confidence points)
plt.subplot(2, 4, 8)
uncertain_points = df_clustered[df_clustered['cluster_confidence'] < 0.6]
if len(uncertain_points) > 0:
    scatter = plt.scatter(uncertain_points['dominant_color_r'], uncertain_points['dominant_color_g'], 
                         c=uncertain_points['cluster_confidence'], cmap='Reds', alpha=0.8)
    plt.xlabel('Dominant Color Red')
    plt.ylabel('Dominant Color Green')
    plt.title('Low Confidence Points (< 0.6)')
    plt.colorbar(scatter, label='Confidence')
else:
    plt.text(0.5, 0.5, 'No low confidence points\n(threshold < 0.6)', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Low Confidence Points (< 0.6)')

plt.tight_layout()
plt.show()

# Calculate and display cluster statistics
print("\nCluster Statistics:")
cluster_stats = df_clustered.groupby('cluster')[features].agg(['mean', 'std'])
print(cluster_stats)

# Print covariance information
print("\nCluster Covariances (in standardized space):")
for i in range(3):
    print(f"\nCluster {i} Covariance Matrix:")
    print(gmm.covariances_[i])

# Optionally, save the clustered data to a new CSV
df_clustered.to_csv('cat_colors_gmm_clustered2.csv', index=False)
print("\nGMM clustered data saved to 'cat_colors_gmm_clustered2.csv'")

# Calculate BIC and AIC for different numbers of components
# This helps evaluate if 3 components is optimal
bic_scores = []
aic_scores = []
log_likelihoods = []
n_components_range = range(1, 11)

for n_comp in n_components_range:
    gmm_temp = GaussianMixture(n_components=n_comp, random_state=42, covariance_type='full')
    gmm_temp.fit(X_scaled)
    bic_scores.append(gmm_temp.bic(X_scaled))
    aic_scores.append(gmm_temp.aic(X_scaled))
    log_likelihoods.append(gmm_temp.score(X_scaled))

# Plot model selection criteria
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(n_components_range, bic_scores, 'bo-', label='BIC')
plt.plot(n_components_range, aic_scores, 'ro-', label='AIC')
plt.xlabel('Number of Components')
plt.ylabel('Information Criterion')
plt.title('BIC and AIC for Model Selection')
plt.axvline(x=3, color='green', linestyle='--', label='n_components=3')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(n_components_range, log_likelihoods, 'go-')
plt.xlabel('Number of Components')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood vs Number of Components')
plt.axvline(x=3, color='red', linestyle='--', label='n_components=3')
plt.legend()
plt.grid(True)

# Plot the improvement in BIC/AIC
plt.subplot(1, 3, 3)
bic_diff = np.diff(bic_scores)
aic_diff = np.diff(aic_scores)
plt.plot(n_components_range[1:], bic_diff, 'bo-', label='BIC Difference')
plt.plot(n_components_range[1:], aic_diff, 'ro-', label='AIC Difference')
plt.xlabel('Number of Components')
plt.ylabel('Change in Information Criterion')
plt.title('Change in BIC/AIC')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=3, color='green', linestyle='--', label='n_components=3')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Find optimal number of components based on BIC
optimal_bic = n_components_range[np.argmin(bic_scores)]
optimal_aic = n_components_range[np.argmin(aic_scores)]
print(f"\nOptimal number of components based on BIC: {optimal_bic}")
print(f"Optimal number of components based on AIC: {optimal_aic}")

# Print GMM model parameters
print(f"\nGMM Model Information:")
print(f"Converged: {gmm.converged_}")
print(f"Number of iterations: {gmm.n_iter_}")
print(f"Log likelihood: {gmm.score(X_scaled):.3f}")
print(f"BIC: {gmm.bic(X_scaled):.3f}")
print(f"AIC: {gmm.aic(X_scaled):.3f}")

# Cluster weights (mixing coefficients)
print(f"\nCluster weights (mixing coefficients):")
for i, weight in enumerate(gmm.weights_):
    print(f"Cluster {i}: {weight:.3f}")


# Create a mapping from cluster numbers to meaningful names
cluster_names = {0: "Terra", 1: "Tot", 2: "Cat3",3:"somethingelse",4:"somethingelse1",5:"somethin3",6:"something45",7:"something445",8:"something4445",9:"something454"}

# Add named clusters to the dataframe
df_clustered['cluster_name'] = df_clustered['cluster'].map(cluster_names)

# Save the complete trained model package
classifier_package = {
    'gmm_model': gmm,
    'scaler': scaler,
    'cluster_names': cluster_names,
    'features': features,
    'model_info': {
        'n_components': 3,
        'covariance_type': 'full',
        'converged': gmm.converged_,
        'n_iter': gmm.n_iter_,
        'log_likelihood': gmm.score(X_scaled),
        'bic': gmm.bic(X_scaled),
        'aic': gmm.aic(X_scaled)
    }
}

# Save the classifier package
with open('cat_color_classifier.pkl', 'wb') as f:
    pickle.dump(classifier_package, f)

print("Classifier package saved to 'cat_color_classifier.pkl'")

# Create a standalone classifier function
def classify_cat_color(dominant_color_b, dominant_color_g, dominant_color_r, color_variation):
    """
    Classify a cat color based on RGB values and color variation.
    
    Parameters:
    -----------
    dominant_color_b : float
        Blue component of dominant color (0-255)
    dominant_color_g : float
        Green component of dominant color (0-255)
    dominant_color_r : float
        Red component of dominant color (0-255)
    color_variation : float
        Color variation metric
    
    Returns:
    --------
    dict : Classification results containing:
        - 'cluster_name': Predicted cluster name
        - 'cluster_id': Predicted cluster ID (0-2)
        - 'confidence': Confidence of the prediction (0-1)
        - 'probabilities': Dict of probabilities for each cluster
    """
    # Load the trained classifier
    with open('cat_color_classifier.pkl', 'rb') as f:
        package = pickle.load(f)
    
    gmm_model = package['gmm_model']
    scaler = package['scaler']
    cluster_names = package['cluster_names']
    
    # Prepare the input data
    input_data = np.array([[dominant_color_b, dominant_color_g, dominant_color_r, color_variation]])
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make predictions
    cluster_id = gmm_model.predict(input_scaled)[0]
    probabilities = gmm_model.predict_proba(input_scaled)[0]
    confidence = np.max(probabilities)
    
    # Format results
    result = {
        'cluster_name': cluster_names[cluster_id],
        'cluster_id': int(cluster_id),
        'confidence': float(confidence),
        'probabilities': {cluster_names[i]: float(prob) for i, prob in enumerate(probabilities)}
    }
    
    return result

# Example usage of the classifier
print("\n" + "="*50)
print("CLASSIFIER USAGE EXAMPLE")
print("="*50)

# Test with a few sample points
test_samples = [
    (100, 120, 140, 25.5),  # Example cat color 1
    (200, 180, 160, 15.2),  # Example cat color 2
    (50, 80, 90, 35.8)      # Example cat color 3
]

for i, (b, g, r, var) in enumerate(test_samples):
    result = classify_cat_color(b, g, r, var)
    print(f"\nSample {i+1}: B={b}, G={g}, R={r}, Variation={var}")
    print(f"Predicted: {result['cluster_name']} (confidence: {result['confidence']:.3f})")
    print(f"Probabilities: {result['probabilities']}")

# Save the cluster statistics for reference
cluster_summary = {}
for cluster_id in range(3):
    cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
    cluster_summary[cluster_names[cluster_id]] = {
        'count': len(cluster_data),
        'mean_values': {
            'dominant_color_b': float(cluster_data['dominant_color_b'].mean()),
            'dominant_color_g': float(cluster_data['dominant_color_g'].mean()),
            'dominant_color_r': float(cluster_data['dominant_color_r'].mean()),
            'color_variation': float(cluster_data['color_variation'].mean())
        },
        'typical_rgb': [
            int(cluster_data['dominant_color_r'].mean()),
            int(cluster_data['dominant_color_g'].mean()),
            int(cluster_data['dominant_color_b'].mean())
        ]
    }

# Save cluster summary as JSON
with open('cluster_summary.json', 'w') as f:
    json.dump(cluster_summary, f, indent=2)

print(f"\nCluster summary saved to 'cluster_summary.json'")

# Print the summary
print("\n" + "="*50)
print("CLUSTER SUMMARY")
print("="*50)
for name, info in cluster_summary.items():
    print(f"\n{name}:")
    print(f"  Count: {info['count']} cats")
    print(f"  Typical RGB: {info['typical_rgb']}")
    print(f"  Average values:")
    for feature, value in info['mean_values'].items():
        print(f"    {feature}: {value:.2f}")

# Create a simple batch classifier function
def classify_batch(csv_file_path, output_file_path=None):
    """
    Classify a batch of cat colors from a CSV file.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to CSV file with columns: dominant_color_b, dominant_color_g, dominant_color_r, color_variation
    output_file_path : str, optional
        Path to save results. If None, returns DataFrame.
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with original data plus classification results
    """
    # Load the data
    df_new = pd.read_csv(csv_file_path)
    
    # Load the trained classifier
    with open('cat_color_classifier.pkl', 'rb') as f:
        package = pickle.load(f)
    
    gmm_model = package['gmm_model']
    scaler = package['scaler']
    cluster_names = package['cluster_names']
    features = package['features']
    
    # Prepare the data
    X_new = df_new[features].dropna()
    X_new_scaled = scaler.transform(X_new)
    
    # Make predictions
    clusters = gmm_model.predict(X_new_scaled)
    probabilities = gmm_model.predict_proba(X_new_scaled)
    
    # Add results to dataframe
    df_results = df_new.loc[X_new.index].copy()
    df_results['predicted_cluster'] = clusters
    df_results['predicted_name'] = [cluster_names[c] for c in clusters]
    df_results['confidence'] = np.max(probabilities, axis=1)
    
    # Add individual probabilities
    for i in range(3):
        df_results[f'prob_{cluster_names[i]}'] = probabilities[:, i]
    
    # Save if output path provided
    if output_file_path:
        df_results.to_csv(output_file_path, index=False)
        print(f"Batch classification results saved to '{output_file_path}'")
    
    return df_results

print("\n" + "="*50)
print("FILES CREATED FOR CLASSIFIER")
print("="*50)
print("1. cat_color_classifier.pkl - Complete trained model package")
print("2. cluster_summary.json - Summary statistics for each cluster")
print("3. cat_colors_gmm_clustered.csv - Original data with cluster assignments")
print("\nFunctions available:")
print("- classify_cat_color(b, g, r, variation) - Classify single cat")
print("- classify_batch(csv_file) - Classify multiple cats from CSV")
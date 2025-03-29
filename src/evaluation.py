from sklearn.metrics import silhouette_score

def get_silhouette_score(features, labels):
    """
    Computes the silhouette score for clustering.

    Parameters:
    - features (np.ndarray): The feature matrix (e.g., TF-IDF, Word2Vec, etc.).
    - labels (np.ndarray): Cluster labels assigned by the clustering algorithm.

    Returns:
    - float: The silhouette score (higher is better, range: -1 to 1).
    """
    return silhouette_score(features, labels)



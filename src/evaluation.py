from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import contingency_matrix
import numpy as np

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



# get the purity of each cluster
def purity_score(labels_true, labels_pred):
    """
    Calculates the purity score for evaluating clustering performance.
    
    """
    # Compute contingency matrix (confusion matrix)
    matrix = contingency_matrix(labels_true, labels_pred)
    # Calculate the purity score
    purity = np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)
    return purity
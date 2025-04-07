from sklearn.decomposition import PCA,LatentDirichletAllocation
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import joblib
import pandas as pd



def apply_pca(features, n_components):
    """
    Applies PCA to reduce dimensionality of the dataset.

    Parameters:
    - features (np.ndarray): The high-dimensional feature matrix >> takes it Dense Not sparse.
    - n_components (int): Number of PCA components (default: 2).

    Returns:
    - pca_result (np.ndarray): Transformed feature representation.
    """
    pca = PCA(n_components=n_components,svd_solver='auto')
    pca_result = pca.fit_transform(features)
    return pca_result



def apply_kmeans(features, n_clusters: int = 3, random_state: int = 42):
    """
    Applies K-Means clustering to the given feature matrix.

    Parameters:
    - features (np.ndarray): Feature matrix (e.g., TF-IDF, Word2Vec).
    - n_clusters (int): Number of clusters to form (default: 3).
    - random_state (int): Random seed for reproducibility (default: 42).
    - save_path (str, optional): Path to save the trained KMeans model.

    Returns:
    - labels (np.ndarray): Cluster labels for each data point.
    - kmeans_model (KMeans): Trained KMeans model.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features)


    return labels, kmeans







def apply_agglomerative_clustering(features: np.ndarray, n_clusters: int = 3, linkage: str = "ward"):
    """
    Applies Agglomerative Hierarchical Clustering.

    Parameters:
    - features (np.ndarray): Feature matrix.
    - n_clusters (int): Number of clusters to form (default: 5).
    - linkage (str): Linkage criterion ('ward', 'complete', 'average', 'single').

    Returns:
    - labels (np.ndarray): Cluster labels.
    - model (AgglomerativeClustering): Trained clustering model.
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(features)
    return labels, model




def apply_lda(text_data, n_topics=3, max_features=5000):
    """
    Applies Latent Dirichlet Allocation (LDA) for topic modeling on text data.

    Parameters:
    - text_data (list or pd.Series): List of text documents.
    - n_topics (int): Number of topics to extract.
    - max_features (int): Maximum number of features for the CountVectorizer.

    Returns:
    - topic_distributions (np.ndarray): Document-topic matrix.
    - lda_model (LatentDirichletAllocation): Trained LDA model.
    - feature_names (list): Vocabulary terms.
    """
    
    # Convert text data into a bag-of-words representation
    vectorizer = CountVectorizer(stop_words='english', max_features=max_features)
    text_matrix = vectorizer.fit_transform(text_data)

    # Apply LDA
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    topic_distributions = lda_model.fit_transform(text_matrix)

    return topic_distributions, lda_model, vectorizer.get_feature_names_out()


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np



def plot_elbow_method(features, max_k=10):
    """
    Plots the Elbow Method to determine the optimal number of clusters (K).
    
    Parameters:
    - features (np.ndarray): The feature matrix (TF-IDF, Word2Vec, etc.).
    - max_k (int): The maximum number of clusters to test.
    
    Returns:
    - None (Displays the Elbow plot).
    """
    distortions = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        distortions.append(kmeans.inertia_)

    # Plot the Elbow Method
    plt.figure(figsize=(6, 4))
    plt.plot(K_range, distortions, marker='o', linestyle='-')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Distortion (Inertia)")
    plt.title("Elbow Method")
    plt.show()



def plot_silhouette_scores(features, max_k=10):
    """
    Plots the Silhouette Scores for different cluster values.
    
    Parameters:
    - features (np.ndarray): The feature matrix (TF-IDF, Word2Vec, etc.).
    - max_k (int): The maximum number of clusters to test.
    
    Returns:
    - None (Displays the Silhouette Score plot).
    """
    silhouette_scores = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        silhouette_scores.append(silhouette_score(features, kmeans.labels_))

    # Plot Silhouette Score
    plt.figure(figsize=(6, 4))
    plt.plot(K_range, silhouette_scores, marker='s', linestyle='-', color='red')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score Analysis")
    plt.show()




def apply_pca_and_visualize(features, labels=None, n_components=2):
    """
    Applies PCA to reduce dimensionality and visualize the dataset.

    Parameters:
    - features (np.ndarray): The high-dimensional feature matrix (TF-IDF, Word2Vec, etc.).
    - labels (Optional[np.ndarray]): Cluster labels for coloring (default: None).
    - n_components (int): Number of PCA components (default: 2 for 2D visualization).

    Returns:
    - pca_result (np.ndarray): Transformed 2D or 3D representation of the features.
    """
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features)

    # Create a DataFrame for visualization
    df_pca = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_components)])
    if labels is not None:
        df_pca["Cluster"] = labels

    # Plotting
    plt.figure(figsize=(10, 6))
    if n_components == 2:
        sns.scatterplot(x=df_pca["PC1"], y=df_pca["PC2"], hue=df_pca["Cluster"] if labels is not None else None, palette="tab10", alpha=0.7)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df_pca["PC1"], df_pca["PC2"], df_pca["PC3"], c=labels, cmap="tab10", alpha=0.7)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
    
    plt.title("PCA Visualization of Clusters")
    plt.show()

    return pca_result
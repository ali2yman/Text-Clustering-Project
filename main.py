import pandas as pd 
from preprocessing import clean_text,drop_columns,tokenize_clean_lemmatize  
from feature_extraction import compute_tfidf
from clustering import apply_kmeans, apply_lda , apply_pca, apply_agglomerative_clustering
from visualization import plot_elbow_method, plot_silhouette_scores, visualize_pca
from evaluation import get_silhouette_score



# Load dataset
data_path = "data/people_wiki.csv"  # Update with the actual dataset path
df = pd.read_csv(data_path)

# Step 1: Preprocessing
df = drop_columns(df,['URI','name'])  # Drop unnecessary columns
df = clean_text(df,'text')  # Clean text
df = tokenize_clean_lemmatize(df,'text')

# Step 2: Feature Extraction (TF-IDF or Word2Vec)
features , vectorizer = compute_tfidf(df)  # or method="word2vec"

# Step 3: Dimensionality Reduction (PCA)
pca_results = apply_pca(features, n_components=50)

# Step 4: Clustering
# K-Means
labels_kmeans ,k_means_model = apply_kmeans(pca_results, n_clusters=3)
# Agglomerative Clustering
# labels_hierarchical = apply_hierarchical(features, n_clusters=5)
# # LDA (Topic Modeling)
# topic_distributions, lda_model, feature_names = apply_lda(df["cleaned_text"], n_topics=5)

# Step 5: Evaluation
silhouette_kmeans = get_silhouette_score(pca_results, labels_kmeans)
# db_kmeans = davies_bouldin_score_evaluation(features, labels_kmeans)

# Step 6: Visualization
# visualize_pca(pca_result, labels_kmeans)

# Print Evaluation Results
print(f"Silhouette Score (K-Means): {silhouette_kmeans}")
# print(f"Davies-Bouldin Index (K-Means): {db_kmeans}")

print("Clustering Completed Successfully! 🚀")

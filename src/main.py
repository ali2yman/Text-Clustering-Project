# 🟢 Set the CPU core count first!
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"  
import warnings
warnings.filterwarnings("ignore")
import pandas as pd 
from preprocessing import clean_text,drop_columns,tokenize_clean_lemmatize  
from feature_extraction import  doc2vec_vectorization , compute_tfidf_from_df
from clustering import apply_kmeans, apply_lda , apply_pca, apply_agglomerative_clustering
from visualization import plot_elbow_method, plot_silhouette_scores, visualize_pca
from evaluation import get_silhouette_score , purity_score
from Download_data import collect_data




print(" Starting Clustering Pipeline...")
print("Which Data U want To Use ? \n 1 - people_wiki \n 2 - 20 newsgroups ")

choice = int(input("Enter your choice: "))

if choice == 1:
    # pipeline on people_wiki
    data_path = 'data/people_wiki.csv'  
    df = pd.read_csv(data_path)

    # Pre-processing step
    print("Pre-processing step...Please wait...")
    # df = clean_text(df,'text')
    # df = drop_columns(df,['URI','name'])
    # df = tokenize_clean_lemmatize(df,'text')
    df = pd.read_csv('data/preprocessed_people_wiki.csv')    # After applying pre-processing

    # Feature extraction step
    features , vectorizer = doc2vec_vectorization(df , 'text')

    # Clustering step
    print("Clustering step...Please wait...")
    pca_results = apply_pca(features.toarray(), n_components=.9)
    labels_kmeans ,k_means_model = apply_kmeans(pca_results, n_clusters=3)

    # visualization step
    print("Visualization step...Please wait...")
    visualize_pca(pca_results, labels_kmeans)

    # Evaluation step
    print("Evaluation step...Please wait...")
    silhouette_kmeans = get_silhouette_score(pca_results, labels_kmeans)
    print(f"Silhouette Score (K-Means): {silhouette_kmeans}")


else:
    # pipeline on 20 newsgroups
    df = collect_data()

    # Pre-processing step
    print("Pre-processing step...Please wait...")
    df = clean_text(df,'data')
    df = tokenize_clean_lemmatize(df,'data')
    df = df[df['data'].str.strip().astype(bool)]
    df.reset_index(drop=True, inplace=True)

    # Feature extraction step
    print("Feature extraction step...Please wait...")
    features , vectorizer  = compute_tfidf_from_df(df,'data')
    # features , vectorizer  = doc2vec_vectorization(df,'data',model_path='models/doc2vec_news.model')

    # Clustering step
    print("Clustering step...Please wait...")
    pca_results = apply_pca(features.toarray(), n_components=.9)
    labels_kmeans ,k_means_model = apply_kmeans(pca_results, n_clusters=3)
    # labels_kmeans ,k_means_model = apply_agglomerative_clustering(pca_results, n_clusters=3)

    # visualization step
    print("Visualization step...Please wait...")
    visualize_pca(pca_results, labels_kmeans)

    # Evaluation step
    print("Evaluation step...Please wait...")
    silhouette_kmeans = get_silhouette_score(pca_results, labels_kmeans)
    print(f"Silhouette Score (K-Means): {silhouette_kmeans}")

    purity_score = purity_score(df['target'],labels_kmeans)
    print(f"Purity Score (K-Means): {purity_score}")

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.preprocessing import clean_text, tokenize_clean_lemmatize
from src.feature_extraction import doc2vec_vectorization, compute_tfidf_from_df
from src.clustering import apply_kmeans, apply_pca
from src.Download_data import collect_data




st.set_page_config(page_title="Text Cluster Classifier", layout="centered")
st.title("🧠 Text Cluster Classifier")

# Step 1: Choose Dataset
dataset_choice = st.selectbox("Select a dataset to train the clustering model:", ["People Wiki", "20 Newsgroups"])

# Step 2: Enter Text
user_text = st.text_area("Enter your text below for classification:", height=200)

# Step 3: Submit
if st.button("Classify Text") and user_text.strip():
    st.info("⏳ Processing...")

    if dataset_choice == "People Wiki":
        # Load preprocessed dataset
        df = pd.read_csv('data/preprocessed_people_wiki.csv')
        text_column = 'text'
        features, _ = doc2vec_vectorization(df, text_column)

    else:
        # Load and preprocess 20 newsgroups
        df = collect_data()
        df = clean_text(df, 'data')
        df = tokenize_clean_lemmatize(df, 'data')
        df = df[df['data'].str.strip().astype(bool)]
        df.reset_index(drop=True, inplace=True)
        text_column = 'data'
        features, _ = compute_tfidf_from_df(df, text_column)

    # Apply PCA and clustering
    pca_results = apply_pca(features.toarray(), n_components=0.9)
    labels, kmeans_model = apply_kmeans(pca_results, n_clusters=3)

    # Preprocess the user input
    new_df = pd.DataFrame({text_column: [user_text]})
    new_df = clean_text(new_df, text_column)
    new_df = tokenize_clean_lemmatize(new_df, text_column)

    # Vectorize input
    if dataset_choice == "People Wiki":
        new_vector, _ = doc2vec_vectorization(new_df, text_column)
    else:
        new_vector, _ = compute_tfidf_from_df(new_df, text_column)

    # Apply same PCA transformation
    new_vector_pca = apply_pca(new_vector.toarray(), n_components=0.9)

    # Predict cluster
    predicted_cluster = kmeans_model.predict(new_vector_pca)[0]

    st.success(f"✅ The input text belongs to **Cluster {predicted_cluster}**")

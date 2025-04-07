from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp



def doc2vec_vectorization(df: pd.DataFrame, text_column: str, model_path="models/doc2vec.model",
                          vector_size=100, window=5, min_count=2, epochs=20, return_sparse=True):
    """
    Applies Doc2Vec vectorization to text data.

    Parameters:
    
    - df (pd.DataFrame): Input DataFrame.
    - text_column (str): Column name containing text data.
    - model_path (str): Path to save/load the trained model.
    - vector_size (int): Dimensionality of the document vectors.
    - window (int): Maximum distance between current and predicted word.
    - min_count (int): Ignores words with total frequency lower than this.
    - epochs (int): Number of training iterations.
    - return_sparse (bool): If True, returns a sparse matrix.

    Returns:
    - doc_vectors (np.ndarray or scipy.sparse.csr_matrix): Document vectors.
    - model (Doc2Vec): Trained Doc2Vec model.
    """
    texts = df[text_column].astype(str).tolist()
    tagged_data = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(texts)]

    if os.path.exists(model_path):
        print(f"🔹 Loading existing Doc2Vec model from {model_path}")
        model = Doc2Vec.load(model_path)
    else:
        print("🚀 Training new Doc2Vec model...")
        model = Doc2Vec(tagged_data, vector_size=vector_size, window=window, min_count=min_count, epochs=epochs)
        model.save(model_path)

    doc_vectors = np.array([model.dv[i] for i in range(len(texts))])

    if return_sparse:
        doc_vectors = sp.csr_matrix(doc_vectors)  # Convert to sparse format

    return doc_vectors, model


def compute_tfidf_from_df(df, text_column="text", max_features=500, min_df=4, max_df=0.85, ngram_range=(1,3)):
    """
    Applies optimized TF-IDF vectorization to text in a Pandas DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        text_column (str): Column name containing text.
        max_features (int): Max number of words to keep.
        min_df (int): Minimum document frequency.
        max_df (float): Maximum document frequency.
        ngram_range (tuple): (min_n, max_n) range for n-grams.

    Returns:
        tfidf_matrix (sparse matrix): TF-IDF transformed feature matrix.
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features, 
        min_df=min_df, 
        max_df=max_df, 
        ngram_range=ngram_range, 
        stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(df[text_column])  
    return tfidf_matrix, vectorizer

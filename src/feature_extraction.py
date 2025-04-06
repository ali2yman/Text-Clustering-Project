from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

#  1 - Appling TF-IDF   
def compute_tfidf(df, text_column="text", max_features=5000, min_df=3, max_df=0.85, ngram_range=(2,2)):
    """
    Converts the text data into TF-IDF vectors with optimized hyperparameters.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the text data.
    - text_column (str): The column name that contains text data.
    - max_features (int): The maximum number of features for vectorization.
    - min_df (int): The minimum document frequency for a word to be included.
    - max_df (float): The maximum document frequency threshold.
    - ngram_range (tuple): The range of n-grams to include.

    Returns:
    - tfidf_matrix (sparse matrix): The TF-IDF transformed data.
    - vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words='english',
        ngram_range=ngram_range
    )
    
    tfidf_matrix = vectorizer.fit_transform(df[text_column])
    
    return tfidf_matrix, vectorizer





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

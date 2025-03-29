from sklearn.feature_extraction.text import TfidfVectorizer


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




import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re

# 1 - make a function to drop columns from the dataframe

def drop_columns(df,columns):
    """
    Drops specified columns from the given DataFrame.
    
    :param df: Input DataFrame.
    :param columns_to_drop: List of column names to drop.
    :return: DataFrame after dropping columns.
    """
    df = df.drop(columns=columns, axis=1)
    return df

# 2 - make a function to clean the text
def clean_text(df, text_column):
    """
    Cleans the text in a specified column of a DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame.
        text_column (str): The name of the column containing text data.

    Returns:
        pd.DataFrame: The DataFrame with cleaned text.
    """
    df[text_column] = df[text_column].astype(str).str.lower()  
    df[text_column] = df[text_column].apply(lambda text: re.sub(r'[^a-z\s]', '', text))  # Keep only letters and spaces
    df[text_column] = df[text_column].apply(lambda text: re.sub(r'\s+', ' ', text).strip())  # Remove extra spaces
    return df


# 3 - make a function to tokenize, remove stopwords, and apply lemmatization
def tokenize_clean_lemmatize(df, text_column):
    """
    Tokenizes text, removes stopwords, and applies lemmatization.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        text_column (str): The column containing text data.

    Returns:
        pd.DataFrame: A DataFrame with processed text.
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def process_sentence(text):
        # Tokenization
        tokens = word_tokenize(text)
        # Removing stopwords & applying lemmatization
        cleaned_tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens if word.lower() not in stop_words]
        return " ".join(cleaned_tokens)

    df[text_column] = df[text_column].astype(str).apply(process_sentence)
    return df

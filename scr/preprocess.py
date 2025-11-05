import nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize
import string
import pandas as pd

nltk.download('movie_reviews', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

def load_movie_reviews():
    """Load the IMDB dataset from NLTK and return a DataFrame."""
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            words = movie_reviews.raw(fileid)
            documents.append((words, category))
    df = pd.DataFrame(documents, columns=["review", "sentiment"])
    return df

def clean_text(text):
    """Clean text: lowercase, remove punctuation and stopwords."""
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

def preprocess_data(df):
    """Apply text cleaning to the entire DataFrame."""
    df["clean_review"] = df["review"].apply(clean_text)
    return df

if __name__ == "__main__":
    df = load_movie_reviews()
    df = preprocess_data(df)
    df.to_csv("data/clean_reviews.csv", index=False)
    print(df.head())
    print("\nDataset size:", df.shape)

import re
import os
import pandas as pd
import nltk
from typing import Optional

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('universal_tagset', quiet=True)

from nltk.corpus import stopwords, movie_reviews
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

STOPWORDS = set(stopwords.words('english'))
LEM = WordNetLemmatizer()


def _clean_html(text: str) -> str:
    return re.sub(r'<[^>]+>', ' ', text)


def _clean_urls_emails(text: str) -> str:
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    return text


def _remove_non_printable(text: str) -> str:
    return ''.join(ch for ch in text if ch.isprintable())


def _clean_punct_digits(text: str) -> str:
    text = re.sub(r'[_–—\-]', ' ', text)
    text = re.sub(r"[^a-zA-Z\s']", ' ', text)
    return text


def _normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def _pos_tag_to_wordnet(pos_tag: str):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize_tokens(tokens):
    pos_tags = nltk.pos_tag(tokens)
    lemmas = []
    for token, pos in pos_tags:
        wn_pos = _pos_tag_to_wordnet(pos)
        lemmas.append(LEM.lemmatize(token, wn_pos))
    return lemmas


def clean_text_advanced(text: str,
                        remove_stopwords: bool = True,
                        lemmatize: bool = True) -> str:
    """
    Advanced text cleaning: HTML, URLs, nonprintable, punct/digits removal,
    tokenize, optional stopwords removal and POS-aware lemmatization.
    Returns cleaned string (space-separated tokens).
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _clean_html(text)
    text = _clean_urls_emails(text)
    text = _remove_non_printable(text)
    text = _clean_punct_digits(text)
    text = _normalize_whitespace(text)

    tokens = word_tokenize(text)

    tokens = [t for t in tokens if t.isalpha() and len(t) > 1]

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]

    if lemmatize and tokens:
        tokens = lemmatize_tokens(tokens)

    return " ".join(tokens)


def load_movie_reviews_nltk() -> pd.DataFrame:
    """
    Load NLTK movie_reviews and return DataFrame with columns 'review' and 'label' (0/1)
    """
    docs = []
    for cat in movie_reviews.categories():
        for fid in movie_reviews.fileids(cat):
            raw = movie_reviews.raw(fid)
            label = 1 if cat == 'pos' else 0
            docs.append({"review": raw, "label": label})
    df = pd.DataFrame(docs)
    return df


def preprocess_dataframe(df: pd.DataFrame,
                         text_col: str = "review",
                         label_col: str = "label",
                         remove_stopwords: bool = True,
                         lemmatize: bool = True,
                         save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Apply cleaning to df[text_col], add column 'clean_text' and return new df.
    Optionally save to CSV (save_path).
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame")

    out = df.copy()
    out['clean_text'] = out[text_col].apply(lambda t: clean_text_advanced(t, remove_stopwords, lemmatize))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        out.to_csv(save_path, index=False)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess dataset (NLTK or CSV)")
    parser.add_argument("--source", choices=["nltk", "csv"], default="nltk",
                        help="Load data from 'nltk' movie_reviews or from a local csv")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to CSV if source=csv")
    parser.add_argument("--out_csv", type=str, default="data/clean_reviews.csv",
                        help="Where to save cleaned CSV")
    parser.add_argument("--no_lemmatize", action="store_true", help="Disable lemmatization")
    parser.add_argument("--no_stopwords", action="store_true", help="Disable stopword removal")
    args = parser.parse_args()

    if args.source == "csv" and not args.csv_path:
        raise SystemExit("csv_path required when source=csv")

    print("Loading dataset...")
    if args.source == "nltk":
        df0 = load_movie_reviews_nltk()
    else:
        df0 = pd.read_csv(args.csv_path)

    print(f"Loaded {len(df0)} rows")
    print("Preprocessing (this might take a while)...")
    df_clean = preprocess_dataframe(df0,
                                    remove_stopwords=(not args.no_stopwords),
                                    lemmatize=(not args.no_lemmatize),
                                    save_path=args.out_csv)
    print("Saved cleaned CSV to", args.out_csv)
    print(df_clean.head(3))

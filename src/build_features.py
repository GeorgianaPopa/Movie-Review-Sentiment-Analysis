import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

DEFAULT_VECT_PATH = os.path.join("models", "tfidf_vectorizer.joblib")
DEFAULT_DATA_PATH = os.path.join("data", "clean_reviews.csv")


def build_tfidf(data_csv: str = DEFAULT_DATA_PATH,
                output_vectorizer: str = DEFAULT_VECT_PATH,
                max_features: int = 20000,
                ngram_range=(1, 2),
                test_size: float = 0.2,
                random_state: int = 42,
                save_split: bool = False):
    df = pd.read_csv(data_csv)
    if 'clean_text' not in df.columns:
        raise ValueError("CSV must contain 'clean_text' column. Run preprocessing first.")

    X_text = df['clean_text'].fillna("")
    y = df['label'].astype(int)

    vect = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_all = vect.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)

    os.makedirs(os.path.dirname(output_vectorizer), exist_ok=True)
    joblib.dump(vect, output_vectorizer)
    print(f"Saved TF-IDF vectorizer to {output_vectorizer}")

    if save_split:
        from scipy import sparse
        os.makedirs("data/splits", exist_ok=True)
        sparse.save_npz("data/splits/X_train.npz", X_train)
        sparse.save_npz("data/splits/X_test.npz", X_test)
        y_train.to_csv("data/splits/y_train.csv", index=False)
        y_test.to_csv("data/splits/y_test.csv", index=False)
        print("Saved train/test splits under data/splits/")

    return X_train, X_test, y_train, y_test, vect


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out_vect", type=str, default=DEFAULT_VECT_PATH)
    parser.add_argument("--max_features", type=int, default=20000)
    parser.add_argument("--save_split", action='store_true')
    args = parser.parse_args()

    build_tfidf(data_csv=args.data_csv, output_vectorizer=args.out_vect,
                max_features=args.max_features, save_split=args.save_split)

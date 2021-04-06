import configparser
import argparse
import pickle
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

def load_data(data_path):
    texts = []
    tags = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
            tags.append(item["tags"])
    return texts, tags

def train(data_dir, model_dir, min_df=5, max_features=None, lowercase=True,
        stopwords="english", ngram_range=(1,1), kernel="linear", class_weight=None):
    print("Loading data")
    data_path = os.path.join(data_dir, "data.jsonl")
    X, Y = load_data(data_path)

    print("Fitting label binarizer")
    label_binarizer = MultiLabelBinarizer()
    label_binarizer.fit(Y)

    Y_vec = label_binarizer.transform(Y)
    
    print("Splitting data")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_vec, random_state=42)
   
    print("Fitting model")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            min_df=min_df,
            max_features=max_features,
            lowercase=lowercase,
            stop_words=stopwords,
            ngram_range=ngram_range
        )),
        ("svm", OneVsRestClassifier(SVC(
            kernel=kernel,
            class_weight=class_weight
        )))
    ])

    model.fit(X_train, Y_train)

    print("Evaluating model")
    Y_pred = model.predict(X_test)
    f1 = f1_score(Y_test, Y_pred, average="micro")
    print(f"Score f1 {f1:.4f}")

    model_path = os.path.join(model_dir, "firebreak-model.pkl")
    label_binarizer_path = os.path.join(model_dir, "firebreak-label_binarizer.pkl")
    with open(model_path, "wb") as f:
        f.write(pickle.dumps(model))
    with open(label_binarizer_path, "wb") as f:
        f.write(pickle.dumps(label_binarizer))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--min_df", type=int, default=5)
    parser.add_argument("--max_features", type=int, default=None)
    parser.add_argument("--lowercase", type=bool, default=True)
    parser.add_argument("--stopwords", type=str, default="english")
    parser.add_argument("--ngram_range", type=tuple, default=(1,1))
    parser.add_argument("--kernel", type=str, default="linear")
    parser.add_argument("--class_weight", type=str, default=None)
    parser.add_argument("--config", type=str)
    parser.add_argument("--cloud", action="store_true")
    args = parser.parse_args()

    if args.config:
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.read(args.config)
        
        data_path = cfg["input"]["path"]
        model_path = cfg["output"]["path"]
        min_df = cfg["tfidf"].getint("min_df", 5)
        max_features = cfg["tfidf"].get("max_features")
        lowercase = cfg["tfidf"].getboolean("lowercase", True)
        stopwords = cfg["tfidf"].get("stopwords", "english")
        ngram_range = cfg["tfidf"].get("ngram_range", [1,1])
        kernel = cfg["svm"].get("kernel", "linear")
        class_weight = cfg["svm"].get("class_weight")
    else:
        data_path = args.data_path
        model_path = args.model_path
        min_df = args.min_df
        max_features = args.max_features
        lowercase = args.lowercase
        stopwords = args.stopwords
        ngram_range = args.ngram_range
        kernel = args.kernel
        class_weight = args.class_weight
    
    if args.cloud:
        data_path = "/opt/ml/input/data/training"
        model_path = "/opt/ml/models"

    train(data_path, model_path, min_df, max_features,
        lowercase, stopwords, ngram_range, kernel, class_weight)


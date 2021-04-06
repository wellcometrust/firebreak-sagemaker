from sagemaker.estimator import Estimator
import sagemaker

import argparse
import configparser
import tarfile
import json
import os


# Read from ENV vars
BUCKET = "sagemaker-eu-west-1-" + os.environ["AWS_ACCOUNT_ID"] 
PREFIX = "grants_tagger"

def create_tar_file(source_files, target=None):
    if target:
        filename = target
    else:
        _, filename = tempfile.mkstemp()

    with tarfile.open(filename, mode="w:gz") as t:
        for sf in source_files:
            t.add(sf)


def json_encode_hyperparameters(hyperparameters):
    return {str(k): json.dumps(v) for (k, v) in hyperparameters.items()}


def train_with_sagemaker(data_path, model_path, config_path, instance_type, role, image_uri):
    session = sagemaker.Session()
    create_tar_file([
        "grants_tagger/__init__.py",
        "grants_tagger/train.py",
    ], "sourcedir.tar.gz")
    sources = session.upload_data('sourcedir.tar.gz', BUCKET, PREFIX + '/code')
    
    cfg = configparser.ConfigParser(allow_no_value=True)
    cfg.read(config_path)
        
#    data_path = cfg["input"]["path"]
#    model_path = cfg["output"]["path"]
    min_df = cfg["tfidf"].getint("min_df", 5)
    max_features = cfg["tfidf"].get("max_features")
    lowercase = cfg["tfidf"].getboolean("lowercase", True)
    stopwords = cfg["tfidf"].get("stopwords", "english")
    ngram_range = cfg["tfidf"].get("ngram_range", [1,1])
    kernel = cfg["svm"].get("kernel", "linear")
    class_weight = cfg["svm"].get("class_weight")


    if instance_type == "local":
        assert "file://" in data_path
        assert "file://" in model_path
    else:
        assert "s3://" in data_path
        assert "s3://" in model_path
        
    container_data_path = "/opt/ml/input/data/training/"
    container_model_path = "/opt/ml/model"

    hyperparameters = {
        "sagemaker_program": "grants_tagger/train.py",
        "sagemaker_submit_directory": sources,
        "data_path": container_data_path,
        "model_path": container_model_path,
        "min_df": min_df,
        "max_features": max_features,
        "lowercase": lowercase,
        #"ngram_range": ngram_range,
        "kernel": kernel,
        "class_weight": class_weight
    }
    hyperparameters = {k:v for k,v in hyperparameters.items() if v is not None}
    hyperparameters = json_encode_hyperparameters(hyperparameters)

    es = Estimator(
        image_uri=image_uri,
        role=role,
        framework_version='0.20.0',
        instance_count=1,
        instance_type=instance_type,
        output=model_path,
        hyperparameters=hyperparameters,
        base_job_name="firebreak-grants-tagger" # is it needed?
    )
    es.fit({"training": data_path})

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str, default="file://data")
    argparser.add_argument("--model_path", type=str, default="file://models")
    argparser.add_argument("--config", type=str, default="configs/config.ini")
    argparser.add_argument("--instance_type", type=str, default="local", help="AWS instance type e.g. ml.m5.large, if local it runs locally")
    argparser.add_argument("--role", type=str, default=os.environ["SAGEMAKER_ROLE"])
    argparser.add_argument("--image_uri", type=str, default=os.environ["AWS_ACCOUNT_ID"] + ".dkr.ecr.eu-west-1.amazonaws.com/firebreak-grants-tagger")
    args = argparser.parse_args()

    train_with_sagemaker(args.data_path, args.model_path, args.config, args.instance_type, args.role, args.image_uri)

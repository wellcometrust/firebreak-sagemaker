import argparse
import os

from sagemaker.sklearn import SKLearn
import sagemaker


def train_with_sagemaker(model_path, data_path, min_df, max_features, lowercase,
        stopwords, ngram_range, kernel, class_weight, instance_type, role):
    # Is that needed?
    #session = sagemaker.Session()
   
    if instance_type == "local":
        assert "file://" in data_path
        assert "file://" in model_path
    else:
        assert "s3://" in data_path
        assert "s3://" in model_path
        
    container_data_path = "/opt/ml/input/data/training/"
    container_model_path = "/opt/ml/model"

    hyperparameters = {
        "data_path": container_data_path,
        "model_path": container_model_path,
        "min_df": min_df,
        "max_features": max_features,
        "lowercase": lowercase,
        "stopwords": stopwords,
        # "ngram_range": ngram_range,
        "class_weight": class_weight
    }
    hyperparameters = {k: v for k, v in hyperparameters.items() if v}
    sk = SKLearn(
        entry_point="grants_tagger/train.py",
        role=role,
        framework_version='0.20.0',
        instance_count=1,
        instance_type=instance_type,
        output=model_path,
        hyperparameters=hyperparameters
    )

    sk.fit({"training": data_path})


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_path", type=str, default="file://models/")
    argparser.add_argument("--data_path", type=str, default="file://data/")
    argparser.add_argument("--min_df", type=int, default=5)
    argparser.add_argument("--max_features", type=int, default=None)
    argparser.add_argument("--lowercase", type=bool, default=True)
    argparser.add_argument("--stopwords", type=str, default="english")
    argparser.add_argument("--ngram_range", type=tuple, default=(1,1))
    argparser.add_argument("--kernel", type=str, default="linear")
    argparser.add_argument("--class_weight", type=str, default=None)
    argparser.add_argument("--instance_type", type=str, default="local", help="AWS instance type e.g. ml.m5.large, if local it runs locally")
    argparser.add_argument("--role", type=str, default=os.environ["SAGEMAKER_ROLE"])
    args = argparser.parse_args()

    train_with_sagemaker(args.model_path, args.data_path, args.min_df, args.max_features,
            args.lowercase, args.stopwords, args.ngram_range, args.kernel, args.class_weight,
            args.instance_type, args.role)

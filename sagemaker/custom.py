import argparse
import os

from sagemaker.estimator import Estimator


def train_with_sagemaker(model_path, data_path, instance_type, role, image_uri):
    if instance_type == "local":
        assert "file://" in data_path
        assert "file://" in model_path
    else:
        assert "s3://" in data_path
        assert "s3://" in model_path
    
    es = Estimator(
        role=role,
        framework_version='0.20.0',
        image_uri=image_uri,
        instance_count=1,
        instance_type="ml.m5.large",
        output=model_path,
        input_mode="File",
        hyperparameters={}
    )
    print(data_path)
    es.fit({"training": data_path})

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_path", type=str, default="file://models/")
    argparser.add_argument("--data_path", type=str, default="file://data/")
    argparser.add_argument("--instance_type", type=str, default="local", help="AWS instance type e.g. ml.m5.large, if local it runs locally")
    argparser.add_argument("--role", type=str, default=os.environ["SAGEMAKER_ROLE"])
    argparser.add_argument("--image_uri", type=str, default=os.environ["AWS_ACCOUNT_ID"]+".dkr.ecr.eu-west-1.amazonaws.com/firebreak-grants-tagger-custom")
    args = argparser.parse_args()

    train_with_sagemaker(args.model_path, args.data_path, args.instance_type,
            args.role, args.image_uri)

# Firebreak AWS SageMaker

AWS SageMaker is a ML platform from AWS where you can train and deploy
machine learning models using AWS instances in an "easier" way.

The aim of the project was to provide some ways that we can leverage 
SageMaker to train our models using AWS that is not very disruptive
to our current flow.

The project currently highlights 3 ways of using SageMaker:

1. Using SageMaker containers pre built with known frameworks like sklearn
2. Using a custom container of our own containing code and requirements
3. Using a custom container of our own that only contain requirements

# Setup

Create a virtualenv `make virtualenv` and download the data `make download-data`

Ideally an environemtn variable needs to be set with the sagemaker role
`export SAGEMAKER_ROLE=xxx`. You can get the role from AWS console or Jeff/Nick.

You also need (not optional) an environment variable to bet with the AWS_ACCOUNT_ID
of datalabs. `export AWS_ACCOUNT_ID=xx`. You can get the id from console or Jeff/Nick
# Project

This project emulates the structure of a common datalabs project. There is a
`data` folder and a `models` folder. There is a `config` file but also the ability
to pass training arguments as arguments. There is a `train.py` inside grants_tagger.

There is an unpinned_requirements.txt which creates the frozen requirements.txt
by running `make update-requirements-txt`.

# SageMaker Containers

SageMaker provides built in containers that contain the frameworks we use.
Frameworks supported that are of interest to us are:

* sklearn https://docs.aws.amazon.com/sagemaker/latest/dg/sklearn.html
* Tensorflow https://docs.aws.amazon.com/sagemaker/latest/dg/tf.html
* huggingface https://docs.aws.amazon.com/sagemaker/latest/dg/hugging-face.html

This mode is useful for rapid experimentations and in case a project only uses
one of those libraries. The benefit is that you do not need to build a container,
just write the code which will be copied and executed inside the pre built container.

You can see an example in `sagemaker/sklearn.py`. You can test it works locally by running

```
python sagemaker/sklearn.py \
	--data_path file://data/ \
	--model_path file://models/ \
	--instance_type local \
	--class_weight balanced
```

Note that data_path and model_path are directory paths. This is mainly because all SageMaker
examples work with directory paths, it can be adjusted to work with exact paths. Also note
that we prepend local paths with `file://` so that sagemaker knows to look locally.

To train the algorithm using an AWS instance you need to change the instance type to one that
sagemaker accepts e.g. `ml.m5.xlarge`, complete list here https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-instance-types.html
and pass data_path and model_path as s3_paths.

As AWS SageMaker currently does not have access to all s3 locations, we cannot use our normal
buckets so to ensure the data is available run `make upload-data`. Note that this can be resolved
by giving appropriate permissions to SageMaker which means we can use the data from the location
we already store them in s3.

```
python sagemaker/sklearn.py \
	--data_path s3://sagemaker-eu-west-1-$AWS_ACCOUNT_ID/grants_tagger/data/ \
	--model_path s3://sagemaker-eu-west-1-$AWS_ACCOUNT_ID/grants_tagger/models/ \
	--instance_type ml.m5.xlarge \
	--class_weight balanced
```

# Custom containers (with dependencies and code)

SageMaker supports bringing your own containers. The standard way to do that
is to create a container that contains the dependencies and the code that needs
to run, push it to ECR and point SageMaker to the container to run it.

You can see an example of that in `sagemaker/custom.py`

To build the container run 
```
docker build -t $AWS_ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com/firebreak-grants-tagger-custom -f Dockerfile.custom .
```

Note that the Dockefile.custom defines an entrypoint that instructs SageMaker what to run, in particular `python train.py --config config.ini --cloud`
the flag cloud is being used to convert the parts to paths that the container uses. Also, we introduce a config file here to showcase how sagemaker can
interact with a config file.

To train locally using sagemaker run
```
python sagemaker/custom.py \
	--data_path file://data/ \
	--model_path file://models/ \
	--instance_type local \
	--image_uri $AWS_ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com/firebreak-grants-tagger-custom
```

To train in the cloud you need to push the image to ECR `docker push $AWS_ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com/firebreak-grants-tagger-custom`
Pushing the container to ECR will take some time as it can be several GBs. You can speed this up if you do this from an AWS instance but this
kind of defies the point as if you are in an AWS instance you may as well run train in the instance you are.

Finally you need to replace paths with s3 equivalents and choose the instance type you want.
```
python sagemaker/custom.py \
	--data_path s3://sagemaker-eu-west-1-$AWS_ACCOUNT_ID/grants_tagger/data/ \
	--model_path s3://sagemaker-eu-west-1-$AWS_ACCOUNT_ID/grants_tagger/models \
	--instance_type ml.m5.xlarge \
	--image_uri $AWS_ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com/firebreak-grants-tagger-custom
```

# Custom containers (with dependencies but no code)

Briginging your own container has the main drawback that a new container needs to be pushed
in every code change and those containers tend to be quite heavy especially if you use
a lot of the commonly used data science libraries. Ideally we want to build our containers once
and to include all the dependencies to run the code but push the code into the container instead
of rebuilding each time we update it. This last example aims to achieve exactly this flow.

Since that is the recommended way in my opinion, I have partly automated that flow with some make commands.

You can build the container with `make build-docker`

The example code is in `sagemaker/train.py`

To train locally run
```
python sagemaker/train.py \
	--data_path file://data/ \
	--model_path file://models/ \
	--instance_type local \
	--image_uri $AWS_ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com/firebreak-grants-tagger
```

To be able to train in the cloud you need to ensure you push the container once. No need
to push every time you change the code, only whenever a new dependency is needed. You 
can push the container by running `make push-docker`

To train in the cloud, as always you need to pass the s3 paths to models and data and 
choose instance type.

```
python sagemaker/train.py \
	--data_path s3://sagemaker-eu-west-1-$AWS_ACCOUNT_ID/grants_tagger/data/ \
	--model_path s3://sagemaker-eu-west-1-$AWS_ACCOUNT_ID/grants_tagger/models \
	--instance_type ml.m5.xlarge \
	--image_uri $AWS_ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com/firebreak-grants-tagger
```

# Future directions

## Integrating with our data science structure more

I also tried but not shared to use the tool sagify https://github.com/Kenza-AI/sagify.
The problem I am seeing with sagify is that you need to copy and paste your code inside
sagify structure to use sagemaker which is disruptive to our current flow and the way
we track code changes and ensure they are reproducible.

I think SageMaker can integrate even more with our current flow if we ensure that
it has access to our s3 buckets which means it can pick up the data from there and
save the locations to environment variables so both sagemaker and make pick them up.

We should also save the ECR location and project name in env variables to eliminate
writing them or hardcoding them. Assuming this is done, the only variable that the user
would need to provide is the instance type.

Taking this a step further it would be nice to be able to pass a flag `--cloud` in train
along with `--instance_type` and trigger training using sagemaker.

## Builtin algorithms 

SageMaker provides a couple of built in algorithms https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html
It would be nice to create WellcomeML wrappers that use those algorithms along with some 
helper functions to preprocess the data to the format they use. The potential
benefit of these algorithms is that they are optimised in terms of speed and
memory. It is not very different from how we have wrapper functions for Spacy that
has its own algorithms with the difference that spacy is open source while these
algorithms are not.


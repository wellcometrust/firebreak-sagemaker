PYTHON = python3.7
VIRTUALENV = venv

ECR_IMAGE = $(AWS_ACCOUNT_ID).dkr.ecr.eu-west-1.amazonaws.com/firebreak-grants-tagger

$(VIRTUALENV)/.installed: requirements.txt
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	virtualenv --python $(PYTHON) $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip3 install -r requirements.txt
	$(VIRTUALENV)/bin/pip3 install --no-dependencies -e .
	touch $@

.PHONY: update-requirements-txt
update-requirements-txt: VIRTUALENV := /tmp/update-requirements-virtualenv
update-requirements-txt:
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	virtualenv --python $(PYTHON) $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip3 install -r unpinned_requirements.txt
	echo "# Created by 'make update-requirements-txt'. DO NOT EDIT!" > requirements.txt
	$(VIRTUALENV)/bin/pip freeze | grep -v pkg-resources==0.0.0 >> requirements.txt

.PHONY: virtualenv
virtualenv: $(VIRTUALENV)/.installed

.PHONY: download-data
download-data:
	aws s3 cp s3://datalabs-data/grants_tagger/data/processed/science_grants_tagged_title_synopsis.jsonl data/data.jsonl

.PHONY: upload-data
upload-data:
	aws s3 cp data/data.jsonl s3://sagemaker-eu-west-1-$(AWS_ACCOUNT_ID)/grants_tagger/data/


.PHONY: aws-docker-login
aws-docker-login:
	aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.eu-west-1.amazonaws.com

.PHONY: build-docker
build-docker:
	docker build -t $(ECR_IMAGE):latest -f Dockerfile .

.PHONY: push-docker
push-docker: aws-docker-login
	docker push $(ECR_IMAGE):latest



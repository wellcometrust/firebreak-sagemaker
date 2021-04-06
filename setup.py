from setuptools import setup, find_packages


setup(
    name='grants_tagger',
    author='Nick Sorros',
    author_email='n.sorros@wellcome.ac.uk',
    description='A firebreak project on how to use AWS SageMaker',
    packages=find_packages(),
    version='0.1',
    install_requires=[
        'wellcomeml[deep-learning]==1.0.2',
    ]
)

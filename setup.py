from setuptools import find_packages, setup
from reddit_data.logging.logger import logging
from reddit_data.exception.exception import CustomException
import os,sys


def get_packages() -> list:
    try:
        packages = []
        with open('requirments.txt','r') as file:
            lines = file.readlines()
        for line in lines:
            line = line.strip()
            if '-e .' not in line:
                packages.append(line)
    except Exception as e:
        raise CustomException(e,sys)


setup(
    name =  'Reddit Classification',
    Author = 'Ravi Garlay',
    version = '0.0.1',
    email = 'ravigarlay@outlook.com',
    description='Reddit Rule violation classification Project using Pytorch/NLP',
    packages = find_packages(),
    install_packages = get_packages()
)


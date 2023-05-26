import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "fedlearning",
    version = "3.0.1",
    description = ("Federated Geometric Monte Carlo Clustering "),
    license = "BSD",
    keywords = "example documentation tutorial",
    packages=find_packages(include=['fedlearning']),
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Beta",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)

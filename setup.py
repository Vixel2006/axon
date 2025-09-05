from setuptools import setup, find_packages

setup(
    name='idrak',
    version='0.1.0',
    packages=find_packages(where='idrak'),
    package_dir={'': 'idrak'},
    install_requires=[
        'numpy',
    ],
)
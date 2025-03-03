from setuptools import setup, find_packages

setup(
    name='ml_models',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn'
    ],
    author='Augustin Cramer',
    description='A simple machine learning package',
    license='MIT',
    url='https://github.com/augustin-cramer/ml_models'
)
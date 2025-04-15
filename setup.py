from setuptools import find_packages, setup

setup(
    name='shapinteractions',
    version='0.1',
    author='Felix Furger',
    author_email='fefurger@hotmail.com',
    description='Build a comprehensive interaction graph visualization',
    url='',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'seaborn',
        'numpy',
        'pyvis',
        'scipy',
        'beautifulsoup4'
    ],
)
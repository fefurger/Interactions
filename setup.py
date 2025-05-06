from setuptools import find_packages, setup

setup(
    name='shapinteractions',
    version='0.1',
    author='Felix Furger',
    author_email='fefurger@hotmail.com',
    description='Build a comprehensive interaction graph visualization',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/fefurger/Interactions',
    packages=find_packages(),
    install_requires=[
        'matplotlib>=3.9.4',
        'seaborn>=0.13.2',
        'numpy>=2.0.2',
        'pyvis>=0.3.2',
        'scipy>=1.13.1',
        'beautifulsoup4>=4.13.4'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
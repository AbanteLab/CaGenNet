from setuptools import setup, find_packages

setup(
    name='cagennet',
    version='0.1.0',
    author='Jordi Abante, Berta Ros',
    author_email='jordi.abante@ub.edu, bertaros@ub.edu',
    description='Calcium Generative Networks (CaGenNet): a package with deep generative models for Ca imaging data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AbanteLab/CaGenNet',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'pyro',
        'scikit-learn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
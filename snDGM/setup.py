from setuptools import setup, find_packages

setup(
    name='snDGM',
    version='0.1.0',
    author='Jordi Abante',
    author_email='jordi.abante@ub.edu',
    description='A package for training a Variational Autoencoder (VAE) on calcium imaging data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/jordiabante/calcium_avae',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'scikit-learn',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
from setuptools import setup, find_packages

setup(
    name='ca_sn_gen_models',
    version='0.1.0',
    author='Jordi Abante',
    author_email='jordi.abante@ub.edu',
    description='A package with single-neuron generative models for calcium imaging data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/jordiabante/ca_sn_gen_models',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'scikit-learn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
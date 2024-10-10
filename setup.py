from setuptools import setup, find_packages

setup(
    name='UQ_airfoil',  
    version='0.1.0',    
    author='Enrico Foglia', 
    author_email='enrico.foglia@isae-supaero.fr', 
    description='Comparison of uncertainty quantification (UQ) techniques for simple graph neural networks (GNNs). All models are trained to predict the aerodynamic performances of the AirfRANS dataset.',
    packages=find_packages(where="src"),  
    package_dir={"": "src"}, 
    install_requires=[
        'numpy',
        'tqdm',
        'torch',
        'torch_geometric',
        'scipy',
        'airfrans',
        'matplotlib',
        'scikit-learn',
        'torchvision'
    ], 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',  
)

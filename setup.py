from setuptools import setup, find_packages

setup(
    name='mixencoder',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'tqdm'
    ],
    author='Max Svensson',
    author_email='maxiemum789@gmail.com',
    description='Package for pre-training embeddings using Mix Encoder.',
    url='https://github.com/msvenssons/mixencoder',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
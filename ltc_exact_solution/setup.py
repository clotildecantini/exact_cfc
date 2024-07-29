from setuptools import setup, find_packages

setup(
    name='ltc_exact_solution',
    version='0.1.0',
    author='Clotilde Cantini',
    author_email='clotildecantini1@outlook.fr',
    description='Exact solution for neuronal dynamic implementation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/ltc_exact_solution',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        # List your package dependencies here
    ],
)

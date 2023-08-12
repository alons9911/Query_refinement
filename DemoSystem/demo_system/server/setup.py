from setuptools import setup

setup(
    name='erica_backend',
    version='1.0',
    description='Query Refinement for Diversity Constraint Satisfaction',
    author='Alon Silberstein',
    author_email='alonzilb@post.bgu.ac.il',
    packages=['erica_backend'],
    install_requires=[
        'numpy',
        'pandas==1.5.2',
        'flask>=2.2.2',
        'flask_cors'
    ],
)
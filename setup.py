# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import pypcurve

setup(

    name='pypcurve',

    version=pypcurve.__version__,

    packages=find_packages(),

    author="Quentin André",

    author_email="quentin.andre@colorado.edu",

    description="A Python library for p-curve estimation",

    long_description=open('README.md', encoding="utf-8").read(),

    long_description_content_type='text/markdown',

    install_requires=["numpy", "matplotlib", "pandas", "scipy", "seaborn"],

    keywords=['mediation-analysis', 'statistics', 'process', 'plotting', 'data-science', 'data-analysis',
              'data-visualization', 'regression-models'],

    url='https://github.com/QuentinAndre/pypcurve/',

    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved",
        "Natural Language :: English",
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7"
    ],

    license="MIT",
    python_requires='>=3.7'
)

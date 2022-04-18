# EHAPI

# EHAPI: A standardized cohort definition for hospital-acquired pressure injury based on electronic health records.

We analyze the complexity of defining hospital acquired bedsores using diverse but inconsistent data sources, provide a definition that more closely resembles nursing guidelines, and showcase the higher accuracy of a hospital acquired bedsore prediction model based on our definition on a large dataset (MIMIC-III).
.


## Table of Contents

1. [Overview](https://github.com/manisci/EHAPI#overview)
2. [Infrastructure](https://github.com/manisci/EHAPI#infrastructure)
3. [Installation and Setup](https://github.com/manisci/EHAPI#installation-and-setup)
4. [Code Structure](https://github.com/manisci/EHAPI#code-structure)
5. [Contributors](https://github.com/manisci/EHAPI#contributors)
6. [References](https://github.com/manisci/EHAPI#references)

## Overview

This repository contains the code to extract and analyze the congruence among HAPI data in clinical notes, ICD9 diagnosis codes, and chart events from the Medical Information Mart for Intensive Care III (MIMIC-III) database. The script to define the four cohorts with different criteria for HAPI based on conflicts among data sources is included, in addition to the code to rest the performance of all cohorts for HAPI classification using tree-based and sequential neural network classifiers. For more technical information about each step, kindly look at the docstrings provided in the scripts.

## Infrastructure

This code has been running and tested with the following specification:

* OS: macOS Monterey	12.3
* Python version: Python 3.9.7
* Architecture: Apple M1 chip

Dependencies required are documented in the ''requirements.txt' file. 

## Installation and Setup

After cloning this repository onto your local machine, do the following:
1. Install Python 3.9.7 from the follwing link:
```
https://www.python.org/ftp/python/3.9.7/python-3.9.7-macos11.pkg
```
2. Install virtualenv, create a virtual environment and activate it:
```
python3.9 -m pip install virtualenv
python3.9 -m virtualenv ehapi
source ehapi/bin/activate
```
3. Navigate to the repository folder
```
cd EHAPI
```
4. Install the package requirements found in 'requirements.txt':
```
pip install -r requirements.txt
```
## Code Structure

### analyze_consistency.py 
Increasing the days-ahead parameter in the range [1,13], we look at how the model's performance on the test set in terms of the defined metric change.

### define_cohorts.py
Script for generating sample features and number of cases for 100 regions

### run_model_comparisons.py 
Runs an emergence detection model using feature time series
and case numbers for a set of locations. 

## Contributors
* Mani Sotoodeh 


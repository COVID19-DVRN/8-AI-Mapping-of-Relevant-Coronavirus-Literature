# Machine Learning Identifies Key Topics and Research Gaps for COVID-19

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Contributors](#contributors)
* [Contents](#contents)
* [Data](#data)
* [Folders](#folders)
* [Dependencies](#dependencies)

Contains work analyzing CORD-19 data.


## Contributors

<!-- Contents -->
## Contents
Given that thousands of publications on coronaviruses have been produced to date, necessitates the use of machine learning, such as principal components analysis (PCA) and topic modeling with latent dirichlet allocation (LDA) – is necessary to quickly identify key topics and knowledge gaps. Thus, we propose a generalizable machine learning framework that may be used to effectively automated identification of knowledge gaps for SARS-CoV-2 and other novel pathogens.<br />

<!-- Data -->
## Data
Publication texts were obtained from the COVID-19 Open Research Dataset CORD-19, 2020. We used this dataset for our analysis. <br />

## Folders
* analysis: jupyter notebooks with analytical workflows
* nlp: python package for streamlining nlp data processing
* big_data (gitignored): CORD-19 data and data outputs

The README.md file in this repository provides well-documented introduction to the directory structure and scripts. Within the 'analysis' folder, there are two jupyter notebooks for conducting PCA and LDA analysis, as well as creating key figures of the paper. The file in the NLP folder is what we used for processing the text we used.

<!-- Dependencies -->
## Dependencies
Packages necessary <br />

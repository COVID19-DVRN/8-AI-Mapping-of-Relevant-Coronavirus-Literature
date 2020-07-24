# Machine Learning Maps Research Needs in COVID-19 Literature

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Contributors](#contributors)
* [Contents](#contents)
* [Data](#data)
* [Folders](#folders)
* [Dependencies](#dependencies)


## Contributors
Anhvinh Doanvo, Xiaolu Qian, Divya Ramjee, Helen Piontkivska, Angel Desai, Maimuna Majumder

<!-- Contents -->
## Contents
Given that thousands of publications on coronaviruses have been produced to date, necessitates the use of machine learning, such as principal components analysis (PCA) and topic modeling with latent dirichlet allocation (LDA) – is necessary to quickly identify key topics and knowledge gaps. Thus, we propose a generalizable machine learning framework that may be used to effectively automated identification of knowledge gaps for SARS-CoV-2 and other novel pathogens.<br />

<!-- Data -->
## Data
Publication abstracts were obtained from the COVID-19 Open Research Dataset CORD-19, 2020. We used this dataset for our analysis. For the current version, we used the data that was produced on May 28, 2020. <br />

## Folders
* [analysis](analysis): jupyter notebooks with analytical workflows
* [nlp](nlp): python class for streamlining nlp data processing

The README.md file in this repository provides well-documented introduction to the directory structure and scripts. Within the 'analysis' folder, there are two jupyter notebooks for conducting PCA and LDA analysis, as well as creating key figures of the paper. The .py file in the NLP folder is what we used for processing the texts we used.

<!-- Dependencies -->
## Dependencies
Key packages used in the model:<br />
sklearn <br />
Gensim <br />
nltk <br />

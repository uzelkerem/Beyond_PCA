- [Getting Started](#getting-started)
  * [Installing Jupyter Notebook](#installing-jupyter-notebook)
      + [Using Conda](#using-conda)
- [Tutorial](#tutorial)
  * [Estimation of Genotype Likelihoods (GLs)](#estimation-of-genotype-likelihoods=gls)
    + [Important points to consider](#important-points-to-consider)
  * [PCA using GLs](#pca-using-gls)

    + [Loading the data](#loading-the-data)
      - [Loading from multiple CSV files](#loading-from-multiple-csv-files)
      - [Loading from TrackingData (DLC Analyzer)](#loading-from-trackingdata--dlc-analyzer-)
    + [Exploring the data structure of USData](#exploring-the-data-structure-of-usdata)
    + [Adding more data to an existing USData object](#adding-more-data-to-an-existing-usdata-object)
      - [Adding new labels to existing files](#adding-new-labels-to-existing-files)
      - [Adding new files](#adding-new-files)
    + [Removing files or labels from the dataset](#removing-files-or-labels-from-the-dataset)
  * [Processing the data and runing basic analyses](#processing-the-data-and-runing-basic-analyses)
    + [Smoothing and creating a basic report](#smoothing-and-creating-a-basic-report)
    + [Calculating Transitionmatrices and ploting behavior flow](#calculating-transitionmatrices-and-ploting-behavior-flow)
    + [Mapping different label groups to each other](#mapping-different-label-groups-to-each-other)
  * [Grouped analyses](#grouped-analyses)
    + [Adding metadata](#adding-metadata)
    + [statiatical two group comparisons and Behavior Flow Analysis](#statiatical-two-group-comparisons-and-behavior-flow-analysis)
    + [Behavior Flow Fingerprinting](#behavior-flow-fingerprinting)
    + [Behavior Flow Fingerprinting across multiple datasets](#behavior-flow-fingerprinting-across-multiple-datasets)
- [Functions Glossary](#functions-glossary)

    
Getting Started
===========================================
This repository is about ..

Installing Jupyter Notebook
------------------------

### Using Conda

For new users, we **highly recommend** [installing Anaconda](https://www.anaconda.com/download).
Anaconda conveniently
installs Python, the Jupyter Notebook, and other commonly used packages for
scientific computing and data science.

Use the following installation steps:

1. Download [Anaconda](https://www.anaconda.com/download). We recommend
   downloading Anaconda's latest Python 3 version (currently Python 3.11).

2. Install the version of Anaconda which you downloaded, following the
   instructions on the download page.

3. Congratulations, you have installed Jupyter Notebook. To run the notebook:

   
    ```bash
    jupyter notebook

   See [Running the Notebook](https://docs.jupyter.org/en/latest/running.html#running) for more details.


Estimation of Genotype Likelihoods (GLs)
------------------------
Bu tutorial'da oryx datasi uzerinden gidicez, onun angsd kodunu acikliycaz.

### Important points to consider
Angsd ile ilgili. Missingness, minor allela freq onemi, citation ile guideline cok guzel buna bakabilirsiniz diycez.

PCA using GLs
------------------------




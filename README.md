- [Getting Started](#getting-started)
  * [Installing Jupyter Notebook](#installing-jupyter-notebook)
      + [Using Conda](#using-conda)
- [Tutorial](#tutorial)
  * [Generating Input Files](#generating-input-files)
    + [Estimation of Genotype Likelihoods](#Estimation-of-Genotype-Likelihoods)
    + [Principal Component Analysis using Genotype Likelihoods](#Principal-Component-Analysis-using-Genotype-Likelihoods)
    + [Important Points to Consider](#Important-Points-to-Consider)

  * [Running UMAP and t-SNE with the Principal Components of Genotype Likelihoods](#running-umap)
     + [Loading Required Libraries](#Loading-libraries)
     + [Creating Color Dictionaries for Different Populations](#creating-color-dicts)
     + [Loading the Population Data and Covariance Matrix](#loading-population-data)
     + [Performing Elbow Method for the Selection of Principal Components](#elbow-method)
     + [Performing UMAP and tSNE with a Grid Search](#umap-tsne)
     + [Visualizing the Results](#visualization)

  * [UMAP and tSNE without PCA Initialization](#alternative-without-pca)

 - [Citation](#citation)



    
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




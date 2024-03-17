- [Getting Started](#Getting-Started)
  * [Installing Jupyter Notebook](#Installing-Jupyter-Notebook)
      + [Using Conda](#Using-Conda)
- [Tutorial](#Tutorial)
  * [Generating Input Files](#Generating-Input-Files)
    + [Estimation of Genotype Likelihoods](#Estimation-of-Genotype-Likelihoods)
    + [Principal Component Analysis using Genotype Likelihoods](#Principal-Component-Analysis-using-Genotype-Likelihoods)
    + [Important Points to Consider](#Important-Points-to-Consider)
  * [Running t-SNE and UMAP with the Principal Components of Genotype Likelihoods](#Running-t-SNE-and-UMAP-with-the-Principal-Components-of-Genotype-Likelihoods)
     + [Loading Required Libraries](#Loading-Required-Libraries)
     + [Creating Color Dictionaries for Different Populations](#Creating-Color-Dictionaries-for-Different-Populations)
     + [Loading the Population Data and Covariance Matrix](#Loading-the-Population-Data-and-Covariance-Matrix)
     + [Performing Elbow Method for the Selection of Principal Components](#Performing-Elbow-Method-for-the-Selection-of-Principal-Components)
     + [Performing t-SNE and UMAP with a Grid Search](#Performing-t-SNE-and-UMAP-with-a-Grid-Search)
     + [Visualizing the Results](#Visualizing-the-Results)
  * [t-SNE and UMAP without PCA Initialization](#t-SNE-and-UMAP-without-PCA-Initialization)
 - [Citation](#Citation)

    
Getting Started
===========================================
This repository is about ..

Installing Jupyter Notebook
------------------------

### Using Conda

For new users, we highly recommend [installing Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and creating a conda environment which includes the required packages to run the Jupyter Notebook and the subsequent analysis detailed in the following tutorial.

Use the following installation steps:

1. Download [Anaconda](https://www.anaconda.com/download). We recommend
   downloading Anaconda's latest Python 3 version (currently Python 3.11).

2. Install the version of Anaconda which you downloaded, following the
   instructions on the download page.

3. Congratulations, you have installed Jupyter Notebook. To run the notebook:

   
    ```bash
    jupyter notebook

See [Running the Notebook](https://docs.jupyter.org/en/latest/running.html#running) for more details.

Tutorial
===========================================

Generating Input Files
------------------------
### Estimation of Genotype Likelihoods
In this tutorial, we will use SO_2x dataset and explain related analysis steps for a single input file. 


Bu tutorial'da oryx datasi uzerinden gidicez, onun angsd kodunu acikliycaz.

### Principal Component Analysis using Genotype Likelihoods

### Important Points to Consider
Angsd ile ilgili. Missingness, minor allela freq onemi, citation ile guideline cok guzel buna bakabilirsiniz diycez.

Running t-SNE and UMAP with the Principal Components of Genotype Likelihoods
------------------------
### Loading Required Libraries
In the following sections we will go over a single input covariance matrix obtained with (SO_2x).
For running subsequent analyses with multiple covariance matrices you can refer to XXX.ipynb

```python
# Import libraries. 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Dimension reduction tools
from sklearn.decomposition import PCA as PCA
from sklearn.manifold import TSNE
import umap
```

### Creating Color Dictionaries for Different Populations

### Loading the Population Data and Covariance Matrix

### Performing Elbow Method for the Selection of Principal Components

### Performing t-SNE and UMAP with a Grid Search

### Visualizing the Results

t-SNE and UMAP without PCA Initialization
------------------------

Citation
===========================================


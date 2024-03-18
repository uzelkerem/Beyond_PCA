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
     + [Creating Color Palette for Different Populations](#Creating-Color-Palette-for-Different-Populations)
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
In the following sections we will go over a single input covariance matrix obtained with (SO_2x). For the corresponding Jupyter Notebook refer to XX1.ipynb

For running subsequent analyses with multiple covariance matrices you can refer to XXX2.ipynb

The first step is to load required libraries. For this, we need to activate the conda environment (env_name):
```bash
conda activate env_name
```

Next, we need to initiate the jupyter notebook:
```bash
jupyter notebook XX1.ipynb
```

Then, a browser windor will be opened. From there we will select the conda environment we created as a kernel:

![Kernel Selection](images/ss1.png)

In this jupyter notebook the first code block is for loading required libraries:

```python
# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from kneed import KneeLocator

# Dimension reduction tools
from sklearn.decomposition import PCA as PCA
from sklearn.manifold import TSNE
import umap
```

### Creating Color Palette for Different Populations

A custom color palette is created for Oryx populations as in (REF):
```python
#for Oryx datasets:
custom_palette = {
    "EAD_A": "#cada45", #green
    "EAD_B": "#d4a2e1", #purple
    "EEP": "#55e0c6", #blue
    "USA": "#f0b13c", #orange
}
```

### Loading the Population Data and Covariance Matrix

Covariance matrix and population data for Oryx samples are loaded:

```python
#load population data
population_names = pd.read_csv('input_files/oryx_pop_info_sorted_46_final.txt', sep='\t', header=0)
#load the covariance matrix
filename='input_files/oryx_2xyh_1K.cov'
cov_mat= pd.read_csv(filename, sep=' ', header=None)
#Generating the pandas dataframe called Data_Struct
Data_Struct=population_names
```

### Performing Elbow Method for the Selection of Principal Components

First the functions to calculate the 'elbow point' (using _kneed_, [Satopaa et al., 2011](https://github.com/arvkevi/kneed/tree/v0.8.5)) and scree plot functions are defined:

```python
# Function to plot the scree plot
def plot_scree(explained_variance,filename_title,elbow_point):
    plt.figure(figsize=(8, 4))
    # Convert to a simple list if it's not already
    explained_variance = list(explained_variance)
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.axvline(x=elbow_point, color='r', linestyle='--')
    plt.text(elbow_point + 0.1, max(explained_variance) * 0.9, f'Elbow: {elbow_point}', color='red', verticalalignment='center')
    plt.title(f'Scree Plot | {filename_title}')
    plt.xlabel('Number of Components')
    plt.ylabel('Variance Explained')
    plt.grid()
    plt.show()

# Function to find the elbow point
def find_elbow_point(explained_variance, sensitivity=1.0):
    explained_variance = list(explained_variance)
    kneedle = KneeLocator(range(1, len(explained_variance) + 1), explained_variance, 
                          curve='convex', direction='decreasing', 
                          S=sensitivity, interp_method='polynomial')
    return kneedle.elbow
```
Next, principal component analysis is performed:

```python
#Calculate PCA
#convert covariance matrix to numpy array
cov_mat_np=cov_mat.to_numpy()

# calculate eigen vectors and eigen values from the initial covariance matrix
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat_np)
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
feature_vector = np.hstack([eigen_pairs[i][1][:, np.newaxis] for i in range(len(eigen_vals))])
principal_components = cov_mat_np.dot(feature_vector) 

# sorting them from largest to smallest
idx = eigen_vals.argsort()[::-1]   
eigenValues = eigen_vals[idx]
eigenVectors = eigen_vecs[:,idx]

# calculating the total explained variance
expl_pre=eigenValues/sum(eigenValues)
expl=np.cumsum(expl_pre)

expl_df=pd.DataFrame(expl_pre*100,columns=['explained_variance'])
expl_df['cumulative_expl']=expl*100
expl_df.set_index(np.arange(1, eigenVectors.shape[0] + 1), inplace=True)
```
Finally, elbow point is calculated and plotted together with the scree plot:

```python
# Plot the scree plot
plot_filename = f'scree_plot_{filename_title}.png'

 # Find the elbow point
elbow_point = find_elbow_point(expl_df['explained_variance'])
print("Optimal number of principal components):", elbow_point)

plot_scree(expl_df['explained_variance'],'Oryx 2x',elbow_point)
```

![Oryx ScreePlot](images/OryxScreePlot.png)

### Performing t-SNE and UMAP with a Grid Search
Using the number of PCs with the elbow method, t-SNE and UMAP is performed with a combination of parameters (Grid search).

First, the parameters are defined:
```python
#define number of principal components to be used as inputs for UMAP and t-SNE calculations
n_pc = elbow_point
#define parameter range for t-SNE
perplexity_values=(5,10,23)
#define parameter space for UMAP
mindists=(0.01,0.1,0.5)
n_neighbors_nums=(5,10,23)
```
Next, t-SNE and UMAP is performed:

```python
#t-SNE calculation
for perp in perplexity_values:
    np.random.seed(111)
    proj_tsne = TSNE(n_components=2,perplexity=perp).fit_transform(principal_components[:,:n_pc])
    Data_Struct['tSNE-1 perp'+str(perp)]=proj_tsne[:,0]
    Data_Struct['tSNE-2 perp'+str(perp)]=proj_tsne[:,1]

#UMAP calculation
for nn in n_neighbors_nums:
    for mind in mindists:
        np.random.seed(111)
        proj_umap = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=mind).fit_transform(principal_components[:,:n_pc])
        Data_Struct['UMAP-1 numn'+str(nn)+' mindist'+str(mind)]=proj_umap[:,0]
        Data_Struct['UMAP-2 numn'+str(nn)+' mindist'+str(mind)]=proj_umap[:,1]
```

### Visualizing the Results
Finally, the results are visualized:
```python
# Ensure the lengths of perplexity_values and mindists are equal
if len(perplexity_values) != len(mindists):
    raise ValueError("The number of perplexity values must be equal to the number of minimum distance values.")

# Determine the number of rows and columns for the subplot grid
n_rows = len(perplexity_values)
n_cols = 1 + len(n_neighbors_nums)  # Adding 1 for t-SNE and the rest for UMAP

# Create the subplot grid
fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
fig.suptitle("input:" + filename + ' / Top ' + str(n_pc) + 'PCs are used (' + str(round(expl_df['cumulative_expl'][n_pc-1], 1)) + '%)', fontsize=14)

# t-SNE plots (first column)
for i, perp in enumerate(perplexity_values):
    sns.scatterplot(ax=axs[i, 0], data=Data_Struct, x='tSNE-1 perp' + str(perp), y='tSNE-2 perp' + str(perp), s=500, hue='Population', palette=custom_palette, legend=False)
    axs[i, 0].set_xlabel('tSNE-1')
    axs[i, 0].set_ylabel('tSNE-2')
    axs[i, 0].set_xticks([])
    axs[i, 0].set_yticks([])
    # Set the title differently for the first plot
    if i == 0:
        axs[i, 0].set_title('tSNE / Perplexity = ' + str(perp))
    else:
        axs[i, 0].set_title('Perplexity = ' + str(perp))


# UMAP plots (next columns)
for j, nn in enumerate(n_neighbors_nums):
    for i, mind in enumerate(mindists):
        is_last_plot = (i == len(mindists) - 1) and (j == len(n_neighbors_nums) - 1)
        sns.scatterplot(ax=axs[i, j + 1], data=Data_Struct, x='UMAP-1 numn' + str(nn) + ' mindist' + str(mind), y='UMAP-2 numn' + str(nn) + ' mindist' + str(mind), s=500, hue='Population', palette=custom_palette, legend=is_last_plot)
        
        if i == 0:
            axs[i, j + 1].set_title('UMAP / n_neighbours = ' + str(nn))
        axs[i, j + 1].set_xlabel('UMAP-1')

        # Set the y-axis label differently for the first column of UMAP plots
        if j == 0:
            axs[i, j + 1].set_ylabel('min_dist = ' + str(mind))
        else:
            axs[i, j + 1].set_ylabel('UMAP-2')

        axs[i, j + 1].set_xticks([])
        axs[i, j + 1].set_yticks([])
        
        # Adjust the legend for the last UMAP plot (bottom-right)
        if is_last_plot:
            axs[i, j + 1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Population')

```
![Oryx GridPlot](images/Oryx_GridPlot.png)

t-SNE and UMAP without PCA Initialization
------------------------


Citation
===========================================


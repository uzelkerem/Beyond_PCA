- [Getting Started](#getting-started)
  + [Installing Jupyter Notebook](#installing-jupyter-notebook)
      - [Using Conda](#installing-jupyter-notebook-conda)
      - [Using pip](#installing-jupyter-notebook-pip)
- [Tutorial](#tutorial)
  * [building the data object](#building-the-data-object)
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

### Installing Jupyter Notebook

#### Using Anaconda and conda

For new users, we **highly recommend** `installing Anaconda
<https://www.anaconda.com/download>`_. Anaconda conveniently
installs Python, the Jupyter Notebook, and other commonly used packages for
scientific computing and data science.

Use the following installation steps:

1. Download `Anaconda <https://www.anaconda.com/download>`_. We recommend
   downloading Anaconda's latest Python 3 version (currently Python 3.9).

2. Install the version of Anaconda which you downloaded, following the
   instructions on the download page.

3. Congratulations, you have installed Jupyter Notebook. To run the notebook:

   .. code-block:: bash

       jupyter notebook

   See :ref:`Running the Notebook <running>` for more details.

.. _existing-python-new-jupyter:

This package enables efficient meta-analyses of unsupervised behavior analysis results. It builds a data object containing label data from multiple recordings/samples, with labeling data from different sources and metadata describing experimental design and grouping variables. The data object can be analyzed using helper functions for clustering-to-clustering mapping, Behavior Flow Analysis (BFA; statistical two group analyses), Behavior Flow Fingerprinting (BFF; 2d embedding with a per sample resolution), and more.

#### Using pip

.. important::

    Jupyter installation requires Python 3.3 or greater, or
    Python 2.7. IPython 1.x, which included the parts that later became Jupyter,
    was the last version to support Python 3.2 and 2.6.

As an existing Python user, you may wish to install Jupyter using Python's
package manager, :term:`pip`, instead of Anaconda.

.. _python-using-pip:

First, ensure that you have the latest pip;
older versions may have trouble with some dependencies:

.. code-block:: bash

    pip3 install --upgrade pip

Then install the Jupyter Notebook using:

.. code-block:: bash

    pip3 install jupyter

(Use ``pip`` if using legacy Python 2.)

Congratulations. You have installed Jupyter Notebook. See
:ref:`Running the Notebook <running>` for more details.


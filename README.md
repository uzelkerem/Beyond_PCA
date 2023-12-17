- [Behavior Flow](#behavior-flow)
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

    
Behavior Flow
===========================================

This package enables efficient meta-analyses of unsupervised behavior analysis results. It builds a data object containing label data from multiple recordings/samples, with labeling data from different sources and metadata describing experimental design and grouping variables. The data object can be analyzed using helper functions for clustering-to-clustering mapping, Behavior Flow Analysis (BFA; statistical two group analyses), Behavior Flow Fingerprinting (BFF; 2d embedding with a per sample resolution), and more.

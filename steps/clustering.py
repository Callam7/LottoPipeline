## Modified By: Callam
## Project: Lotto Predictor
## Purpose of File: Perform K-Means Clustering on Number Frequencies
## Description:
## This file applies K-Means clustering to identify patterns or groupings in the frequency data of main lottery numbers.
## Clustering results, including labels for each number and cluster centroids, are stored in the pipeline for use in 
## subsequent predictive modeling steps.

import numpy as np  # For numerical operations and data formatting
from sklearn.cluster import KMeans  # For performing K-Means clustering
import logging  # For logging warnings and informational messages

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def kmeans_clustering_and_correlation(pipeline, n_clusters=5):
    """
    Performs K-Means clustering on the number frequency data to identify patterns or groupings.
    Adds the cluster labels and centroids to the pipeline for use in subsequent steps.

    Parameters:
    - pipeline (DataPipeline): The pipeline object containing shared data across steps.
    - n_clusters (int): The number of clusters to form. Default is 5.

    Returns:
    - None: Adds "clusters" and "centroids" to the pipeline for downstream use.
    """

    # Step 1: Retrieves number frequency data from the pipeline
    ## The "number_frequency" key is expected to contain a normalized 1D array of size 40.
    frequency_data = pipeline.get_data("number_frequency")
    if frequency_data is None:
        ## Case: No frequency data available
        logging.warning("No frequency data available for clustering.")
        return  # Exit the function as there's no data to process

    # Step 2: Validate and reshape frequency data
    ## Converts frequency data to a NumPy array for compatibility with scikit-learn
    frequency_data = np.array(frequency_data)

    ## Ensures the data is in the expected 1D format with 40 elements
    if frequency_data.ndim != 1 or frequency_data.size != 40:
        logging.error("Frequency data is not in the expected format. Expected a 1D array of size 40.")
        return  # Exit the function if data is invalid

    ## Reshapes the data to a 2D array as required by K-Means (rows: data points, columns: features)
    data = frequency_data.reshape(-1, 1)

    try:
        # Step 3: Perform K-Means clustering
        ## Initializes the KMeans model with specified number of clusters
        ## A fixed random_state is used for reproducibility of clustering results
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
        ## Fits the KMeans model to the reshaped frequency data
        kmeans.fit(data)

        ## Retrieves the cluster labels for each number and the centroids of each cluster
        labels = kmeans.labels_  # Labels indicate which cluster each number belongs to
        centroids = kmeans.cluster_centers_.flatten()  # Centroids represent the center of each cluster

    except Exception as e:
        # Handles any errors that occur during the clustering process
        logging.error(f"Error during KMeans clustering: {e}")
        return  # Exit the function if an error occurs

    # Step 4: Store clustering results in the pipeline
    ## Adds cluster labels to the pipeline for use in downstream steps
    pipeline.add_data("clusters", labels)
    ## Add cluster centroids to the pipeline for use in downstream steps
    pipeline.add_data("centroids", centroids)

    # Step 5: Log a success message
    logging.info("Clustering completed successfully.")
    ## The clustering results are now ready for use in subsequent steps of the pipeline.



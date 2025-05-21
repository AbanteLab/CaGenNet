###################################################################################
# This file contains functions for evaluating the latent representation of the
# model. It includes functions for:
# 1. Evaluating the latent SVM classifier
# 2. Computing the Adjusted Rand Index (ARI) for HDBSCAN clustering
# 3. Computing local entropy of neuron types
##################################################################################

import numpy as np

from sklearn.svm import SVC

from scipy.stats import entropy

from sklearn.cluster import HDBSCAN,DBSCAN
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score

# Evaluate latent SVM classifier
def evaluate_latent_svm(z_train, y_train, z_val, y_val):
    """
    Trains a linear SVM on z_train/y_train and evaluates on z_val/y_val.
    Returns balanced accuracy, macro precision, recall, and f1-score.
    """
    
    # Create a pipeline with feature scaling and SVM classifier
    pipeline = make_pipeline(StandardScaler(), SVC())
    
    # Define hyperparameter grid for grid search
    param_grid = {
        'svc__C': [0.1, 1, 10],              # Regularization parameter
        'svc__kernel': ['linear'],           # Use linear kernel
        'svc__gamma': ['scale', 'auto']      # Gamma parameter (not used for linear, but included for completeness)
    }
    
    # Perform grid search with cross-validation to find best hyperparameters
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=4)
    grid_search.fit(z_train, y_train)
    
    # Get the best model from grid search
    best_model = grid_search.best_estimator_
    
    # Predict labels for validation set
    y_pred = best_model.predict(z_val)
    
    # Compute balanced accuracy
    accuracy = balanced_accuracy_score(y_val, y_pred)
    
    # Get classification report as a dictionary
    report = classification_report(y_val, y_pred, output_dict=True)
    macro_avg = report['macro avg']
    
    # Extract macro-averaged precision, recall, and F1-score
    precision = macro_avg['precision']
    recall = macro_avg['recall']
    f1 = macro_avg['f1-score']
    
    return accuracy, precision, recall, f1

# get ari for hdbscan
def get_hdbscan_ari(z, y):  
    """
    Performs HDBSCAN clustering on the given data and computes the Adjusted Rand Index (ARI) with respect to true labels.
    Args:
        z (array-like): Feature matrix or embeddings to cluster, shape (n_samples, n_features).
        y (array-like): Ground truth labels for each sample, shape (n_samples,).
    Returns:
        tuple:
            - labels_ (np.ndarray): Cluster labels assigned by HDBSCAN for each sample.
            - ari (float): Adjusted Rand Index score comparing predicted clusters to true labels.
    Raises:
        ValueError: If input arrays have incompatible shapes.
    Help:
        This function uses the HDBSCAN algorithm to cluster the input data `z` and evaluates the clustering quality using the Adjusted Rand Index (ARI) against the provided ground truth labels `y`. The ARI is a measure of the similarity between two clusterings, with a value close to 1.0 indicating a high degree of agreement.
    """
    
    # DBSCAN clustering
    clustering = HDBSCAN().fit(z)

    # return labels,ari    
    return clustering.labels_, adjusted_rand_score(y, clustering.labels_)

# compute local entropy of neuron type
def compute_local_entropy(latents, labels, k=10):
    """
    Computes the median local entropy of the neuron type for each latent point.

    Parameters:
        latents (np.ndarray): Latent embedding of shape (n_samples, n_dims)
        labels (list or np.ndarray): Categorical labels (e.g., RS, FS, LTS)
        k (int): Number of neighbors to consider (excluding the central point)

    Returns:
        local_entropies (np.ndarray): Local entropy for each point
    """

    # Ensure latents is a 2D array
    labels = np.array(labels)
    uniq_labels = np.unique(labels)
    n_classes = len(uniq_labels)

    # Encode labels as integers for easier counting
    label_to_idx = {label: i for i, label in enumerate(uniq_labels)}
    y_int = np.array([label_to_idx[label] for label in labels])

    # Neighbors (excluding the point itself)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(latents)
    _, indices = nbrs.kneighbors(latents)

    local_entropies = []
    for i, neighbors in enumerate(indices):
        neighbor_labels = y_int[neighbors[1:]]  # exclude the point itself
        counts = np.bincount(neighbor_labels, minlength=n_classes)
        probs = counts / counts.sum()
        local_entropies.append(entropy(probs, base=2))  # entropy in bits

    return np.median(local_entropies)

# Define a function to find the optimal epsilon for DBSCAN clustering
def find_optimal_dbscan_epsilon(z, y):  
    
    ari = []
    epsilons = [round(i, 2) for i in np.arange(0.05, 2.0, 0.05)]
    
    # loop through epsilons
    for epsilon in epsilons:
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=epsilon, min_samples=20).fit(z) 

        # Get cluster labels
        clusters = clustering.labels_
        
        # Assign each noise point (-1) a unique cluster label
        noise_indices = np.where(clusters == -1)[0]
        for i, idx in enumerate(noise_indices):
            clusters[idx] = max(clusters) + 1 + i
        
        # append adjusted rand index
        ari.append(adjusted_rand_score(y, clusters))
        
    # get optimal eps
    opt_eps = epsilons[ari.index(max(ari))]
    
    # fit dbscan with optimal eps
    clustering = DBSCAN(eps=opt_eps, min_samples=20).fit(z)
    clusters = clustering.labels_
    
    return clusters, opt_eps, max(ari)

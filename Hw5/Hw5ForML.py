# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:59:05 2024

@author: Jiajun Li
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score, euclidean_distances
from scipy.spatial import distance
from scipy.spatial.distance import squareform
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
data = pd.read_csv("./wines.csv")

#pre-process the data so that it is standardized scale the data
data = (data - data.mean())/data.std()
'''
#How many Eignevalues are above one? 
PCA = PCA(whiten = True)
PCA.fit(data)

Eigenvalues = PCA.explained_variance_

count = 0

for i in range(len(Eigenvalues)):
    if Eigenvalues[i] > 1:
        count += 1

print(f"The number of Eigenvalues above 1 is {count}")

Eigenvectors = PCA.components_

data_pca = PCA.transform(data)
Component1 = data_pca[:,0]
Component2 = data_pca[:,1]
plt.scatter(Component1,Component2)
plt.ylabel('Component 2')
plt.xlabel('Component 1')
plt.title('2 Components PCA')
plt.show()

pca_explained_var = PCA.explained_variance_ratio_
PCV1 = pca_explained_var[0]
PCV2 = pca_explained_var[1]
PCV3 = pca_explained_var[2]
print(f"The variance explained by these two dimension are {PCV1} and {PCV2} and {PCV3}")
'''


#Question2 

#initialize the array to store the KL divergence
'''
KLD = np.zeros(shape = (146))
perplexity = np.zeros(shape = (146))
for i in range(5,151):
    data_TSNE = TSNE(n_components = 2, perplexity= i).fit(data)
    KLD[i-5] = data_TSNE.kl_divergence_
    perplexity[i-5] = i

plt.plot(perplexity, KLD)
plt.ylabel('KL-divergence')
plt.xlabel('Perplexity')
plt.title('Perplexity VS KL-divergence')
plt.show()
'''

'''
tsne = TSNE(n_components = 2, perplexity= 20, random_state = 1000000, n_iter = 1200)
tsne.fit(data)
data_tsne = tsne.fit_transform(data)
plt.scatter(data_tsne[:,0], data_tsne[:,1])
plt.ylabel('Component 2')
plt.xlabel('Component 1')
plt.title('2 Components TSNE')
plt.show()
print("The KL-divergence of t-SNE is:", tsne.kl_divergence_)
'''


#Question 3 Let us do a MDS
#Due to the nature of sklearn, it is not appropriate to calculate the stress by mds.stress_ so 
#I have to calculate it manually
#This is the original distance
'''
Dis_origin = euclidean_distances(data)
#Now I want to find the mds embedding
mds = MDS(n_components=3, n_init=100, random_state = 10000, max_iter= 1000)
#here is the created embedding
low_data = mds.fit_transform(data)
Dis_low = euclidean_distances(low_data)
Numerator = np.sum((Dis_low - Dis_origin)**2)
stress = np.sqrt(Numerator /(np.sum(Dis_origin**2)))
print(stress)


#Now I seek to plot this solution
plt.scatter(low_data[:,0], low_data[:,1])
plt.ylabel('Component 2')
plt.xlabel('Component 1')
plt.title('2 Components MDS')
plt.show()
'''

'''
#Question 4 
#use the TSNE because the clustering works the best 
TSNE = TSNE(n_components= 2, perplexity = 20, random_state = 100000)

data_tsne = TSNE.fit_transform(data)

for i in range(2, 11): #idk how many different types of wine I have in here
    cluster = KMeans(n_clusters = i, random_state = 1000)
    labels = cluster.fit_predict(data_tsne)
    s_average  = silhouette_score(data_tsne, labels)
    print("For n_clusters =", i, "The average silhouette_score is :", s_average )

#now we can conclude the the best number of clustering would be 3
cluster = KMeans(n_clusters = 3, random_state = 1000)
labels = cluster.fit_predict(data_tsne)
colors = cm.nipy_spectral(labels.astype(float) / 3)
center = cluster.cluster_centers_

# Setting up the plot
plt.figure(figsize=(10, 8))

# Scatter plot of the t-SNE data points
colors = cm.nipy_spectral(labels.astype(float) / 3)
scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], s=50, c=colors, alpha=0.6)  # Plot data points with color coding

# Annotating the plot with the cluster centers
for i, center_coord in enumerate(center):
    plt.scatter(center_coord[0], center_coord[1], s=200, c='red', marker='o', edgecolors='black')  # Plot center with a circle
    plt.text(center_coord[0], center_coord[1], str(i), color='black', fontsize=12, ha='center', va='center')

# Adding annotations and labels
plt.title('t-SNE visualization of data colored by KMeans clustering')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.colorbar(scatter, label='Cluster label')

# Show the plot
plt.show()

#Now find the total sum of distance of all points to their own centers
# Assuming X_TSNE is your t-SNE transformed data
cluster = KMeans(n_clusters=3, random_state=10)
cluster.fit(data_tsne)  # Fit the model

# Using the transform method to get distances to cluster centers
distances_to_center = cluster.transform(data_tsne)

# Sum the minimum distance for each point to its cluster center
total_distance = sum(min(distances) for distances in distances_to_center)

print("The total sum of the distance of all points to their respective clusters centers is:", total_distance)
'''


#Question 5 code
#The following code is from the lab
def make_scatter_plot (df, x="x_1", y="x_2", hue="label",
                       palette={0: "red", 1: "olive", 2: "blue", 3: "green"},
                       size=5,
                       centers=None):
    if (hue is not None) and (hue in df.columns):
        sns.lmplot (x=x, y=y, hue=hue, data=df, palette=palette,
                    fit_reg=False)
    else:
        sns.lmplot (x=x, y=y, data=df, fit_reg=False)

    if centers is not None:
        plt.scatter (centers[:,0], centers[:,1],
                     marker=u'*', s=500,
                     c=[palette[0]])

def make_scatter_plot2 (df, x="x_1", y="x_2", hue="label", size=5):
    if (hue is not None) and (hue in df.columns):
        sns.lmplot (x=x, y=y, hue=hue, data=df,
                    fit_reg=False)
    else:
        sns.lmplot (x=x, y=y, data=df, fit_reg=False)
        
def region_query (p, eps, X):
    # These lines check that the inputs `p` and `X` have
    # the right shape.
    _, dim = X.shape
    assert (p.shape == (dim,)) or (p.shape == (1, dim)) or (p.shape == (dim, 1))
    
    return np.linalg.norm (p - X, axis=1) <= eps

def index_set (y):
    """
    Given a boolean vector, this function returns
    the indices of all True elements.
    """
    assert len (y.shape) == 1

    return set (np.where (y)[0])

def find_neighbors (eps, X):
    m, d = X.shape
    neighbors = [] # Empty list to start
    for i in range (len (X)):
        n_i = index_set (region_query (X[i, :], eps, X))
        neighbors.append (n_i)
    assert len (neighbors) == m
    return neighbors

def find_core_points (s, neighbors):
    assert type (neighbors) is list
    assert all ([type (n) is set for n in neighbors])
    
    core_set = set ()
    for i, n_i in enumerate (neighbors):
        if len (n_i) >= s:
            core_set.add (i)
    return core_set

def expand_cluster (p, neighbors, core_set, visited, assignment):
    # Assume the caller performs Steps 1 and 2 of the procedure.
    # That means 'p' must be a core point that is part of a cluster.
    assert (p in core_set) and (p in visited) and (p in assignment)
    
    reachable = set (neighbors[p])  # Step 3
    while reachable:
        q = reachable.pop () # Step 4

        if q not in visited:
            visited.add (q) # Mark q as visited
            if q in core_set:
                reachable |= neighbors[q]
        if q not in assignment:
            assignment[q] = assignment[p]
        
    # This procedure does not return anything
    # except via updates to `visited` and
    # `assignment`.
    
def dbscan (eps, s, X):
    clusters = []
    point_to_cluster = {}
    
    neighbors = find_neighbors (eps, X)
    core_set = find_core_points (s, neighbors)
    
    assignment = {}
    next_cluster_id = 0

    visited = set ()
    for i in core_set: # for each core point i
        if i not in visited:
            visited.add (i) # Mark i as visited
            assignment[i] = next_cluster_id
            expand_cluster (i, neighbors, core_set,
                            visited, assignment)
            next_cluster_id += 1

    return assignment, core_set



tsne = TSNE(n_components=2, perplexity=20, learning_rate=200, random_state=42)
data_tsne = tsne.fit_transform(data)

# Apply DBSCAN clustering algorithm to the t-SNE transformed data
assignment, core_set = dbscan(3.5, 8, data_tsne)
num_clusters = max(assignment.values()) + 1 if assignment else 0

# Display the results of the clustering
print("Number of core points:", len(core_set))
print("Number of clusters:", num_clusters)
print("Number of unclassified points:", len(data) - len(assignment))

# Function to plot the clustering results with labels
def plot_labels(df, labels):
    df_labeled = df.copy()
    df_labeled['label'] = labels
    make_scatter_plot2(df_labeled)

# Assign labels to each point; default to -1 for unclassified points
labels = [-1] * len(data_tsne)
for i, cluster_id in assignment.items():
    labels[i] = cluster_id

# Create a DataFrame for the t-SNE results and plot
data_tsne_df = pd.DataFrame(data_tsne, columns=["x_1", "x_2"])
plot_labels(data_tsne_df, labels)
plt.title("2 components t-SNE using Lab's DBSCAN")
















import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import rand_score


def adjust_kmeans_labels(labels):
    for i in range(len(labels)):
        if labels[i] == 0:
            labels[i] = 1
        elif labels[i] == 1:
            labels[i] = 2
        elif labels[i] == 2:
            labels[i] = 3
    return labels


def kmeans_clustering(X, n_clusters=3):
    num_clusters_to_try = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    silhouette_scores = []
    inertia_values = []

    for num_clusters in num_clusters_to_try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        inertia = kmeans.inertia_
        inertia_values.append(inertia)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(num_clusters_to_try, silhouette_scores, marker='o')
    plt.xlabel('Cluster number')
    plt.ylabel('Silhouette Score')
    plt.title('Evaluation of the number of clusters with Silhouette Score')

    plt.subplot(1, 2, 2)
    plt.plot(num_clusters_to_try, inertia_values, marker='o')
    plt.xlabel('Cluster number')
    plt.ylabel('Inertia')
    plt.title('Evaluation of the number of clusters with Inertia')
    plt.savefig('../plots/unsupervised_learning.png')
    plt.close()

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(df)
    return kmeans.labels_


df = pd.read_csv('../Dataset/new_dataset.csv')
df.drop('accident_index', axis=1, inplace=True)
y = df['accident_severity']
df.drop('accident_severity', axis=1, inplace=True)

label_kmeans = kmeans_clustering(df)
label_kmeans = adjust_kmeans_labels(label_kmeans)
s = "KMeans\n" + "y Labels Count:\n" + str(y.value_counts().sort_index()) + "\n" + "KMeans Labels Count:\n" + str(
    pd.Series(label_kmeans).value_counts().sort_index()) + "\n" + "Rand Index: " + str(
    rand_score(y, label_kmeans))

file = open("../report/unsupervised_learning.txt", "w")
file.write(s)
file.close()

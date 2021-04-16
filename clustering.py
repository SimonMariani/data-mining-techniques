from utils import load_object
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator



def aggregate_participants_by_mean():
    dfs = load_object("data_processed/subdata_pr_su.pkl")[0]
    averages_df = []
    for df in dfs:
        averages_per_variable = pd.DataFrame(df).mean(axis=0)
        averages_df.append(averages_per_variable)

    return pd.DataFrame(averages_df)


def perform_elbow_method(data):
    sse = []
    for k in range(1, 28):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

    # Locating the kink
    elbow = KneeLocator(range(1, 28), sse, curve="convex", direction="decreasing")
    print("Number of Optimal Clusters: ", elbow.elbow)

    # Plotting
    plt.plot(range(1, 28), sse)
    plt.xticks(range(1, 28))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Within SSE")
    plt.grid()
    plt.show()

    return elbow.elbow


def calculate_silhouette(data):
    silhouette = []
    for k in range(2, 27):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        score = silhouette_score(data, kmeans.labels_)
        silhouette.append(score)

    # Plotting
    plt.plot(range(2, 27), silhouette)
    plt.xticks(range(2, 27))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid()
    plt.show()

    return np.where(silhouette == np.amax(silhouette))[0][0] + 2


def cluster_by_kmeans(k_clusters, data):
    kmeans = KMeans(n_clusters=k_clusters)
    kmeans.fit(data)
    labels = kmeans.labels_
    data["Cluster"] = labels+1

    return data


def plot_clusters(data):
    labels = np.unique(data["Cluster"])
    cols = data.shape[1] - 1
    for c in range(1, cols):
        plt.subplot(4, 4, c)
        for i in labels:
            plt.scatter(data[data["Cluster"] == i][c], data[data["Cluster"] == i][0])
    plt.legend(labels)
    plt.show()

def summarize_clusters(data):
    labels = np.unique(data["Cluster"])
    for i in labels:
        data_cluster = data[data["Cluster"] == i]
        print("Cluster", i)
        print(data_cluster.describe())
        print("\n")


def main():
    #load_object("data_processed/subdata_pr_su.pkl")
    #dfs = pd.concat(, axis=0)
    #dfs.to_csv('all_data.csv')


    np.random.seed(1234)
    pd.set_option('max_columns', None)
    aggregated_participants = aggregate_participants_by_mean()
    aggregated_participants.to_csv('data_processed/aggregated_participants.csv')
    k_opt_elbow = perform_elbow_method(aggregated_participants)
    k_opt_silhouette = calculate_silhouette(aggregated_participants)
    print("Silhouette k: ", k_opt_silhouette)
    clustered_data = cluster_by_kmeans(k_opt_silhouette, aggregated_participants)
    plot_clusters(clustered_data)
    summarize_clusters(clustered_data)


if __name__ == "__main__":
    main()
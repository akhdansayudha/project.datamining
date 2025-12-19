from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def elbow_method(data):
    inertia = []
    for k in range(2, 10):
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(data)
        inertia.append(model.inertia_)

    plt.plot(range(2, 10), inertia, marker='o')
    plt.xlabel("Jumlah Cluster")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.show()

def kmeans_clustering(df, scaled_data, n_cluster=3):
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    score = silhouette_score(scaled_data, df['Cluster'])
    return df, score

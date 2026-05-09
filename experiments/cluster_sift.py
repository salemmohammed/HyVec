import numpy as np
from sklearn.cluster import MiniBatchKMeans
import struct

def read_fvecs(filename):
    with open(filename, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32)
    data = data.reshape(-1, dim + 1)
    return data[:, 1:].copy()

print("Loading SIFT1M base vectors...")
base = read_fvecs('sift/sift_base.fvecs')
print(f"Loaded: {base.shape[0]} vectors x {base.shape[1]} dimensions")

K = 100
print(f"Running MiniBatchKMeans with K={K}...")
kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=10000, verbose=1)
labels = kmeans.fit_predict(base)

np.save('sift/cluster_labels.npy', labels)
np.save('sift/cluster_centroids.npy', kmeans.cluster_centers_)

sizes = np.bincount(labels)
print(f"\n=== CLUSTERING RESULTS ===")
print(f"Number of clusters: {K}")
print(f"Cluster sizes: min={sizes.min()}, max={sizes.max()}, avg={sizes.mean():.0f}")
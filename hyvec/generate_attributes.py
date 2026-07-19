"""
generate_attributes.py
----------------------
Step 1 of Clustered Attributed Vector Search.

Runs MiniBatchKMeans on SIFT1M vectors.
Each cluster = one attribute.
K is set to 1000 (configurable).

Output:
  ../sift/attr_labels.npy    → cluster ID for each of the 1M vectors
  ../sift/attr_centroids.npy → 1000 centroid vectors (128-dim each)
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import struct
import os
import time

# ─── CONFIG ───────────────────────────────────────────────────────────────────
K          = 1000          # number of clusters = number of attributes
BATCH_SIZE = 10000         # K-Means mini-batch size
SIFT_PATH  = '../sift/sift_base.fvecs'
OUTPUT_DIR = '../sift/'
# ──────────────────────────────────────────────────────────────────────────────


def read_fvecs(filename):
    """Read .fvecs binary file into numpy array."""
    with open(filename, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32)
    data = data.reshape(-1, dim + 1)
    return data[:, 1:].copy()  # drop the dim header column


def main():
    print("=" * 50)
    print("  Attribute Generation via K-Means Clustering")
    print("=" * 50)

    # 1. Load data
    print(f"\nLoading SIFT1M from {SIFT_PATH}...")
    t0 = time.time()
    base = read_fvecs(SIFT_PATH)
    print(f"Loaded: {base.shape[0]:,} vectors x {base.shape[1]} dimensions")
    print(f"Load time: {time.time() - t0:.2f}s")

    # 2. Run K-Means
    print(f"\nRunning MiniBatchKMeans with K={K}...")
    print(f"Batch size: {BATCH_SIZE}")
    t1 = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=K,
        random_state=42,
        batch_size=BATCH_SIZE,
        verbose=0,
        n_init=3
    )
    labels = kmeans.fit_predict(base)
    cluster_time = time.time() - t1
    print(f"Clustering time: {cluster_time:.2f}s")

    # 3. Compute stats
    sizes = np.bincount(labels)

    print(f"\n=== ATTRIBUTE RESULTS ===")
    print(f"Number of attributes (clusters): {K}")
    print(f"Min cluster size:  {sizes.min():,}")
    print(f"Max cluster size:  {sizes.max():,}")
    print(f"Avg cluster size:  {sizes.mean():.0f}")
    print(f"Std cluster size:  {sizes.std():.0f}")

    # 4. Save outputs
    labels_path    = os.path.join(OUTPUT_DIR, 'attr_labels.npy')
    centroids_path = os.path.join(OUTPUT_DIR, 'attr_centroids.npy')

    np.save(labels_path,    labels)
    np.save(centroids_path, kmeans.cluster_centers_)

    print(f"\nSaved: {labels_path}")
    print(f"Saved: {centroids_path}")
    print("\nDone! Ready for Step 2: build_indexes.cpp")


if __name__ == '__main__':
    main()

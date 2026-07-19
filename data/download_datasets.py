"""
download_datasets.py
--------------------
Downloads all ANN-Benchmarks datasets in HDF5 format.
Source: https://ann-benchmarks.com

Each dataset includes:
  - train vectors (database)
  - test vectors (queries)
  - ground truth top-100 nearest neighbors

Usage:
  python3 download_datasets.py              # download all
  python3 download_datasets.py --only sift  # download one
"""

import os
import sys
import urllib.request

# ─── ALL DATASETS FROM ANN-BENCHMARKS ────────────────────────────────────────
DATASETS = {
    "sift": {
        "name":     "SIFT-128-Euclidean",
        "url":      "https://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "file":     "sift-128-euclidean.hdf5",
        "size":     "501MB",
        "dims":     128,
        "train":    1_000_000,
        "test":     10_000,
        "distance": "Euclidean",
        "our_use":  True,   # ← we use this one
    },
    "gist": {
        "name":     "GIST-960-Euclidean",
        "url":      "https://ann-benchmarks.com/gist-960-euclidean.hdf5",
        "file":     "gist-960-euclidean.hdf5",
        "size":     "3.6GB",
        "dims":     960,
        "train":    1_000_000,
        "test":     1_000,
        "distance": "Euclidean",
        "our_use":  False,
    },
    "glove-25": {
        "name":     "GloVe-25-Angular",
        "url":      "https://ann-benchmarks.com/glove-25-angular.hdf5",
        "file":     "glove-25-angular.hdf5",
        "size":     "121MB",
        "dims":     25,
        "train":    1_183_514,
        "test":     10_000,
        "distance": "Angular",
        "our_use":  False,
    },
    "glove-100": {
        "name":     "GloVe-100-Angular",
        "url":      "https://ann-benchmarks.com/glove-100-angular.hdf5",
        "file":     "glove-100-angular.hdf5",
        "size":     "463MB",
        "dims":     100,
        "train":    1_183_514,
        "test":     10_000,
        "distance": "Angular",
        "our_use":  False,
    },
    "fashion-mnist": {
        "name":     "Fashion-MNIST-784-Euclidean",
        "url":      "https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
        "file":     "fashion-mnist-784-euclidean.hdf5",
        "size":     "217MB",
        "dims":     784,
        "train":    60_000,
        "test":     10_000,
        "distance": "Euclidean",
        "our_use":  False,
    },
    "nytimes": {
        "name":     "NYTimes-256-Angular",
        "url":      "https://ann-benchmarks.com/nytimes-256-angular.hdf5",
        "file":     "nytimes-256-angular.hdf5",
        "size":     "301MB",
        "dims":     256,
        "train":    290_000,
        "test":     10_000,
        "distance": "Angular",
        "our_use":  False,
    },
    "lastfm": {
        "name":     "Last.fm-65-Angular",
        "url":      "https://ann-benchmarks.com/lastfm-64-dot.hdf5",
        "file":     "lastfm-64-dot.hdf5",
        "size":     "135MB",
        "dims":     65,
        "train":    292_385,
        "test":     50_000,
        "distance": "Angular",
        "our_use":  False,
    },
}
# ─────────────────────────────────────────────────────────────────────────────


def progress_bar(downloaded, total, width=40):
    pct = downloaded / total if total > 0 else 0
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    mb_down = downloaded / 1024 / 1024
    mb_total = total / 1024 / 1024
    print(f"\r  [{bar}] {mb_down:.1f}/{mb_total:.1f} MB ({pct*100:.1f}%)", end='', flush=True)


def download(key, dataset, output_dir):
    filepath = os.path.join(output_dir, dataset["file"])

    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"  Already exists ({size_mb:.0f} MB) — skipping")
        return

    # Try direct download first
    try:
        print(f"  Downloading from {dataset['url']}...")
        def hook(count, block_size, total_size):
            progress_bar(count * block_size, total_size)
        urllib.request.urlretrieve(dataset["url"], filepath, reporthook=hook)
        print()
    except Exception as e:
        print(f"  Direct download failed ({e})")
        print(f"  Use ann-benchmarks tool instead:")
        print(f"    git clone https://github.com/erikbern/ann-benchmarks.git")
        print(f"    cd ann-benchmarks && pip install -r requirements.txt")
        print(f"    python create_dataset.py --dataset {dataset['file'].replace('.hdf5','')}")

def print_table():
    print("\nAvailable datasets:")
    print(f"{'Key':<15} {'Name':<35} {'Dims':>6} {'Train':>10} {'Distance':<12} {'Size':<8} {'Use'}")
    print("-" * 95)
    for key, d in DATASETS.items():
        use = "✅ ours" if d["our_use"] else ""
        print(f"{key:<15} {d['name']:<35} {d['dims']:>6} {d['train']:>10,} {d['distance']:<12} {d['size']:<8} {use}")
    print()


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Parse --only argument
    only = None
    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        if idx + 1 < len(sys.argv):
            only = sys.argv[idx + 1]

    print_table()

    if only:
        if only not in DATASETS:
            print(f"Unknown dataset: {only}")
            print(f"Available: {', '.join(DATASETS.keys())}")
            sys.exit(1)
        to_download = {only: DATASETS[only]}
    else:
        to_download = DATASETS

    print(f"Downloading {len(to_download)} dataset(s) to {output_dir}/\n")

    for key, dataset in to_download.items():
        print(f"[{key}] {dataset['name']} ({dataset['size']})")
        download(key, dataset, output_dir)
        print()

    print("Done!")
    print("\nTo load a dataset in Python:")
    print("  import h5py")
    print("  f = h5py.File('sift-128-euclidean.hdf5', 'r')")
    print("  train = f['train'][:]   # shape: (1000000, 128)")
    print("  test  = f['test'][:]    # shape: (10000, 128)")
    print("  neighbors = f['neighbors'][:]  # shape: (10000, 100) ground truth")


if __name__ == "__main__":
    main()

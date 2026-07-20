# Crack-IVF Baseline: Setup Log

Documents how the Crack-IVF baseline (github.com/mageirakos/crack-ivf-vldb) was vendored
and made to run inside the HyVec project, on an aarch64 (ARM) Linux machine.

## 1. Vendoring the repo

Cloned directly into `third_party/`, `.git` stripped so it's plain committed source,
not a nested repo or submodule:

```bash
cd HyVec
git clone https://github.com/mageirakos/crack-ivf-vldb.git third_party/crack-ivf
rm -rf third_party/crack-ivf/.git
```

Bundled sample datasets and the authors' own paper result dumps were excluded from git
(regenerated locally, not needed in our history):

```
# added to .gitignore
third_party/crack-ivf/others/AVtree/datasets/
third_party/crack-ivf/main_code/results/
```

## 2. Why this needs its own environment

Crack-IVF is a **separately patched fork of Faiss** (adds mutability to `IndexIVFFlat`),
built via `faiss-crack/install_faiss_mkl_release.sh`. It cannot share a build or a binary
with our own vendored vanilla Faiss (`third_party/faiss/`, v1.14.2) — two different Faiss
builds in one process would clash on symbol names. It runs as a **standalone Python
pipeline**, invoked separately from our C++ benchmark harness.

## 3. Conda environment (aarch64-specific fixes)

Installed Miniforge (not Miniconda — needs native aarch64 conda-forge builds):

```bash
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh
# accept default install location (~/miniforge3), say yes to shell init
source ~/.bashrc
```

The repo's `faiss-crack/crackivf.yml` was written for **linux-64 (x86_64) with Intel MKL**,
neither of which works on aarch64. Fixes applied, in order:

1. **Removed MKL** (`mkl=2024.*`) — Intel doesn't ship MKL for ARM. The build script falls
   back to whatever BLAS/LAPACK it finds via `LD_LIBRARY_PATH`, and the file already listed
   OpenBLAS as well.
2. **Stripped all exact version+build-hash pins** — conda-forge's hashes are platform-specific;
   linux-64 hashes don't exist under linux-aarch64.
   ```bash
   sed -i -E 's/^(\s*- [a-zA-Z0-9_.+-]+)=.*/\1/' crackivf.yml
   ```
3. **Deleted x86_64-named toolchain packages outright** (names, not just hashes, are arch-specific):
   `_libgcc_mutex`, `binutils_impl_linux-64`, `gcc_impl_linux-64`, `gxx_impl_linux-64`,
   `kernel-headers_linux-64`, `sysroot_linux-64`, `libgcc-devel_linux-64`,
   `libstdcxx-devel_linux-64`, `ld_impl_linux-64` — plain `gcc`/`gxx` pull in the correct
   aarch64 toolchain as transitive deps anyway.
4. **Re-pinned `python=3.10`** — stripping all pins let the solver grab Python 3.14, which broke
   compatibility with code written against 3.10 (and numpy 1.26-era APIs).

Result:
```bash
conda env create -f crackivf.yml
conda activate crack-paper-final
```

## 4. Building patched Faiss

```bash
cd third_party/crack-ivf/faiss-crack
chmod +x install_faiss_mkl_release.sh
./install_faiss_mkl_release.sh --python
```

Builds `libfaiss.a` + SWIG Python bindings, installs `faiss==1.9.0` into the conda env.
(The repeated `libcurl.so.4: no version information available` warnings during the build
are harmless — an ABI mismatch between the conda env's libcurl and system cmake, not a
build failure.)

Verified with:
```bash
python -c "import faiss; print(faiss.__version__)"
python -c "import faiss; import numpy as np; idx = faiss.IndexFlatL2(8); idx.add(np.random.rand(10,8).astype('float32')); D,I = idx.search(np.random.rand(1,8).astype('float32'), 3); print(I)"
```

## 5. Dataset setup

`main_code/vasili_helpers.py` hardcoded dataset paths to the original authors' cluster
(`/pub/scratch/{username}/datasets/bigann`). Repointed to our own `data/` folder:

```bash
sed -i 's#/pub/scratch/{username}/datasets/bigann#/home/salemmohammed/Projects/research-projects/HyVec/data/bigann#g' \
    third_party/crack-ivf/main_code/vasili_helpers.py
```

(Other hardcoded paths in the same file — `deep-96-angular`, `vdb-project-data`, and a
different author's username for `sift-128-euclidean` — were left untouched; only fix
those if/when those specific datasets are actually needed.)

Downloaded BigANN SIFT (full 1B base set, sliced to 1M/10M at load time — no smaller
pre-sliced source available from this source):

```bash
mkdir -p data/bigann && cd data/bigann
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz
gunzip bigann_base.bvecs.gz          # ~92GB -> ~123GB decompressed, slow, no progress bar
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz
gunzip bigann_query.bvecs.gz
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz
tar xzvf bigann_gnd.tar.gz
rm bigann_gnd.tar.gz
```

## 6. Code fixes required

- `run_baselines.py` used `np.NaN`, removed in NumPy 2.0 (our env resolved numpy 2.2.6
  since we stripped version pins). Patched:
  ```bash
  sed -i 's/np\.NaN/np.nan/g' third_party/crack-ivf/main_code/run_baselines.py
  ```
- `os.mkdir()` in `run_baselines.py` doesn't create the `results/` parent dir. Created manually:
  ```bash
  mkdir -p third_party/crack-ivf/main_code/results
  ```

## 7. Verified working smoke test

```bash
cd third_party/crack-ivf/main_code
python run_baselines.py --store --index_name BruteForce IVFFlat --dbname SIFT1M \
    --runid smoke_test --nthreads 16 --niter 1 --nprobe 16 --nlist 1000
```

Produces `./results/SIFT1M_default_BASELINES/baseline_results_smoke_test.csv` with real
recall/latency/QPS numbers for both BruteForce (exact, recall=1.0) and IVFFlat
(recall@1=0.935, ~8.4k QPS) — confirming the full chain (patched Faiss, dataset,
harness) works end to end on this machine.

## Known open items

- Full `run_baselines.sh` grid (all 6 datasets × full nprobe/nlist sweep) not yet run —
  smoke test only covered SIFT1M with one nprobe/nlist setting.
- `run_ours.py` (actual Crack-IVF adaptive method) not yet run — only the baseline
  IVFFlat/BruteForce path has been exercised so far.
- Integration into `bench/run_benchmark.cpp`: Crack-IVF stays a standalone Python step
  (shell out or merge CSVs after the fact), not linked into the C++ binary — see reasoning
  in Section 2.

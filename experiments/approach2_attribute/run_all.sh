#!/bin/bash
set -e  # stop if any command fails

echo "================================================"
echo "  Clustered Attributed Vector Search Pipeline"
echo "================================================"

# Step 1: Generate attributes
echo ""
echo ">>> STEP 1: Generate Attributes (K-Means K=1000)"
python3 generate_attributes.py

# Step 2: Compile and build indexes
echo ""
echo ">>> STEP 2: Build Per-Cluster HNSW Indexes"
g++ -O3 -std=c++17 build_indexes.cpp -o build_indexes -I../../hnswlib
./build_indexes

# Step 3: Compile and search
echo ""
echo ">>> STEP 3: Clustered Search"
g++ -O3 -std=c++17 search.cpp -o search -I../../hnswlib
./search

# Step 4: Compile and run dynamic reindex
echo ""
echo ">>> STEP 4: Dynamic Re-clustering (Real-Time)"
g++ -O3 -std=c++17 dynamic_reindex.cpp -o dynamic_reindex -I../../hnswlib -lpthread
./dynamic_reindex

echo ""
echo "================================================"
echo "  All steps complete!"
echo "================================================"

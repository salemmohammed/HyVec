#include "../hnswlib/hnswlib/hnswlib.h"
#include <vector>
#include <map>

int K = 100;  // number of clusters
int dim = 128;
int M = 16;
int ef_construction = 200;

// One index per cluster
std::vector<hnswlib::HierarchicalNSW<float>*> indexes(K);
hnswlib::L2Space space(dim);

// Initialize one index per cluster
for (int c = 0; c < K; c++) {
    indexes[c] = new hnswlib::HierarchicalNSW<float>(
        &space, cluster_sizes[c], M, ef_construction
    );
}

// Insert each vector into its cluster's index
for (int i = 0; i < num_base; i++) {
    int cluster = labels[i];           // which cluster this vector belongs to
    indexes[cluster]->addPoint(base[i].data(), i);  // add to that cluster's index
}
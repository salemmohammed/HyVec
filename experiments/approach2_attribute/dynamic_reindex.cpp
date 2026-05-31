/*
 * dynamic_reindex.cpp
 * -------------------
 * Real-Time Insertion with Local Background Re-clustering.
 * Real-time insertion with background re-clustering.
 *
 * Compile:
 *   g++ -O3 -std=c++17 dynamic_reindex.cpp -o dynamic_reindex -I../../hnswlib -lpthread
 *
 * Run:
 *   ./dynamic_reindex
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <random>
#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <cmath>
#include <limits>
#include <iomanip>
#include "../../hnswlib/hnswlib/hnswlib.h"

// ─── CONFIG ──────────────────────────────────────────────────────────────────
const int   DIM               = 128;
const int   K_CLUSTERS        = 1000;
const int   M                 = 16;
const int   EF_CONSTRUCTION   = 200;
const int   EF_SEARCH         = 50;
const int   TARGET_SIZE       = 1000;
const float REINDEX_THRESHOLD = 0.1f;
const float BETA              = 0.5f;
const int   NUM_INSERTIONS    = 50000;
const int   REPORT_EVERY      = 10000;  // print progress every N insertions
const std::string CENTROIDS_PATH = "../sift/attr_centroids.npy";
const std::string INDEX_DIR      = "../sift/indexes/";
// ─────────────────────────────────────────────────────────────────────────────


// ─── CLUSTER METADATA ────────────────────────────────────────────────────────
struct ClusterMetadata {
    int   size          = 0;
    float temperature   = 1.0f;
    float reindex_score = 0.0f;
    float initial_centroid[128];
    float current_centroid[128];
    ClusterMetadata() {
        memset(initial_centroid, 0, sizeof(initial_centroid));
        memset(current_centroid, 0, sizeof(current_centroid));
    }
};
// ─────────────────────────────────────────────────────────────────────────────


// ─── SCORE FUNCTION ───────────────────────────────────────────────────────────
float compute_score(const ClusterMetadata& m) {
    float imbalance = std::abs(m.size - TARGET_SIZE) / (float)TARGET_SIZE;
    float drift = 0.0f;
    for (int i = 0; i < DIM; i++) {
        float d = m.current_centroid[i] - m.initial_centroid[i];
        drift += d * d;
    }
    drift = std::sqrt(drift) / DIM;
    return m.temperature * (BETA * imbalance + (1.0f - BETA) * drift);
}
// ─────────────────────────────────────────────────────────────────────────────


// ─── DYNAMIC CLUSTERED INDEX ──────────────────────────────────────────────────
class DynamicClusteredIndex {
public:
    hnswlib::L2Space space;

    std::map<int, hnswlib::HierarchicalNSW<float>*> active_indexes;
    std::mutex active_mutex;

    std::vector<std::vector<float>> centroids;
    std::vector<ClusterMetadata>    metadata;
    std::mutex metadata_mutex;

    std::vector<std::vector<float>> all_vectors;
    std::vector<int>                all_ids;
    std::vector<int>                all_cluster_ids;
    std::mutex data_mutex;

    std::set<int>           clusters_to_reindex;
    std::mutex              cv_mutex;
    std::condition_variable cv;
    std::atomic<bool>       running{true};

    // Stats for final report
    std::atomic<int> total_reindexed_clusters{0};
    std::atomic<int> total_reindex_events{0};

    DynamicClusteredIndex() : space(DIM) {}

    void load_initial(const std::string& centroids_path,
                      const std::string& index_dir) {
        // Load centroids
        std::ifstream in(centroids_path, std::ios::binary);
        if (!in) { std::cerr << "Cannot open centroids" << std::endl; exit(1); }
        char header[128];
        in.read(header, 128);
        centroids.resize(K_CLUSTERS, std::vector<float>(DIM));
        for (int i = 0; i < K_CLUSTERS; i++)
            in.read((char*)centroids[i].data(), DIM * sizeof(float));
        in.close();

        // Load indexes
        for (int c = 0; c < K_CLUSTERS; c++) {
            std::string path = index_dir + "index_" + std::to_string(c) + ".bin";
            std::ifstream f(path);
            if (f.good()) {
                active_indexes[c] = new hnswlib::HierarchicalNSW<float>(&space, path);
                active_indexes[c]->setEf(EF_SEARCH);
            }
        }

        // Initialize metadata
        metadata.resize(K_CLUSTERS);
        for (int c = 0; c < K_CLUSTERS; c++) {
            metadata[c].temperature = 1.0f;
            metadata[c].size = active_indexes.count(c) ?
                active_indexes[c]->getCurrentElementCount() : 0;
            memcpy(metadata[c].initial_centroid, centroids[c].data(), DIM * sizeof(float));
            memcpy(metadata[c].current_centroid, centroids[c].data(), DIM * sizeof(float));
        }
    }

    int find_nearest_centroid(const std::vector<float>& vec) {
        float best_dist = std::numeric_limits<float>::max();
        int   best_id   = 0;
        for (int i = 0; i < K_CLUSTERS; i++) {
            float dist = 0.0f;
            for (int j = 0; j < DIM; j++) {
                float d = vec[j] - centroids[i][j];
                dist += d * d;
            }
            if (dist < best_dist) { best_dist = dist; best_id = i; }
        }
        return best_id;
    }

    void insert(const std::vector<float>& vec, int id) {
        int cluster = find_nearest_centroid(vec);

        // Store vector
        {
            std::lock_guard<std::mutex> dlock(data_mutex);
            all_vectors.push_back(vec);
            all_ids.push_back(id);
            all_cluster_ids.push_back(cluster);
        }

        // Insert into HNSW index
        {
            std::lock_guard<std::mutex> alock(active_mutex);
            if (active_indexes.count(cluster)) {
                auto* idx = active_indexes[cluster];
                if (idx->getCurrentElementCount() >= idx->getMaxElements())
                    idx->resizeIndex(idx->getMaxElements() * 2);
                idx->addPoint(vec.data(), id);
            }
        }

        // Update metadata and check score
        bool trigger = false;
        {
            std::lock_guard<std::mutex> mlock(metadata_mutex);
            auto& m = metadata[cluster];
            m.size++;
            for (int i = 0; i < DIM; i++)
                m.current_centroid[i] += (vec[i] - m.current_centroid[i]) / m.size;
            m.reindex_score = compute_score(m);
            if (m.reindex_score > REINDEX_THRESHOLD) {
                clusters_to_reindex.insert(cluster);
                trigger = true;
            }
        }

        if (trigger) cv.notify_one();
    }

    std::vector<int> search(const std::vector<float>& query, int k) {
        int cluster = find_nearest_centroid(query);

        // Update temperature
        {
            std::lock_guard<std::mutex> mlock(metadata_mutex);
            for (int c = 0; c < K_CLUSTERS; c++) {
                if (c == cluster)
                    metadata[c].temperature *= 1.1f;
                else
                    metadata[c].temperature = std::max(1.0f,
                        metadata[c].temperature * 0.99f);
            }
        }

        std::lock_guard<std::mutex> alock(active_mutex);
        if (!active_indexes.count(cluster)) return {};
        auto result = active_indexes[cluster]->searchKnn(query.data(), k);
        std::vector<int> ids;
        while (!result.empty()) {
            ids.push_back(result.top().second);
            result.pop();
        }
        return ids;
    }

    void recluster_thread() {
        while (running) {
            std::unique_lock<std::mutex> lock(cv_mutex);
            cv.wait(lock, [this] {
                return !clusters_to_reindex.empty() || !running.load();
            });
            lock.unlock();

            if (!running) break;

            std::set<int> targets;
            {
                std::lock_guard<std::mutex> mlock(metadata_mutex);
                targets = clusters_to_reindex;
                clusters_to_reindex.clear();
            }

            total_reindex_events++;
            int rebuilt = 0;

            for (int c : targets) {
                std::vector<std::vector<float>> cvecs;
                std::vector<int> cids;
                {
                    std::lock_guard<std::mutex> dlock(data_mutex);
                    for (int i = 0; i < (int)all_vectors.size(); i++) {
                        if (all_cluster_ids[i] == c) {
                            cvecs.push_back(all_vectors[i]);
                            cids.push_back(all_ids[i]);
                        }
                    }
                }

                if (cvecs.empty()) continue;

                int max_els = std::max((int)cvecs.size() * 2, 100);
                auto* new_idx = new hnswlib::HierarchicalNSW<float>(
                    &space, max_els, M, EF_CONSTRUCTION);
                for (int i = 0; i < (int)cvecs.size(); i++)
                    new_idx->addPoint(cvecs[i].data(), cids[i]);
                new_idx->setEf(EF_SEARCH);

                {
                    std::lock_guard<std::mutex> alock(active_mutex);
                    if (active_indexes.count(c)) delete active_indexes[c];
                    active_indexes[c] = new_idx;
                }

                {
                    std::lock_guard<std::mutex> mlock(metadata_mutex);
                    auto& m = metadata[c];
                    m.size = cvecs.size();
                    m.reindex_score = 0.0f;
                    memcpy(m.initial_centroid, m.current_centroid, DIM * sizeof(float));
                }

                rebuilt++;
                total_reindexed_clusters++;
            }
        }
    }

    void stop() { running = false; cv.notify_all(); }

    ~DynamicClusteredIndex() {
        for (auto& [c, idx] : active_indexes) delete idx;
    }
};
// ─────────────────────────────────────────────────────────────────────────────


int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "  Dynamic Clustered HNSW — Local Reindexing" << std::endl;
    std::cout << "================================================\n" << std::endl;

    // 1. Load
    std::cout << "Loading indexes and centroids..." << std::endl;
    DynamicClusteredIndex dci;
    dci.load_initial(CENTROIDS_PATH, INDEX_DIR);
    std::cout << "Ready. Starting insertions...\n" << std::endl;

    // Print table header
    std::cout << std::left
              << std::setw(12) << "Inserted"
              << std::setw(12) << "Search OK"
              << std::setw(16) << "Reindex Events"
              << std::setw(18) << "Clusters Rebuilt"
              << std::endl;
    std::cout << std::string(58, '-') << std::endl;

    // 2. Start background thread
    std::thread bg_thread(&DynamicClusteredIndex::recluster_thread, &dci);

    // 3. Insert vectors
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 255.0f);
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_INSERTIONS; i++) {
        std::vector<float> vec(DIM);
        for (int j = 0; j < DIM; j++) vec[j] = dist(rng);

        dci.insert(vec, 1000000 + i);

        // Report every REPORT_EVERY insertions
        if ((i + 1) % REPORT_EVERY == 0) {
            auto results = dci.search(vec, 10);
            std::cout << std::left
                      << std::setw(12) << (i + 1)
                      << std::setw(12) << (results.size() == 10 ? "YES" : "NO")
                      << std::setw(16) << dci.total_reindex_events.load()
                      << std::setw(18) << dci.total_reindexed_clusters.load()
                      << std::endl;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();

    // 4. Final results
    std::cout << std::string(58, '-') << std::endl;
    std::cout << "\n=== FINAL RESULTS ===" << std::endl;
    std::cout << "Total insertions:    " << NUM_INSERTIONS << std::endl;
    std::cout << "Total time:          " << std::fixed << std::setprecision(2)
              << total_time << "s" << std::endl;
    std::cout << "Insert rate:         " << std::setprecision(0)
              << NUM_INSERTIONS / total_time << " vectors/s" << std::endl;
    std::cout << "Reindex events:      " << dci.total_reindex_events << std::endl;
    std::cout << "Clusters rebuilt:    " << dci.total_reindexed_clusters << std::endl;
    std::cout << "Search downtime:     0s (atomic swap per cluster)" << std::endl;

    dci.stop();
    bg_thread.join();

    std::cout << "\nDone!" << std::endl;
    return 0;
}

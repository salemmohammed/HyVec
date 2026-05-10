/*
 * dynamic_reindex.cpp
 * -------------------
 * Step 4: Real-Time Insertion with Background Re-clustering.
 *
 * Demonstrates:
 *   - inserting new vectors into the clustered index
 *   - background thread re-clusters every N insertions
 *   - atomic swap: search NEVER stops during re-clustering
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
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <condition_variable>
#include "../../hnswlib/hnswlib/hnswlib.h"

// ─── CONFIG ──────────────────────────────────────────────────────────────────
const int DIM               = 128;
const int K_CLUSTERS        = 1000;
const int M                 = 16;
const int EF_CONSTRUCTION   = 200;
const int EF_SEARCH         = 50;
const int RECLUSTER_EVERY   = 10000;   // re-cluster every N insertions
const std::string CENTROIDS_PATH = "../sift/attr_centroids.npy";
const std::string INDEX_DIR      = "../sift/indexes/";
// ─────────────────────────────────────────────────────────────────────────────


// ─── CLUSTERED INDEX CLASS ────────────────────────────────────────────────────
class DynamicClusteredIndex {
public:
    hnswlib::L2Space space;

    // Current active indexes (served to queries)
    std::map<int, hnswlib::HierarchicalNSW<float>*> active_indexes;
    std::vector<std::vector<float>> active_centroids;
    std::mutex active_mutex;    // protects active_indexes and active_centroids

    // All vectors inserted so far (needed for re-clustering)
    std::vector<std::vector<float>> all_vectors;
    std::vector<int> all_ids;
    std::mutex data_mutex;      // protects all_vectors

    // Re-clustering control
    std::atomic<int>  insert_count{0};
    std::atomic<bool> running{true};
    std::atomic<bool> recluster_requested{false};
    std::condition_variable cv;
    std::mutex cv_mutex;

    DynamicClusteredIndex() : space(DIM) {}

    // ── Load initial indexes and centroids ────────────────────────────────────
    void load_initial(const std::string& centroids_path, const std::string& index_dir) {
        std::cout << "Loading initial centroids..." << std::endl;

        // Load centroids
        std::ifstream in(centroids_path, std::ios::binary);
        char header[128];
        in.read(header, 128);
        active_centroids.resize(K_CLUSTERS, std::vector<float>(DIM));
        for (int i = 0; i < K_CLUSTERS; i++) {
            in.read((char*)active_centroids[i].data(), DIM * sizeof(float));
        }
        in.close();

        // Load indexes
        std::cout << "Loading initial indexes..." << std::endl;
        for (int c = 0; c < K_CLUSTERS; c++) {
            std::string path = index_dir + "index_" + std::to_string(c) + ".bin";
            std::ifstream f(path);
            if (f.good()) {
                active_indexes[c] = new hnswlib::HierarchicalNSW<float>(&space, path);
                active_indexes[c]->setEf(EF_SEARCH);
            }
        }
        std::cout << "Loaded " << active_indexes.size() << " indexes." << std::endl;
    }

    // ── Find nearest centroid ─────────────────────────────────────────────────
    int find_nearest_centroid(const std::vector<float>& vec) {
        float best_dist = std::numeric_limits<float>::max();
        int best_id = 0;
        for (int i = 0; i < (int)active_centroids.size(); i++) {
            float dist = 0;
            for (int j = 0; j < DIM; j++) {
                float d = vec[j] - active_centroids[i][j];
                dist += d * d;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_id = i;
            }
        }
        return best_id;
    }

    // ── Insert new vector ─────────────────────────────────────────────────────
    void insert(const std::vector<float>& vec, int id) {
        // Store vector for future re-clustering
        {
            std::lock_guard<std::mutex> lock(data_mutex);
            all_vectors.push_back(vec);
            all_ids.push_back(id);
        }

        // Insert into correct cluster's index
        {
            std::lock_guard<std::mutex> lock(active_mutex);
            int cluster = find_nearest_centroid(vec);
            // NEW - resize if needed
            if (active_indexes.find(cluster) != active_indexes.end()) {
                auto* idx = active_indexes[cluster];
                if (idx->getCurrentElementCount() >= idx->getMaxElements()) {
                    idx->resizeIndex(idx->getMaxElements() * 2);
                }
                idx->addPoint(vec.data(), id);
            }
        }
        
        // Trigger re-clustering if needed
        int count = ++insert_count;
        if (count % RECLUSTER_EVERY == 0) {
            std::cout << "[Main] Re-clustering triggered at "
                      << count << " insertions" << std::endl;
            recluster_requested = true;
            cv.notify_one();
        }
    }

    // ── Search ────────────────────────────────────────────────────────────────
    std::vector<int> search(const std::vector<float>& query, int k) {
        std::lock_guard<std::mutex> lock(active_mutex);

        int cluster = find_nearest_centroid(query);
        if (active_indexes.find(cluster) == active_indexes.end()) return {};

        auto result = active_indexes[cluster]->searchKnn(query.data(), k);
        std::vector<int> ids;
        while (!result.empty()) {
            ids.push_back(result.top().second);
            result.pop();
        }
        return ids;
    }

    // ── Background re-clustering thread ───────────────────────────────────────
    void recluster_thread() {
        std::cout << "[Background] Re-clustering thread started." << std::endl;

        while (running) {
            // Wait for trigger
            std::unique_lock<std::mutex> lock(cv_mutex);
            cv.wait(lock, [this] {
                return recluster_requested.load() || !running.load();
            });
            recluster_requested = false;
            lock.unlock();

            if (!running) break;

            std::cout << "[Background] Starting re-clustering..." << std::endl;
            auto t0 = std::chrono::high_resolution_clock::now();

            // 1. Copy current data safely
            std::vector<std::vector<float>> vectors_copy;
            std::vector<int> ids_copy;
            {
                std::lock_guard<std::mutex> dlock(data_mutex);
                vectors_copy = all_vectors;
                ids_copy     = all_ids;
            }

            int n = vectors_copy.size();
            if (n == 0) continue;

            // 2. Simple re-clustering: reassign to existing centroids
            //    (In production: re-run full K-Means here)
            std::map<int, std::vector<int>> new_cluster_to_ids;
            std::vector<std::vector<float>> new_centroids = active_centroids;

            for (int i = 0; i < n; i++) {
                int c = find_nearest_centroid(vectors_copy[i]);
                new_cluster_to_ids[c].push_back(i);
            }

            // 3. Build new indexes
            std::map<int, hnswlib::HierarchicalNSW<float>*> new_indexes;
            for (auto& [c, ids] : new_cluster_to_ids) {
                if (ids.empty()) continue;
                // Add 50% buffer for future insertions
                int max_elements = std::max((int)ids.size() * 2, 100);
                int max_els = std::max((int)ids.size() * 2, 100);
                auto* idx = new hnswlib::HierarchicalNSW<float>(
                    &space, max_els, M, EF_CONSTRUCTION);
                for (int idx_pos : ids) {
                    idx->addPoint(vectors_copy[idx_pos].data(), ids_copy[idx_pos]);
                }
                idx->setEf(EF_SEARCH);
                new_indexes[c] = idx;
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            double rebuild_time = std::chrono::duration<double>(t1 - t0).count();

            // 4. ATOMIC SWAP — search never stops!
            {
                std::lock_guard<std::mutex> alock(active_mutex);

                // Delete old indexes
                for (auto& [c, idx] : active_indexes) delete idx;
                active_indexes.clear();

                // Activate new indexes
                active_indexes  = new_indexes;
                active_centroids = new_centroids;
            }

            std::cout << "[Background] Re-clustering done in " << rebuild_time
                      << "s. New indexes: " << new_indexes.size()
                      << ". Search continues normally." << std::endl;
        }

        std::cout << "[Background] Re-clustering thread stopped." << std::endl;
    }

    void stop() {
        running = false;
        cv.notify_all();
    }

    ~DynamicClusteredIndex() {
        for (auto& [c, idx] : active_indexes) delete idx;
    }
};
// ─────────────────────────────────────────────────────────────────────────────


int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "  Dynamic Clustered Index with Re-clustering" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "Re-clusters every " << RECLUSTER_EVERY << " insertions" << std::endl;
    std::cout << "Atomic swap: search never stops during re-clustering" << std::endl;

    // 1. Initialize index
    DynamicClusteredIndex dci;
    dci.load_initial(CENTROIDS_PATH, INDEX_DIR);

    // 2. Start background re-clustering thread
    std::thread bg_thread(&DynamicClusteredIndex::recluster_thread, &dci);

    // 3. Simulate real-time insertions
    std::cout << "\nSimulating real-time insertions..." << std::endl;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 255.0f);

    int num_insertions = 50000;
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_insertions; i++) {
        // Generate random vector (in practice: real incoming data)
        std::vector<float> vec(DIM);
        for (int j = 0; j < DIM; j++) vec[j] = dist(rng);

        // Insert into index
        dci.insert(vec, 1000000 + i);  // use IDs beyond original 1M

        // Occasionally do a search to show system stays live
        if (i % 5000 == 0) {
            auto results = dci.search(vec, 10);
            std::cout << "[Main] Inserted " << i
                      << " vectors, search returned "
                      << results.size() << " results" << std::endl;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n=== DYNAMIC INDEX RESULTS ===" << std::endl;
    std::cout << "Total insertions: " << num_insertions << std::endl;
    std::cout << "Total time:       " << total_time << "s" << std::endl;
    std::cout << "Insert rate:      " << num_insertions / total_time << " vectors/s" << std::endl;

    // 4. Stop background thread
    dci.stop();
    bg_thread.join();

    std::cout << "\nDone!" << std::endl;
    return 0;
}

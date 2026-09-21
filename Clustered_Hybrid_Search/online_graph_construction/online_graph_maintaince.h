#pragma once
#include "hnswlib/hnswlib.h"
#include "hnswlib/visited_list_pool.h"
#include <vector>
#include <cstring>
#include <cstdint>
#include <sys/stat.h>
#include <sys/types.h>
#include <map>
#include <omp.h>
namespace Online_Graph_Maintiance
{
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;
    template <typename dist_t>

    class OnlineGraphHNSW : public hnswlib::HierarchicalNSW<dist_t>
    {
        // Comparator
        struct CompareByFirstElement
        {
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept
            {
                return a.first < b.first;
            }
        };
        using Candidate = std::pair<dist_t, tableint>;
        using CandidateQueue = std::priority_queue<Candidate, std::vector<Candidate>, CompareByFirstElement>;

    public:
        std::unordered_map<tableint, std::vector<std::string>> meta_data_predicates;
        std::unique_ptr<hnswlib::VisitedListPool> visited_list_pool_{nullptr};
        // Buffered memory for online graph construction

        std::unordered_map<std::string, std::unordered_set<tableint>> attribute_to_node_map;
        std::vector<std::vector<tableint>> online_links_;
        size_t max_elements_;
        char *memory_for_Vectors;
        uint8_t *bloom_filter_;

    public:
        OnlineGraphHNSW(hnswlib::SpaceInterface<dist_t> *space, size_t max_elements, const std::string &location_of_index, std::unordered_map<tableint, std::vector<std::string>> &meta_data_predicates_)
            : hnswlib::HierarchicalNSW<dist_t>(space, max_elements)
        {

            this->loadIndex(location_of_index, space, max_elements);
            meta_data_predicates = meta_data_predicates_;
            max_elements_ = max_elements;
            visited_list_pool_ = std::unique_ptr<hnswlib::VisitedListPool>(new hnswlib::VisitedListPool(1, max_elements_));
            size_t num_bytes = (max_elements_ + 7) / 8;
            memory_for_Vectors = new char[num_bytes];
            std::memset(memory_for_Vectors, 0, num_bytes);
            bloom_filter_ = new uint8_t[max_elements_];
            std::memset(bloom_filter_, 0, max_elements_);
            online_links_.resize(max_elements_);
        }

    public
        void searchAndUpdate(const void *query_data, size_t query_number, std::vector<std::string> &query_attribute)
        {
            // Implement the search and update logic here query_data, top_k,
            auto search_results = this->searchKnnForOnlineGraphConstruction(query_data);
            while (!search_results.empty())
            {
                auto [dist, id] = search_results.top();
                search_results.pop();

                if (query_attribute[0] == meta_data_predicates[id][0])
                {
                    // Update the attribute_to_node_map for the retrieved id
                    attribute_to_node_map[query_attribute[0]].insert(id);
                }

                // Update the Bloom filter for the retrieved id
            }
        }
        // This method is used to update the Bloom filter
        inline void bloomFilterUpdate(size_t node_index, std::string &query_attribute)
        {
            // Set the bit in the memory_for_Vectors array
            memory_for_Vectors[node_index >> 3] |= (1u << (node_index & 7));
            size_t h = std::hash<std::string>{}(query_attribute);

            uint8_t h1 = h & 7; // hash % 8
            uint8_t h2 = (h >> 3) & 7;
            uint8_t h3 = (h >> 6) & 7;

            // Set the three Bloom filter bits
            bloom_filter_[node_index] |= (1u << h1);
            bloom_filter_[node_index] |= (1u << h2);
            bloom_filter_[node_index] |= (1u << h3);
        }
        // This method is used to retrieve the Bloom filter

        inline bool bloomFilterContains(size_t node_index,
                                        const std::string &query_attribute) const
        {
            // Step 1: Does this node even have a Bloom filter?
            if ((memory_for_Vectors[node_index >> 3] &
                 (1u << (node_index & 7))) == 0)
            {
                return false;
            }

            // Step 2: Compute the same hash values
            size_t h = std::hash<std::string>{}(query_attribute);

            uint8_t h1 = h & 7;
            uint8_t h2 = (h >> 3) & 7;
            uint8_t h3 = (h >> 6) & 7;

            // Step 3: Check whether all three bits are set
            uint8_t bloom = bloom_filter_[node_index];

            return ((bloom & (1u << h1)) &&
                    (bloom & (1u << h2)) &&
                    (bloom & (1u << h3)));
        }

    public:
        void graphUpdate(const std::string &attribute, int M_att)
        {
            // Find the attribute
            auto it = attribute_to_node_map.find(attribute);
            if (it == attribute_to_node_map.end())
                return;

            // Copy for safe iteration
            auto node_set_copy = it->second;

            // Process every node that originally belongs to this attribute
            for (tableint u : node_set_copy)
            {
                // Update bloom filter
                bloomFilterUpdate(u, attribute);

                // Expand the candidate set using existing online neighbors
                if (!online_links_[u].empty())
                {
                    it->second.insert(
                        online_links_[u].begin(),
                        online_links_[u].end());
                }

                // Max-heap: keeps the M_att closest neighbors
                std::priority_queue<
                    std::pair<dist_t, tableint>,
                    std::vector<std::pair<dist_t, tableint>>,
                    std::less<std::pair<dist_t, tableint>>>
                    nearest_neighbors;

                // Use the UPDATED candidate set
                for (tableint v : it->second)
                {
                    if (u == v)
                        continue;

                    dist_t dist = this->fstdistfunc_(
                        this->getDataByInternalId(u),
                        this->getDataByInternalId(v),
                        this->dist_func_param_);

                    if ((int)nearest_neighbors.size() < M_att)
                    {
                        nearest_neighbors.emplace(dist, v);
                    }
                    else if (dist < nearest_neighbors.top().first)
                    {
                        nearest_neighbors.pop();
                        nearest_neighbors.emplace(dist, v);
                    }
                }

                // Replace the old online neighbors
                online_links_[u].clear();

                while (!nearest_neighbors.empty())
                {
                    online_links_[u].push_back(nearest_neighbors.top().second);
                    nearest_neighbors.pop();
                }
            }
        }
    };
}

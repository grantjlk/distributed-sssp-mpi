#include "local_graph.h"
#include "csr.h"
#include <fstream>
#include <stdexcept>

int32_t block_begin(int64_t num_nodes, int num_ranks, int rank) {
    return (int32_t)(num_nodes * rank / num_ranks);
}

// inverse of block_begin: the largest r with block_begin(r) <= node
int block_owner(int64_t num_nodes, int num_ranks, int32_t node) {
    return (int)(((int64_t)(node + 1) * num_ranks - 1) / num_nodes);
}

template <typename T>
static void read_at(std::ifstream& in, int64_t pos, std::vector<T>& v, int64_t count) {
    v.resize(count);
    in.seekg(pos);
    in.read(reinterpret_cast<char*>(v.data()), count * sizeof(T));
}

LocalGraph load_graph_local(const std::string& path, int rank, int num_ranks) {
    CsrHeader h = read_csr_header(path);
    if (num_ranks > h.num_nodes) {
        throw std::runtime_error("More ranks than nodes in " + path);
    }

    LocalGraph lg;
    lg.num_nodes = h.num_nodes;
    lg.num_edges = h.num_edges;
    lg.rank      = rank;
    lg.num_ranks = num_ranks;
    lg.first     = block_begin(h.num_nodes, num_ranks, rank);
    lg.last      = block_begin(h.num_nodes, num_ranks, rank + 1);

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open CSR file: " + path);
    }

    // offsets for owned nodes, plus one more to know where the last row ends
    read_at(in, csr_offsets_pos() + (int64_t)lg.first * sizeof(int64_t),
            lg.offsets, lg.num_owned() + 1);

    int64_t edge_begin = lg.offsets.front();
    int64_t edge_count = lg.offsets.back() - edge_begin;

    read_at(in, csr_targets_pos(h.num_nodes) + edge_begin * (int64_t)sizeof(int32_t),
            lg.targets, edge_count);
    read_at(in, csr_weights_pos(h.num_nodes, h.num_edges) + edge_begin * (int64_t)sizeof(int32_t),
            lg.weights, edge_count);

    if (!in) {
        throw std::runtime_error("CSR file is truncated: " + path);
    }

    // rebase so local row i spans [offsets[i], offsets[i+1]) in the local arrays
    for (auto& o : lg.offsets) {
        o -= edge_begin;
    }

    return lg;
}

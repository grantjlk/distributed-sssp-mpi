#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Block partition: rank r owns global node ids [block_begin(r), block_begin(r+1)).
// Computed from (n, num_ranks) alone, so no partition file is needed.
int32_t block_begin(int64_t num_nodes, int num_ranks, int rank);
int     block_owner(int64_t num_nodes, int num_ranks, int32_t node);

// One rank's share of a CSR graph: only the edges leaving the nodes it owns.
// Edge targets stay as global ids because they may point at remote nodes.
struct LocalGraph {
    int64_t num_nodes = 0;   // global node count
    int64_t num_edges = 0;   // global directed edge count
    int rank = 0;
    int num_ranks = 1;
    int32_t first = 0;       // first owned global id
    int32_t last  = 0;       // one past the last owned global id

    std::vector<int64_t> offsets;  // size num_owned()+1, starts at 0
    std::vector<int32_t> targets;  // global ids
    std::vector<int32_t> weights;

    int32_t num_owned() const { return last - first; }
    int64_t num_local_edges() const { return (int64_t)targets.size(); }
    bool owns(int32_t v) const { return v >= first && v < last; }
    int owner(int32_t v) const { return block_owner(num_nodes, num_ranks, v); }
    int32_t to_local(int32_t v) const { return v - first; }
    int32_t to_global(int32_t local) const { return local + first; }
};

// Reads only this rank's rows of a CSR file (no MPI calls inside)
LocalGraph load_graph_local(const std::string& path, int rank, int num_ranks);

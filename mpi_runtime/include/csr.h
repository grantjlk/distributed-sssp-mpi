#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Binary CSR (compressed sparse row) graph file.
//
// Layout (little-endian):
//   char    magic[8]        "CSRGRAF1"
//   int64   n               number of nodes
//   int64   m               number of directed edges
//   int64   offsets[n + 1]  edges of node u are [offsets[u], offsets[u+1])
//   int32   targets[m]      destination node of each edge
//   int32   weights[m]      positive weight of each edge
//
// Because each node's edges are contiguous, a rank that owns a range of
// nodes can read just its slice of targets/weights.

struct CsrGraph {
    int64_t num_nodes = 0;
    int64_t num_edges = 0;
    std::vector<int64_t> offsets;
    std::vector<int32_t> targets;
    std::vector<int32_t> weights;
};

// Header fields read without loading the edge arrays
struct CsrHeader {
    int64_t num_nodes;
    int64_t num_edges;
};

void write_csr(const std::string& path, const CsrGraph& g);

// Read the whole file (used by tests and the sequential checker)
CsrGraph read_csr(const std::string& path);

CsrHeader read_csr_header(const std::string& path);

// byte offsets of each section, used for partial reads
int64_t csr_offsets_pos();
int64_t csr_targets_pos(int64_t num_nodes);
int64_t csr_weights_pos(int64_t num_nodes, int64_t num_edges);

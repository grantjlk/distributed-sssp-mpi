#include "csr.h"
#include <cstring>
#include <fstream>
#include <stdexcept>

static const char CSR_MAGIC[8] = {'C', 'S', 'R', 'G', 'R', 'A', 'F', '1'};

int64_t csr_offsets_pos() {
    return sizeof(CSR_MAGIC) + 2 * sizeof(int64_t);
}

int64_t csr_targets_pos(int64_t num_nodes) {
    return csr_offsets_pos() + (num_nodes + 1) * (int64_t)sizeof(int64_t);
}

int64_t csr_weights_pos(int64_t num_nodes, int64_t num_edges) {
    return csr_targets_pos(num_nodes) + num_edges * (int64_t)sizeof(int32_t);
}

template <typename T>
static void write_array(std::ofstream& out, const std::vector<T>& v) {
    out.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
}

template <typename T>
static void read_array(std::ifstream& in, std::vector<T>& v, int64_t count) {
    v.resize(count);
    in.read(reinterpret_cast<char*>(v.data()), count * sizeof(T));
}

void write_csr(const std::string& path, const CsrGraph& g) {
    if ((int64_t)g.offsets.size() != g.num_nodes + 1 ||
        (int64_t)g.targets.size() != g.num_edges ||
        (int64_t)g.weights.size() != g.num_edges) {
        throw std::runtime_error("CSR arrays do not match num_nodes/num_edges");
    }

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Could not open CSR file for writing: " + path);
    }

    out.write(CSR_MAGIC, sizeof(CSR_MAGIC));
    out.write(reinterpret_cast<const char*>(&g.num_nodes), sizeof(int64_t));
    out.write(reinterpret_cast<const char*>(&g.num_edges), sizeof(int64_t));
    write_array(out, g.offsets);
    write_array(out, g.targets);
    write_array(out, g.weights);

    if (!out) {
        throw std::runtime_error("Failed while writing CSR file: " + path);
    }
}

// opens the file and checks the magic tag, leaving the stream after the header
static std::ifstream open_csr(const std::string& path, CsrHeader& h) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open CSR file: " + path);
    }

    char magic[8];
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, CSR_MAGIC, sizeof(magic)) != 0) {
        throw std::runtime_error("Not a CSR graph file: " + path);
    }

    in.read(reinterpret_cast<char*>(&h.num_nodes), sizeof(int64_t));
    in.read(reinterpret_cast<char*>(&h.num_edges), sizeof(int64_t));
    if (!in || h.num_nodes < 0 || h.num_edges < 0) {
        throw std::runtime_error("Corrupt CSR header: " + path);
    }
    return in;
}

CsrHeader read_csr_header(const std::string& path) {
    CsrHeader h;
    open_csr(path, h);
    return h;
}

CsrGraph read_csr(const std::string& path) {
    CsrHeader h;
    std::ifstream in = open_csr(path, h);

    CsrGraph g;
    g.num_nodes = h.num_nodes;
    g.num_edges = h.num_edges;
    read_array(in, g.offsets, g.num_nodes + 1);
    read_array(in, g.targets, g.num_edges);
    read_array(in, g.weights, g.num_edges);

    if (!in) {
        throw std::runtime_error("CSR file is truncated: " + path);
    }
    if (g.offsets.front() != 0 || g.offsets.back() != g.num_edges) {
        throw std::runtime_error("CSR offsets are inconsistent: " + path);
    }
    return g;
}

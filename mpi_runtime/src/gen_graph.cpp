// Synthetic graph generator. Writes a binary CSR file (see csr.h).
//
//   gen_graph --type rmat --scale 20 [--edge-factor 16] --out g.csr
//   gen_graph --type grid --rows 1000 --cols 1000 --out grid.csr
//
// Common options: --max-weight W (weights are uniform in [1, W]), --seed S
//
// Both graph types are undirected: every edge is stored in both directions
// with the same weight.

#include "csr.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// splitmix64: tiny, fast, and gives the same stream on every platform
// (std:: distributions are implementation-defined, so seeds wouldn't be portable)
struct Rng {
    uint64_t state;

    uint64_t next() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    // uniform double in [0, 1)
    double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }

    // uniform integer in [lo, hi]
    int32_t range(int32_t lo, int32_t hi) {
        return lo + (int32_t)(next() % (uint64_t)(hi - lo + 1));
    }
};

struct WEdge {
    int32_t src;
    int32_t dst;
    int32_t weight;
};

// sorts edges, drops duplicate (src, dst) pairs keeping the lightest,
// and packs the result into CSR
static CsrGraph build_csr(int64_t n, std::vector<WEdge>& edges) {
    std::sort(edges.begin(), edges.end(), [](const WEdge& x, const WEdge& y) {
        if (x.src != y.src) return x.src < y.src;
        if (x.dst != y.dst) return x.dst < y.dst;
        return x.weight < y.weight;
    });
    auto last = std::unique(edges.begin(), edges.end(), [](const WEdge& x, const WEdge& y) {
        return x.src == y.src && x.dst == y.dst;
    });
    edges.erase(last, edges.end());

    CsrGraph g;
    g.num_nodes = n;
    g.num_edges = (int64_t)edges.size();
    g.offsets.assign(n + 1, 0);
    g.targets.resize(edges.size());
    g.weights.resize(edges.size());

    for (size_t i = 0; i < edges.size(); i++) {
        g.offsets[edges[i].src + 1]++;
        g.targets[i] = edges[i].dst;
        g.weights[i] = edges[i].weight;
    }
    for (int64_t u = 0; u < n; u++) {
        g.offsets[u + 1] += g.offsets[u];
    }
    return g;
}

static CsrGraph gen_rmat(int scale, int edge_factor, int max_weight, Rng& rng) {
    // Graph500 R-MAT quadrant probabilities (d = 1 - a - b - c = 0.05)
    const double a = 0.57, b = 0.19, c = 0.19;

    int64_t n = int64_t(1) << scale;
    int64_t samples = n * edge_factor;

    // random relabeling so high-degree nodes aren't clustered at low ids,
    // which would overload whichever rank owns the first block
    std::vector<int32_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    for (int64_t i = n - 1; i > 0; i--) {
        std::swap(perm[i], perm[rng.next() % (uint64_t)(i + 1)]);
    }

    std::vector<WEdge> edges;
    edges.reserve(2 * samples);

    for (int64_t e = 0; e < samples; e++) {
        int64_t u = 0, v = 0;
        // pick one quadrant of the adjacency matrix per bit of the node id
        for (int bit = 0; bit < scale; bit++) {
            double r = rng.uniform();
            if (r < a) {
                // top-left: neither bit set
            } else if (r < a + b) {
                v |= int64_t(1) << bit;
            } else if (r < a + b + c) {
                u |= int64_t(1) << bit;
            } else {
                u |= int64_t(1) << bit;
                v |= int64_t(1) << bit;
            }
        }
        if (u == v) continue;  // drop self-loops

        int32_t w = rng.range(1, max_weight);
        edges.push_back({perm[u], perm[v], w});
        edges.push_back({perm[v], perm[u], w});
    }

    return build_csr(n, edges);
}

static CsrGraph gen_grid(int rows, int cols, int max_weight, Rng& rng) {
    int64_t n = (int64_t)rows * cols;
    std::vector<WEdge> edges;
    edges.reserve(4 * n);

    // row-major ids, so a block partition gives each rank a strip of rows
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int32_t u = r * cols + c;
            if (c + 1 < cols) {
                int32_t w = rng.range(1, max_weight);
                edges.push_back({u, u + 1, w});
                edges.push_back({u + 1, u, w});
            }
            if (r + 1 < rows) {
                int32_t w = rng.range(1, max_weight);
                edges.push_back({u, u + cols, w});
                edges.push_back({u + cols, u, w});
            }
        }
    }

    return build_csr(n, edges);
}

static void print_stats(const CsrGraph& g) {
    int64_t max_degree = 0, isolated = 0;
    for (int64_t u = 0; u < g.num_nodes; u++) {
        int64_t d = g.offsets[u + 1] - g.offsets[u];
        max_degree = std::max(max_degree, d);
        if (d == 0) isolated++;
    }
    std::cout << "  nodes:          " << g.num_nodes << "\n"
              << "  directed edges: " << g.num_edges << "\n"
              << "  avg degree:     " << (double)g.num_edges / g.num_nodes << "\n"
              << "  max degree:     " << max_degree << "\n"
              << "  isolated nodes: " << isolated << "\n";
}

static void usage() {
    std::cerr << "Usage:\n"
              << "  gen_graph --type rmat --scale S [--edge-factor 16] --out FILE\n"
              << "  gen_graph --type grid --rows R --cols C --out FILE\n"
              << "Options: --max-weight W (default 100), --seed S (default 1)\n";
}

int main(int argc, char* argv[]) {
    std::string type, out;
    int scale = 0, edge_factor = 16, rows = 0, cols = 0, max_weight = 100;
    uint64_t seed = 1;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (i + 1 >= argc) { usage(); return 1; }
        if (arg == "--type")             type        = argv[++i];
        else if (arg == "--out")         out         = argv[++i];
        else if (arg == "--scale")       scale       = std::stoi(argv[++i]);
        else if (arg == "--edge-factor") edge_factor = std::stoi(argv[++i]);
        else if (arg == "--rows")        rows        = std::stoi(argv[++i]);
        else if (arg == "--cols")        cols        = std::stoi(argv[++i]);
        else if (arg == "--max-weight")  max_weight  = std::stoi(argv[++i]);
        else if (arg == "--seed")        seed        = std::stoull(argv[++i]);
        else { usage(); return 1; }
    }

    if (out.empty() || max_weight < 1) { usage(); return 1; }

    Rng rng{seed};
    CsrGraph g;

    if (type == "rmat") {
        // node ids are int32, so cap at 2^30 nodes
        if (scale < 1 || scale > 30 || edge_factor < 1) { usage(); return 1; }
        g = gen_rmat(scale, edge_factor, max_weight, rng);
    } else if (type == "grid") {
        if (rows < 1 || cols < 1 || (int64_t)rows * cols > INT32_MAX) { usage(); return 1; }
        g = gen_grid(rows, cols, max_weight, rng);
    } else {
        usage();
        return 1;
    }

    write_csr(out, g);
    std::cout << "[gen_graph] wrote " << type << " graph (seed " << seed << ") to " << out << "\n";
    print_stats(g);
    return 0;
}

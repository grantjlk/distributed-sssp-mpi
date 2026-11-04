#include "sssp_reference.h"
#include <functional>
#include <queue>
#include <utility>

std::vector<int64_t> sequential_sssp(const CsrGraph& g, int32_t source) {
    std::vector<int64_t> dist(g.num_nodes, SSSP_INF);

    // min-heap of (distance, node); outdated entries are skipped when popped
    using Item = std::pair<int64_t, int32_t>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;

    dist[source] = 0;
    heap.push({0, source});

    while (!heap.empty()) {
        auto [d, u] = heap.top();
        heap.pop();
        if (d > dist[u]) continue;

        for (int64_t e = g.offsets[u]; e < g.offsets[u + 1]; e++) {
            int32_t v = g.targets[e];
            int64_t nd = d + g.weights[e];
            if (nd < dist[v]) {
                dist[v] = nd;
                heap.push({nd, v});
            }
        }
    }
    return dist;
}

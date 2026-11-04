#include <gtest/gtest.h>
#include <fstream>
#include "graph.h"
#include "partition.h"
#include "csr.h"
#include "local_graph.h"
#include "sssp_reference.h"
#include <algorithm>
#include <cstdio>

// ─── Graph Tests ───────────────────────────────────────────────────────────

TEST(GraphTest, LoadsCorrectly) {
    Graph g = load_graph("tests/small_graph.json");
    EXPECT_EQ(g.num_nodes, 5);
    EXPECT_EQ(g.edges.size(), 10);
}

TEST(GraphTest, AdjacencyListCorrect) {
    Graph g = load_graph("tests/small_graph.json");
    // node 0 should have edges to 1 and 3
    EXPECT_EQ(g.adj[0].size(), 2);
}

TEST(GraphTest, AllWeightsPositive) {
    Graph g = load_graph("tests/small_graph.json");
    for (const auto& e : g.edges) {
        EXPECT_GT(e.weight, 0) << "Edge " << e.src 
                               << "->" << e.dst 
                               << " has non-positive weight";
    }
}

TEST(GraphTest, NodeIdsInRange) {
    Graph g = load_graph("tests/small_graph.json");
    for (const auto& e : g.edges) {
        EXPECT_GE(e.src, 0);
        EXPECT_LT(e.src, g.num_nodes);
        EXPECT_GE(e.dst, 0);
        EXPECT_LT(e.dst, g.num_nodes);
    }
}

// ─── Partition Tests ────────────────────────────────────────────────────────

TEST(PartitionTest, LoadsCorrectly) {
    Partition p = load_partition("tests/small_part.json");
    EXPECT_EQ(p.num_ranks, 4);
    EXPECT_EQ(p.num_nodes, 5);
}

TEST(PartitionTest, AllNodesAssigned) {
    Partition p = load_partition("tests/small_part.json");
    EXPECT_EQ((int)p.ownership.size(), p.num_nodes);
    for (int i = 0; i < p.num_nodes; i++) {
        EXPECT_GE(p.ownership[i], 0);
        EXPECT_LT(p.ownership[i], p.num_ranks) 
            << "Node " << i << " assigned to invalid rank";
    }
}

TEST(PartitionTest, OwnedNodesCorrect) {
    Partition p = load_partition("tests/small_part.json");
    // rank 0 owns nodes 0 and 4
    std::vector<int> owned = get_owned_nodes(p, 0);
    EXPECT_EQ(owned.size(), 2);
}

// ─── CSR Format Tests ───────────────────────────────────────────────────────

// 0 -> {1, 2}, 1 -> {2}, 2 -> {}
static CsrGraph tiny_csr() {
    CsrGraph g;
    g.num_nodes = 3;
    g.num_edges = 3;
    g.offsets = {0, 2, 3, 3};
    g.targets = {1, 2, 2};
    g.weights = {5, 9, 1};
    return g;
}

TEST(CsrTest, RoundTrip) {
    CsrGraph g = tiny_csr();
    write_csr("csr_roundtrip.bin", g);
    CsrGraph r = read_csr("csr_roundtrip.bin");

    EXPECT_EQ(r.num_nodes, g.num_nodes);
    EXPECT_EQ(r.num_edges, g.num_edges);
    EXPECT_EQ(r.offsets, g.offsets);
    EXPECT_EQ(r.targets, g.targets);
    EXPECT_EQ(r.weights, g.weights);

    CsrHeader h = read_csr_header("csr_roundtrip.bin");
    EXPECT_EQ(h.num_nodes, 3);
    EXPECT_EQ(h.num_edges, 3);
    std::remove("csr_roundtrip.bin");
}

TEST(CsrTest, RejectsNonCsrFile) {
    EXPECT_THROW(read_csr("tests/small_graph.json"), std::runtime_error);
}

// ─── Local Graph Tests ──────────────────────────────────────────────────────

TEST(BlockPartitionTest, OwnerMatchesBlockRanges) {
    // every node lands in exactly the block that block_begin describes,
    // including uneven splits and one node per rank
    for (int64_t n : {1, 5, 7, 10, 64, 1000}) {
        for (int ranks = 1; ranks <= std::min<int64_t>(n, 12); ranks++) {
            for (int r = 0; r < ranks; r++) {
                for (int32_t v = block_begin(n, ranks, r); v < block_begin(n, ranks, r + 1); v++) {
                    EXPECT_EQ(block_owner(n, ranks, v), r)
                        << "n=" << n << " ranks=" << ranks << " v=" << v;
                }
            }
            EXPECT_EQ(block_begin(n, ranks, ranks), n);
        }
    }
}

TEST(LocalGraphTest, SlicesCoverWholeGraph) {
    // 5 nodes: 0 -> {1, 2}, 1 -> {}, 2 -> {0}, 3 -> {4}, 4 -> {3, 0}
    CsrGraph g;
    g.num_nodes = 5;
    g.num_edges = 6;
    g.offsets = {0, 2, 2, 3, 4, 6};
    g.targets = {1, 2, 0, 4, 3, 0};
    g.weights = {1, 2, 3, 4, 5, 6};
    write_csr("local_slices.bin", g);

    for (int ranks = 1; ranks <= 5; ranks++) {
        int64_t total_edges = 0;
        for (int r = 0; r < ranks; r++) {
            LocalGraph lg = load_graph_local("local_slices.bin", r, ranks);
            total_edges += lg.num_local_edges();

            // each owned node's local row must equal its row in the full graph
            for (int32_t i = 0; i < lg.num_owned(); i++) {
                int32_t u = lg.to_global(i);
                ASSERT_EQ(lg.offsets[i + 1] - lg.offsets[i], g.offsets[u + 1] - g.offsets[u]);
                for (int64_t k = 0; k < lg.offsets[i + 1] - lg.offsets[i]; k++) {
                    EXPECT_EQ(lg.targets[lg.offsets[i] + k], g.targets[g.offsets[u] + k]);
                    EXPECT_EQ(lg.weights[lg.offsets[i] + k], g.weights[g.offsets[u] + k]);
                }
            }
        }
        EXPECT_EQ(total_edges, g.num_edges) << "ranks=" << ranks;
    }
    std::remove("local_slices.bin");
}

// ─── Reference SSSP Tests ───────────────────────────────────────────────────

TEST(ReferenceSsspTest, ShortestPathsAndUnreachable) {
    // 0 -> 1 (4), 0 -> 2 (1), 2 -> 1 (2), 1 -> 3 (5); node 4 is unreachable
    CsrGraph g;
    g.num_nodes = 5;
    g.num_edges = 4;
    g.offsets = {0, 2, 3, 4, 4, 4};
    g.targets = {1, 2, 3, 1};
    g.weights = {4, 1, 5, 2};

    std::vector<int64_t> dist = sequential_sssp(g, 0);
    EXPECT_EQ(dist[0], 0);
    EXPECT_EQ(dist[1], 3);   // 0 -> 2 -> 1 beats the direct edge
    EXPECT_EQ(dist[2], 1);
    EXPECT_EQ(dist[3], 8);
    EXPECT_EQ(dist[4], SSSP_INF);
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}